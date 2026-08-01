"""End-to-end EdgeTAM → DINOv3 → MobileCLIP2 offline CUDA pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.model_registry import LocalModelRegistry, ModelId
from visionforge.observability.gpu_metrics import empty_cuda_cache, reset_peak_stats, snapshot_gpu
from visionforge.pipeline.annotate import encode_annotated_mp4
from visionforge.pipeline.features import (
    l2_normalize,
    pairwise_cosine,
    summarize_similarities,
)
from visionforge.pipeline.geometry import (
    extract_masked_crop,
    overlay_mask,
    scale_bounding_box,
    validate_bounding_box,
)
from visionforge.pipeline.io import (
    FrameSequence,
    create_run_directory,
    load_frame_sequence,
    maybe_resize_frames,
    read_rgb,
)
from visionforge.pipeline.manifest import bytes_to_mb, write_json
from visionforge.pipeline.progress import open_progress, overall_percent_for
from visionforge.pipeline.quality import (
    DEFAULT_THRESHOLDS,
    detect_sequence_issues,
    diagnose_mask,
)
from visionforge.pipeline.recovery import (
    evaluate_recovery,
    recovery_thresholds_for_manifest,
)
from visionforge.tracking import (
    require_available,
    select_tracker_backend,
)


@dataclass
class PipelineConfig:
    input_path: Path
    box_xyxy: tuple[float, float, float, float]
    text_labels: list[str]
    output_root: Path
    tracker: str = "edgetam"
    max_frames: int | None = None
    start_frame: int = 0
    process_width: int | None = None
    process_height: int | None = None
    allow_existing: bool = False
    run_id: str | None = None
    progress_file: Path | None = None
    parent_job_id: str | None = None
    revision_id: str | None = None
    analysis_mode: str = "full"
    selection_mode: str = "manual"


@dataclass
class PipelineResult:
    run_dir: Path
    manifest_path: Path
    manifest: dict = field(default_factory=dict)


class PipelineError(Exception):
    pass


def _require_cuda() -> str:
    try:
        import torch
    except ImportError as exc:
        raise PipelineError("PyTorch is not installed") from exc
    if not torch.cuda.is_available():
        raise PipelineError("CUDA is required for the native Windows E2E pipeline")
    return torch.cuda.get_device_name(0)


def _validate_models() -> dict[str, str]:
    reg = LocalModelRegistry()
    paths = {}
    for mid in (ModelId.EDGETAM, ModelId.DINOV3_VITS16, ModelId.MOBILECLIP2_S0, ModelId.SAM31):
        pkg = reg.validate(mid)
        paths[mid.value] = str(pkg.primary_checkpoint)
    return paths


def _release() -> None:
    import gc

    import torch

    gc.collect()
    empty_cuda_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    labels = [str(t).strip() for t in (config.text_labels or []) if str(t).strip()]

    progress = open_progress(config.progress_file)
    progress.emit(
        "validating",
        overall_percent=overall_percent_for("validating", 0.0),
    )
    gpu_name = _require_cuda()
    model_paths = _validate_models()

    backend = select_tracker_backend(config.tracker)
    # Explicit capability recording — never silent fallback.
    sam_cap = select_tracker_backend("sam31").capability()
    edge_cap = select_tracker_backend("edgetam").capability()
    require_available(backend)
    progress.emit(
        "validating",
        overall_percent=overall_percent_for("validating", 1.0),
    )

    run_dir, run_id = create_run_directory(
        config.output_root,
        allow_existing=config.allow_existing,
        run_id=config.run_id,
    )
    frames_work = run_dir / "frames_proc"
    masks_dir = run_dir / "masks"
    overlays_dir = run_dir / "overlays"
    crops_dir = run_dir / "crops"
    for d in (frames_work, masks_dir, overlays_dir, crops_dir):
        d.mkdir(parents=True, exist_ok=True)

    t_pipeline0 = time.perf_counter()
    warnings: list[str] = []

    # ---- Input ----
    progress.emit(
        "preparing_input",
        overall_percent=overall_percent_for("preparing_input", 0.0),
    )
    seq = load_frame_sequence(
        config.input_path,
        max_frames=config.max_frames,
        start_frame=config.start_frame,
        extract_dir=run_dir / "video_frames",
    )
    original_box = validate_bounding_box(
        config.box_xyxy,
        image_width=seq.width,
        image_height=seq.height,
    )
    proc_seq, _sx, _sy = maybe_resize_frames(
        seq,
        process_width=config.process_width,
        process_height=config.process_height,
        out_dir=frames_work,
    )
    if config.process_width and config.process_height:
        proc_box = scale_bounding_box(
            original_box,
            src_width=seq.width,
            src_height=seq.height,
            dst_width=proc_seq.width,
            dst_height=proc_seq.height,
        )
        # Use resized frames directory as tracker input
        tracker_frames = FrameSequence(
            source=seq.source,
            frame_paths=proc_seq.frame_paths,
            width=proc_seq.width,
            height=proc_seq.height,
            is_extracted_video=seq.is_extracted_video,
        )
    else:
        # Copy/symlink not required — use original frame paths in a dedicated dir listing
        # EdgeTAM expects a directory of JPGs; if source is already a dir, use it.
        if seq.source.is_dir() and not seq.is_extracted_video:
            # Materialize ordered copies into frames_work for stable input
            for i, fp in enumerate(seq.frame_paths):
                dest = frames_work / f"{i:05d}.jpg"
                if not dest.exists():
                    Image.open(fp).convert("RGB").save(dest, quality=95)
            tracker_frames = FrameSequence(
                source=seq.source,
                frame_paths=[frames_work / f"{i:05d}.jpg" for i in range(len(seq.frame_paths))],
                width=seq.width,
                height=seq.height,
            )
        else:
            tracker_frames = seq
            # Ensure frames live under a directory EdgeTAM can open
            if seq.is_extracted_video:
                tracker_frames = FrameSequence(
                    source=seq.source,
                    frame_paths=seq.frame_paths,
                    width=seq.width,
                    height=seq.height,
                    is_extracted_video=True,
                )
        proc_box = original_box

    # Ensure tracker input directory
    tracker_input_dir = tracker_frames.frame_paths[0].parent
    n_frames = len(tracker_frames.frame_paths)
    progress.emit(
        "preparing_input",
        completed=n_frames,
        total=n_frames,
        overall_percent=overall_percent_for("preparing_input", 1.0),
    )

    # ---- Stage 1: Tracker ----
    progress.emit(
        "tracking",
        completed=0,
        total=n_frames,
        overall_percent=overall_percent_for("tracking", 0.0),
    )
    stage_track: dict = {"name": "tracker", "backend": backend.tracker_id.value}
    _release()
    reset_peak_stats()
    backend.load()
    track = backend.track(
        tracker_input_dir,
        box_xyxy=proc_box.as_tuple(),
        frame_width=tracker_frames.width,
        frame_height=tracker_frames.height,
        object_id=1,
    )
    progress.emit(
        "tracking",
        completed=n_frames,
        total=n_frames,
        overall_percent=overall_percent_for("tracking", 1.0),
    )
    stage_track.update(
        {
            "load_time_sec": track.load_time_sec,
            "inference_time_sec": track.inference_time_sec,
            "peak_cuda_allocated_mb": bytes_to_mb(track.peak_allocated_bytes),
            "peak_cuda_reserved_mb": bytes_to_mb(track.peak_reserved_bytes),
            "used_real_cuda": track.used_real_cuda,
            "warnings": track.warnings,
            "checkpoint_path": track.checkpoint_path,
        }
    )
    warnings.extend(track.warnings)
    backend.close()
    _release()

    # Save masks / overlays / crops with quality diagnostics
    progress.emit(
        "creating_artifacts",
        completed=0,
        total=n_frames,
        overall_percent=overall_percent_for("creating_artifacts", 0.0),
    )
    frame_records = []
    valid_crops: list[np.ndarray] = []
    valid_frame_indices: list[int] = []
    mask_diagnostics = []
    mask_paths_ordered: list[Path | None] = []

    by_idx = {f.frame_index: f for f in track.frames}
    for i, fp in enumerate(tracker_frames.frame_paths):
        rgb = read_rgb(fp)
        fr = by_idx.get(i)
        rec: dict = {
            "frame_index": i,
            "frame_path": str(fp),
            "success": False,
        }
        if fr is None:
            rec["error"] = "missing_tracker_output"
            frame_records.append(rec)
            mask_paths_ordered.append(None)
            mask_diagnostics.append(
                diagnose_mask(
                    np.zeros((tracker_frames.height, tracker_frames.width), dtype=bool),
                    frame_index=i,
                    frame_w=tracker_frames.width,
                    frame_h=tracker_frames.height,
                    box_xyxy=proc_box.as_tuple(),
                )
            )
            continue
        if fr.mask.shape != (tracker_frames.height, tracker_frames.width):
            rec["error"] = (
                f"mask_shape_mismatch:{fr.mask.shape}!="
                f"{(tracker_frames.height, tracker_frames.width)}"
            )
            frame_records.append(rec)
            mask_paths_ordered.append(None)
            mask_diagnostics.append(
                diagnose_mask(
                    np.zeros((tracker_frames.height, tracker_frames.width), dtype=bool),
                    frame_index=i,
                    frame_w=tracker_frames.width,
                    frame_h=tracker_frames.height,
                    box_xyxy=proc_box.as_tuple(),
                )
            )
            continue

        diag = diagnose_mask(
            fr.mask,
            frame_index=i,
            frame_w=tracker_frames.width,
            frame_h=tracker_frames.height,
            box_xyxy=proc_box.as_tuple() if i == 0 else None,
        )
        mask_diagnostics.append(diag)
        mask_path = masks_dir / f"mask_{i:05d}.png"
        Image.fromarray((fr.mask.astype(np.uint8) * 255), mode="L").save(mask_path)
        mask_paths_ordered.append(mask_path)
        rec["mask_path"] = str(mask_path)
        rec["mask_area_px"] = diag.area_px
        rec["mask_valid"] = diag.valid
        rec["mask_reasons"] = diag.reasons

        if not fr.valid or not diag.valid:
            rec["error"] = fr.error or (",".join(diag.reasons) if diag.reasons else "invalid_mask")
            frame_records.append(rec)
            continue

        overlay = overlay_mask(rgb, fr.mask)
        overlay_path = overlays_dir / f"overlay_{i:05d}.jpg"
        Image.fromarray(overlay).save(overlay_path, quality=90)
        crop = extract_masked_crop(rgb, fr.mask)
        crop_path = crops_dir / f"crop_{i:05d}.jpg"
        Image.fromarray(crop.crop_rgb).save(crop_path, quality=95)
        rec.update(
            {
                "success": True,
                "overlay_path": str(overlay_path),
                "crop_path": str(crop_path),
                "crop_xyxy": [crop.x1, crop.y1, crop.x2, crop.y2],
            }
        )
        frame_records.append(rec)
        valid_crops.append(crop.crop_rgb)
        valid_frame_indices.append(i)
        progress.emit(
            "creating_artifacts",
            completed=i + 1,
            total=n_frames,
            overall_percent=overall_percent_for(
                "creating_artifacts", (i + 1) / max(1, n_frames)
            ),
        )

    if not valid_crops:
        raise PipelineError("No valid non-empty masks/crops produced by tracker")

    progress.emit(
        "validating_masks",
        overall_percent=overall_percent_for("validating_masks", 0.5),
    )
    # Preliminary quality (DINOv3 drop filled later)
    quality_pre = detect_sequence_issues(
        mask_diagnostics,
        frame_w=tracker_frames.width,
        frame_h=tracker_frames.height,
    )
    write_json(run_dir / "quality_report.json", quality_pre)
    progress.emit(
        "validating_masks",
        overall_percent=overall_percent_for("validating_masks", 1.0),
    )

    # Annotated video (primary product output)
    progress.emit(
        "encoding_video",
        overall_percent=overall_percent_for("encoding_video", 0.0),
    )
    annotated_path = run_dir / "annotated.mp4"
    source_video = Path(config.input_path) if Path(config.input_path).is_file() else None
    if source_video and source_video.suffix.lower() not in {
        ".mp4",
        ".mov",
        ".webm",
        ".avi",
        ".mkv",
        ".m4v",
    }:
        source_video = None
    annotated_meta = encode_annotated_mp4(
        frame_paths=list(tracker_frames.frame_paths),
        mask_paths=mask_paths_ordered,
        output_path=annotated_path,
        fps=float(getattr(seq, "fps", None) or 24.0),
        source_video=source_video,
        audio_start_sec=0.0,
    )
    progress.emit(
        "encoding_video",
        completed=annotated_meta["frames"],
        total=n_frames,
        overall_percent=overall_percent_for("encoding_video", 1.0),
    )

    # ---- Stage 2: DINOv3 ----
    from visionforge.model_adapters.dinov3_adapter import DINOv3Adapter

    progress.emit(
        "dinov3",
        completed=0,
        total=len(valid_crops),
        overall_percent=overall_percent_for("dinov3", 0.0),
    )
    stage_dino: dict = {"name": "dinov3"}
    dino_pkg = LocalModelRegistry().validate(ModelId.DINOV3_VITS16)
    _release()
    reset_peak_stats()
    dino = DINOv3Adapter(dino_pkg.package_dir, device="cuda")
    t_load0 = time.perf_counter()
    dino.load()
    load_dino = time.perf_counter() - t_load0
    emb_list = []
    infer_dino = 0.0
    peak_a = 0
    peak_r = 0
    t_inf0 = time.perf_counter()
    for di, crop_img in enumerate(valid_crops):
        # Avoid per-call peak reset dominating; measure wall + snapshot after loop
        result = dino.encode_rgb_image(crop_img)
        if not result.finite:
            raise PipelineError("DINOv3 produced non-finite embedding")
        vec = np.asarray(result.embedding, dtype=np.float32).reshape(-1)
        if vec.size != 384:
            raise PipelineError(f"Unexpected DINOv3 dim {vec.shape}")
        emb_list.append(vec.reshape(384))
        if result.peak_allocated_bytes:
            peak_a = max(peak_a, result.peak_allocated_bytes)
        if result.peak_reserved_bytes:
            peak_r = max(peak_r, result.peak_reserved_bytes)
        progress.emit(
            "dinov3",
            completed=di + 1,
            total=len(valid_crops),
            overall_percent=overall_percent_for(
                "dinov3", (di + 1) / max(1, len(valid_crops))
            ),
        )
    infer_dino = time.perf_counter() - t_inf0
    snap = snapshot_gpu()
    peak_a = max(peak_a, snap.max_allocated_bytes or 0)
    peak_r = max(peak_r, snap.max_reserved_bytes or 0)
    features = l2_normalize(np.stack(emb_list, axis=0))
    vs_first, consecutive = pairwise_cosine(features)
    identity = {
        "vs_first_frame": [float(x) for x in vs_first.tolist()],
        "consecutive": [float(x) for x in consecutive.tolist()],
        "vs_first_summary": summarize_similarities(vs_first),
        "consecutive_summary": summarize_similarities(
            consecutive[1:] if len(consecutive) > 1 else consecutive
        ),
        "valid_frame_indices": valid_frame_indices,
        "note": (
            "Diagnostic identity-consistency similarities only; "
            "no fixed threshold applied."
        ),
    }
    emb_path = run_dir / "dinov3_embeddings.npy"
    np.save(emb_path, features)
    write_json(run_dir / "identity_similarities.json", identity)
    stage_dino.update(
        {
            "load_time_sec": round(load_dino, 4),
            "inference_time_sec": round(infer_dino, 4),
            "peak_cuda_allocated_mb": bytes_to_mb(peak_a),
            "peak_cuda_reserved_mb": bytes_to_mb(peak_r),
            "checkpoint_path": str(dino_pkg.primary_checkpoint),
            "feature_shape": list(features.shape),
            "used_real_cuda": True,
            "embeddings_path": str(emb_path),
        }
    )
    dino.close()
    _release()

    # ---- Stage 3: MobileCLIP2 (only when user supplied labels) ----
    stage_mclip: dict = {"name": "mobileclip2", "skipped": False}
    mean_scores: dict = {}
    best_label = None
    img_np = None
    txt_np = None
    if labels:
        from visionforge.model_adapters.mobileclip2_adapter import MobileCLIP2Adapter

        progress.emit(
            "mobileclip2",
            completed=0,
            total=len(valid_crops),
            overall_percent=overall_percent_for("mobileclip2", 0.0),
        )
        mclip_pkg = LocalModelRegistry().validate(ModelId.MOBILECLIP2_S0)
        _release()
        reset_peak_stats()
        mclip = MobileCLIP2Adapter(mclip_pkg.primary_checkpoint, device="cuda")
        t_load0 = time.perf_counter()
        mclip.load()
        load_m = time.perf_counter() - t_load0

        import torch

        image_feats = []
        t_inf0 = time.perf_counter()
        with torch.inference_mode():
            text_tokens = mclip._tokenizer(labels).to(mclip.device)
            text_features = mclip._model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            for crop_img in valid_crops:
                pil = Image.fromarray(crop_img.astype(np.uint8), mode="RGB")
                tensor = mclip._preprocess(pil).unsqueeze(0).to(mclip.device)
                feat = mclip._model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                image_feats.append(feat)
            image_mat = torch.cat(image_feats, dim=0)
            sims = (image_mat @ text_features.T).float().cpu().numpy()
            img_np = image_mat.float().cpu().numpy()
            txt_np = text_features.float().cpu().numpy()
        infer_m = time.perf_counter() - t_inf0
        snap = snapshot_gpu()
        if not (
            np.isfinite(img_np).all()
            and np.isfinite(txt_np).all()
            and np.isfinite(sims).all()
        ):
            raise PipelineError("MobileCLIP2 produced non-finite values")

        per_frame_scores = []
        for i, fi in enumerate(valid_frame_indices):
            per_frame_scores.append(
                {
                    "frame_index": fi,
                    "scores": {labels[j]: float(sims[i, j]) for j in range(len(labels))},
                }
            )
        mean_scores = {labels[j]: float(sims[:, j].mean()) for j in range(len(labels))}
        best_label = max(mean_scores.items(), key=lambda kv: kv[1])[0]
        mclip_payload = {
            "labels": labels,
            "image_feature_shape": list(img_np.shape),
            "text_feature_shape": list(txt_np.shape),
            "per_frame_scores": per_frame_scores,
            "mean_scores": mean_scores,
            "highest_scoring_aggregate_label": best_label,
            "valid_crops_used": len(valid_crops),
            "aggregation": "mean_over_valid_crops",
            "note": (
                "These scores express similarity between the tracked visual result "
                "and the labels supplied for this job. They are not guaranteed "
                "detections or classifications."
            ),
        }
        write_json(run_dir / "mobileclip2_similarities.json", mclip_payload)
        np.save(run_dir / "mobileclip2_image_embeddings.npy", img_np.astype(np.float32))
        np.save(run_dir / "mobileclip2_text_embeddings.npy", txt_np.astype(np.float32))
        stage_mclip.update(
            {
                "load_time_sec": round(load_m, 4),
                "inference_time_sec": round(infer_m, 4),
                "peak_cuda_allocated_mb": bytes_to_mb(snap.max_allocated_bytes),
                "peak_cuda_reserved_mb": bytes_to_mb(snap.max_reserved_bytes),
                "checkpoint_path": str(mclip_pkg.primary_checkpoint),
                "used_real_cuda": True,
            }
        )
        mclip.close()
        _release()
        progress.emit(
            "mobileclip2",
            completed=len(valid_crops),
            total=len(valid_crops),
            overall_percent=overall_percent_for("mobileclip2", 1.0),
        )
    else:
        stage_mclip["skipped"] = True
        stage_mclip["reason"] = "no_user_labels"
        progress.emit(
            "mobileclip2",
            overall_percent=overall_percent_for("mobileclip2", 1.0),
            message="Skipped — no user-supplied labels",
        )

    # Conservative identity recovery for invalid / drifted frames
    progress.emit(
        "recovering_identity",
        overall_percent=overall_percent_for("recovering_identity", 0.0),
    )
    recovery_decisions: list[dict] = []
    recovered_count = 0
    lost_count = 0
    dino_recovery = None
    try:
        from visionforge.detection import get_default_detector
        from visionforge.model_adapters.dinov3_adapter import DINOv3Adapter

        detector = get_default_detector()
        det_ok = detector.capabilities().status == "AVAILABLE"
        ref_feat = features[0] if len(features) else None
        last_feat = ref_feat
        last_box = proc_box.as_tuple()
        last_valid_i = 0
        invalid_indices = [d.frame_index for d in mask_diagnostics if not d.valid]
        if invalid_indices and ref_feat is not None and det_ok:
            dino_pkg = LocalModelRegistry().validate(ModelId.DINOV3_VITS16)
            dino_recovery = DINOv3Adapter(dino_pkg.package_dir, device="cuda")
            dino_recovery.load()
        for di, diag in enumerate(mask_diagnostics):
            if diag.valid and diag.bbox_xyxy is not None:
                last_box = (
                    float(diag.bbox_xyxy[0]),
                    float(diag.bbox_xyxy[1]),
                    float(diag.bbox_xyxy[2]),
                    float(diag.bbox_xyxy[3]),
                )
                if di in valid_frame_indices and ref_feat is not None:
                    vi = valid_frame_indices.index(di)
                    last_feat = features[vi]
                    last_valid_i = di
                continue
            if ref_feat is None or not det_ok or dino_recovery is None:
                recovery_decisions.append(
                    {
                        "frame_index": diag.frame_index,
                        "recovered": False,
                        "lost": True,
                        "require_review": True,
                        "reasons": ["no_reference_or_detector"],
                    }
                )
                lost_count += 1
                continue
            rgb = read_rgb(tracker_frames.frame_paths[diag.frame_index])
            cands = detector.detect(rgb, text_prompt=None)
            boxes = [c.box_xyxy for c in cands]
            cand_feats: list = []
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(rgb.shape[1], max(x1 + 1, x2))
                y2 = min(rgb.shape[0], max(y1 + 1, y2))
                crop_rgb = rgb[y1:y2, x1:x2]
                if crop_rgb.size == 0:
                    cand_feats.append(None)
                    continue
                try:
                    enc = dino_recovery.encode_rgb_image(crop_rgb)
                    cand_feats.append(
                        np.asarray(enc.embedding, dtype=np.float32).reshape(-1)
                    )
                except Exception:
                    cand_feats.append(None)
            decision = evaluate_recovery(
                frame_index=diag.frame_index,
                frame_w=tracker_frames.width,
                frame_h=tracker_frames.height,
                last_valid_box=last_box,
                reference_feature=ref_feat,
                last_feature=last_feat,
                candidate_boxes=boxes,
                candidate_features=cand_feats,
            )
            payload = {
                "frame_index": decision.frame_index,
                "recovered": decision.recovered,
                "lost": decision.lost,
                "require_review": decision.require_review,
                "chosen_box": list(decision.chosen_box) if decision.chosen_box else None,
                "chosen_score": decision.chosen_score,
                "candidates_considered": decision.candidates_considered,
                "reasons": decision.reasons,
                "candidate_scores": decision.candidate_scores[:12],
                "last_valid_frame": last_valid_i,
            }
            recovery_decisions.append(payload)
            if decision.recovered:
                recovered_count += 1
                if decision.chosen_box:
                    last_box = decision.chosen_box
            else:
                lost_count += 1
        if dino_recovery is not None:
            try:
                dino_recovery.close()
            except Exception:
                pass
            _release()
    except Exception as exc:
        warnings.append(f"recovery_pass_error:{type(exc).__name__}")
    write_json(
        run_dir / "recovery_decisions.json",
        {
            "decisions": recovery_decisions,
            "recovered_frames": recovered_count,
            "lost_frames": lost_count,
            "thresholds": recovery_thresholds_for_manifest(),
            "note": (
                "Conservative detector+DINOv3 recovery heuristics. "
                "Recovered frames are marked explicitly; ambiguous cases require review."
            ),
        },
    )
    progress.emit(
        "recovering_identity",
        overall_percent=overall_percent_for("recovering_identity", 1.0),
    )

    # Final quality with DINOv3 signals
    quality = detect_sequence_issues(
        mask_diagnostics,
        frame_w=tracker_frames.width,
        frame_h=tracker_frames.height,
        dino_vs_first=identity.get("vs_first_frame"),
        valid_frame_indices=valid_frame_indices,
    )
    quality["recovery_attempts"] = len(recovery_decisions)
    quality["recovered_frames"] = recovered_count
    quality["lost_frames"] = lost_count
    if lost_count > 0 and quality["recommended_status"] == "succeeded":
        quality["recommended_status"] = "review_required"
        quality["warnings"] = list(quality.get("warnings") or []) + [
            "identity_loss_requires_review"
        ]
    write_json(run_dir / "quality_report.json", quality)

    total_s = time.perf_counter() - t_pipeline0
    successful = sum(1 for r in frame_records if r.get("success"))
    failed = [r for r in frame_records if not r.get("success")]
    files_masks = len(list(masks_dir.glob("mask_*.png")))
    files_overlays = len(list(overlays_dir.glob("overlay_*.jpg")))
    files_crops = len(list(crops_dir.glob("crop_*.jpg")))

    progress.emit(
        "finalizing",
        overall_percent=overall_percent_for("finalizing", 0.5),
    )
    manifest = {
        "run_id": run_id,
        "selected_tracker_backend": backend.tracker_id.value,
        "tracker_capabilities": {
            "edgetam": {
                "status": edge_cap.status.value,
                "detail": edge_cap.detail,
            },
            "sam31": {
                "status": sam_cap.status.value,
                "detail": sam_cap.detail,
            },
        },
        "model_checkpoint_paths": model_paths,
        "input": {
            "path": str(Path(config.input_path).resolve()),
            "identity": str(seq.source),
            "num_source_frames": len(seq.frame_paths),
            "source_resolution": [seq.width, seq.height],
            "processed_resolution": [tracker_frames.width, tracker_frames.height],
            "max_frames": config.max_frames,
            "start_frame": config.start_frame,
            "analysis_mode": config.analysis_mode,
            "selection_mode": config.selection_mode,
        },
        "bounding_box_original_xyxy": list(original_box.as_tuple()),
        "bounding_box_processed_xyxy": list(proc_box.as_tuple()),
        "text_labels": labels,
        "frames": {
            "processed": len(frame_records),
            "successful": successful,
            "failed": failed,
            "records": frame_records,
        },
        "artifact_inventory": {
            "mask_files": files_masks,
            "valid_masks": quality["valid_masks"],
            "invalid_masks": quality["invalid_masks"],
            "empty_masks": quality["empty_masks"],
            "overlay_files": files_overlays,
            "crop_files": files_crops,
            "annotated_video": annotated_meta.get("path"),
        },
        "quality": quality,
        "recommended_status": quality["recommended_status"],
        "annotated_video": annotated_meta,
        "outputs": {
            "run_dir": str(run_dir),
            "masks_dir": str(masks_dir),
            "overlays_dir": str(overlays_dir),
            "crops_dir": str(crops_dir),
            "annotated_mp4": str(annotated_path),
            "dinov3_embeddings": str(emb_path),
            "identity_similarities": str(run_dir / "identity_similarities.json"),
            "mobileclip2_similarities": str(run_dir / "mobileclip2_similarities.json")
            if labels
            else None,
            "quality_report": str(run_dir / "quality_report.json"),
        },
        "stages": {
            "tracker": stage_track,
            "dinov3": stage_dino,
            "mobileclip2": stage_mclip,
            "encoding_video": annotated_meta,
        },
        "identity_summary": identity,
        "mobileclip2_summary": {
            "mean_scores": mean_scores,
            "highest_scoring_aggregate_label": best_label,
            "image_feature_shape": list(img_np.shape) if img_np is not None else None,
            "text_feature_shape": list(txt_np.shape) if txt_np is not None else None,
            "valid_crops_used": len(valid_crops) if labels else 0,
            "skipped": not bool(labels),
        },
        "timing": {
            "total_pipeline_sec": round(total_s, 4),
        },
        "gpu": {"name": gpu_name},
        "real_cuda_inference": True,
        "sam31_capability_status": sam_cap.status.value,
        "offline_local_only": True,
        "warnings": warnings + quality.get("warnings", []),
        "mock_or_fallback_used": False,
        "quality_thresholds": DEFAULT_THRESHOLDS,
        "recovery": {
            "attempts": len(recovery_decisions),
            "recovered_frames": recovered_count,
            "lost_frames": lost_count,
            "thresholds": recovery_thresholds_for_manifest(),
        },
        "manual_reinitializations": 1 if config.parent_job_id else 0,
        "parent_job_id": config.parent_job_id,
        "revision_id": config.revision_id,
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(run_dir / "run_metadata.json", {
        "run_id": run_id,
        "tracker": backend.tracker_id.value,
        "created": True,
        "recommended_status": quality["recommended_status"],
    })
    progress.emit(
        "completed",
        completed=successful,
        total=n_frames,
        overall_percent=100.0,
    )
    return PipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)
