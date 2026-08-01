#!/usr/bin/env python3
"""WSL-side SAM 3.1 multiplex tracker worker (JSON in/out).

Invoked by Windows VisionForge bridge. Offline-only: local checkpoint required.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _configure_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/mnt/d/caches/huggingface")
    os.environ.setdefault("TORCH_HOME", "/mnt/d/caches/torch")
    os.environ.setdefault("TMPDIR", "/mnt/d/caches/tmp")
    os.environ.setdefault("PIP_CACHE_DIR", "/mnt/d/caches/pip")


def _fail(msg: str, *, code: int = 2) -> None:
    payload = {"ok": False, "error": msg}
    print(json.dumps(payload), flush=True)
    raise SystemExit(code)


def _audit_checkpoint(path: Path) -> dict:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    size = path.stat().st_size
    if size < 1_000_000_000:
        raise ValueError(f"checkpoint too small: {size} bytes")
    # Offline load — map to CPU for structure audit only.
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    info: dict = {
        "path": str(path),
        "size_bytes": size,
        "type": type(obj).__name__,
    }
    if isinstance(obj, dict):
        info["top_keys"] = sorted(str(k) for k in list(obj.keys())[:40])
        # Common checkpoint layouts
        for key in ("model", "state_dict", "model_state_dict", "predictor"):
            if key in obj and isinstance(obj[key], dict):
                info[f"{key}_n_tensors"] = len(obj[key])
                break
        else:
            # Flat state dict?
            tensorish = sum(
                1
                for v in obj.values()
                if hasattr(v, "shape") or (isinstance(v, dict) and v)
            )
            info["approx_tensor_entries"] = tensorish
    del obj
    return info


def _run_track(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    ckpt = Path(args.checkpoint)
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable inside WSL")

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(ckpt),
        compile=False,
        use_fa3=False,
        use_rope_real=False,
        async_loading_frames=False,
    )
    # Upstream Sam3BasePredictor.start_session passes offload_state_to_cpu /
    # video_loader_type, but multiplex init_state does not accept them yet.
    _orig_init = predictor.model.init_state

    def _init_state_compat(*a, **kw):
        kw.pop("offload_state_to_cpu", None)
        vlt = kw.pop("video_loader_type", None)
        if vlt == "torchcodec":
            kw.setdefault("use_torchcodec", True)
        elif vlt == "cv2":
            kw.setdefault("use_cv2", True)
        return _orig_init(*a, **kw)

    predictor.model.init_state = _init_state_compat  # type: ignore[method-assign]
    # Short clips (e.g. 8-frame smoke) must not wait for default hotstart_delay=15.
    for attr, val in (
        ("hotstart_delay", 0),
        ("hotstart_unmatch_thresh", 0),
        ("hotstart_dup_thresh", 0),
    ):
        if hasattr(predictor.model, attr):
            setattr(predictor.model, attr, val)
    load_sec = time.perf_counter() - t0

    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.jpeg"))
    if not frames:
        raise FileNotFoundError(f"no JPEG frames in {frames_dir}")
    with Image.open(frames[0]) as im:
        width, height = im.size

    t1 = time.perf_counter()
    resp = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(frames_dir),
            "offload_video_to_cpu": True,
        }
    )
    session_id = resp.get("session_id")
    if not session_id:
        raise RuntimeError(f"start_session failed: {resp}")

    box = [float(x) for x in args.box]
    if len(box) != 4:
        raise ValueError("box must be x0 y0 x1 y1")
    # Prefer interactive SAM2 center-click for single-object box tracking
    # (matches notebook PoL). Also send normalized XYWH box as secondary path
    # if the point path yields empty masks.
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2.0) / width
    cy = ((y0 + y1) / 2.0) / height
    prompt_resp = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "obj_id": int(args.object_id),
            "points": torch.tensor([[cx, cy]], dtype=torch.float32),
            "point_labels": torch.tensor([1], dtype=torch.int32),
            "rel_coordinates": True,
        }
    )
    prompt_outputs = (prompt_resp or {}).get("outputs") or {}
    prompt_masks = prompt_outputs.get("out_binary_masks")
    pm = np.asarray(prompt_masks) if prompt_masks is not None else np.zeros((0,))
    used_prompt = "center_point"
    if pm.size == 0 or not np.asarray(pm).any():
        rel_xywh = [
            x0 / width,
            y0 / height,
            (x1 - x0) / width,
            (y1 - y0) / height,
        ]
        if min(rel_xywh[2], rel_xywh[3]) <= 0:
            raise ValueError(f"invalid box xyxy={box}")
        prompt_resp = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "obj_id": int(args.object_id),
                "bounding_boxes": torch.tensor([rel_xywh], dtype=torch.float32),
                "bounding_box_labels": torch.tensor([1], dtype=torch.int32),
                "rel_coordinates": True,
            }
        )
        prompt_outputs = (prompt_resp or {}).get("outputs") or {}
        prompt_masks = prompt_outputs.get("out_binary_masks")
        pm = np.asarray(prompt_masks) if prompt_masks is not None else np.zeros((0,))
        used_prompt = "box_xywh"
        if pm.size == 0 or not np.asarray(pm).any():
            raise RuntimeError(
                "add_prompt returned empty masks for center-point and box(xywh); "
                f"prompt_resp_keys={list(prompt_resp)}"
            )

    frame_results = []
    for response in predictor.handle_stream_request(
        {"type": "propagate_in_video", "session_id": session_id}
    ):
        frame_index = int(response["frame_index"])
        outputs = response.get("outputs") or {}
        binary = outputs.get("out_binary_masks")
        obj_ids = outputs.get("out_obj_ids")
        if binary is None:
            binary = outputs.get("binary_masks") or outputs.get("masks")
        if obj_ids is None:
            obj_ids = outputs.get("obj_ids") or outputs.get("object_ids")
        if binary is None:
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "object_id": int(args.object_id),
                    "valid": False,
                    "error": "missing_masks",
                    "mask_path": None,
                }
            )
            continue

        binary_arr = np.asarray(binary)
        if binary_arr.ndim == 2:
            binary_arr = binary_arr[None, ...]
        if obj_ids is None:
            obj_ids = list(range(1, len(binary_arr) + 1))
        obj_ids = [int(x) for x in list(obj_ids)]

        chosen = None
        for i, oid in enumerate(obj_ids):
            if oid == int(args.object_id):
                chosen = binary_arr[i]
                break
        if chosen is None:
            if len(binary_arr) == 0:
                frame_results.append(
                    {
                        "frame_index": frame_index,
                        "object_id": int(args.object_id),
                        "valid": False,
                        "error": "empty_mask_batch",
                        "mask_path": None,
                    }
                )
                continue
            chosen = binary_arr[0]
        if hasattr(chosen, "detach"):
            chosen = chosen.detach().cpu().numpy()
        mask = np.asarray(chosen).astype(bool)
        if mask.ndim != 2:
            raise RuntimeError(f"unexpected mask ndim={mask.ndim}")
        valid = bool(mask.any())
        mask_path = masks_dir / f"mask_{frame_index:05d}.npy"
        np.save(mask_path, mask.astype(np.uint8))
        frame_results.append(
            {
                "frame_index": frame_index,
                "object_id": int(args.object_id),
                "valid": valid,
                "error": None if valid else "empty_mask",
                "mask_path": str(mask_path),
                "shape": list(mask.shape),
            }
        )

    infer_sec = time.perf_counter() - t1
    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    # Best-effort session close
    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception:
        pass

    return {
        "ok": True,
        "tracker_id": "sam31",
        "checkpoint_path": str(ckpt),
        "frames_dir": str(frames_dir),
        "out_dir": str(out_dir),
        "frame_width": width,
        "frame_height": height,
        "load_time_sec": load_sec,
        "inference_time_sec": infer_sec,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "used_real_cuda": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "prompt_response_keys": sorted(prompt_resp.keys())
        if isinstance(prompt_resp, dict)
        else [],
        "frames": frame_results,
        "warnings": [
            "use_fa3=False",
            "use_rope_real=False",
            "compile=False",
            "HF_HUB_OFFLINE=1",
            "hotstart_delay=0",
            f"prompt={used_prompt}",
        ],
    }


def main() -> None:
    _configure_offline_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "audit", "track"), required=True)
    parser.add_argument(
        "--checkpoint",
        default="/mnt/d/project/models/sam31/sam3.1_multiplex.pt",
    )
    parser.add_argument("--frames-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--box", nargs=4, type=float, default=None)
    parser.add_argument("--object-id", type=int, default=1)
    args = parser.parse_args()

    try:
        if args.mode == "probe":
            import sam3
            import torch
            import triton
            from sam3.model_builder import build_sam3_multiplex_video_predictor

            payload = {
                "ok": True,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "triton": triton.__version__,
                "sam3": getattr(sam3, "__version__", "unknown"),
                "sam3_file": sam3.__file__,
                "builder": build_sam3_multiplex_video_predictor.__name__,
                "checkpoint_exists": Path(args.checkpoint).is_file(),
            }
            print(json.dumps(payload), flush=True)
            return

        if args.mode == "audit":
            info = _audit_checkpoint(Path(args.checkpoint))
            print(json.dumps({"ok": True, "audit": info}), flush=True)
            return

        # track
        if not args.frames_dir or not args.out_dir or args.box is None:
            _fail("track requires --frames-dir --out-dir --box x0 y0 x1 y1")
        result = _run_track(args)
        print(json.dumps(result), flush=True)
    except SystemExit:
        raise
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            ),
            flush=True,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
