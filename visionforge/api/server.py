"""Uvicorn entrypoint for the VisionForge local job API."""
from __future__ import annotations

import argparse
import ipaddress
import sys

import uvicorn

from visionforge.api.app import create_app
from visionforge.api.config import ApiSettings


def _is_loopback(host: str) -> bool:
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VisionForge local FastAPI job service")
    p.add_argument("--host", default=None, help="Bind host (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None)
    p.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Explicitly allow non-loopback bind (no auth in this step)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = ApiSettings.from_env()
    if args.host:
        settings.host = args.host
    if args.port is not None:
        settings.port = args.port
    if args.allow_non_loopback:
        settings.allow_non_loopback = True

    if not _is_loopback(settings.host) and not settings.allow_non_loopback:
        print(
            "Refusing to bind non-loopback host without --allow-non-loopback. "
            "Default bind is 127.0.0.1 only.",
            file=sys.stderr,
        )
        return 2
    if settings.uvicorn_workers != 1:
        print("uvicorn workers must be 1", file=sys.stderr)
        return 2

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
