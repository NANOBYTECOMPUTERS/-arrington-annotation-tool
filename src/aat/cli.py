"""Main CLI for Arrington Annotation Tool (AAT)."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="aat",
        description="Arrington Annotation Tool (AAT) — modular AI-assisted YOLO annotation",
        epilog="Examples:\n  aat generate --model best.pt --images ./raw --out ./dataset --conf 0.2\n  aat view",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", aliases=["gen"], help="Generate YOLO detect dataset from a model/engine")
    gen.add_argument("--model", "-m", required=True)
    gen.add_argument("--images", "-i", required=True)
    gen.add_argument("--out", "-o", required=True)
    gen.add_argument("--conf", type=float, default=0.25)
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(func=lambda args: _cmd_generate(args))

    view = subparsers.add_parser("view", aliases=["viewer"], help="Launch AAT Viewer")
    view.set_defaults(func=lambda args: _cmd_view(args))

    info = subparsers.add_parser("info", help="Show version and info")
    info.set_defaults(func=lambda args: _cmd_info(args))

    args = parser.parse_args()
    args.func(args)


def _cmd_generate(args):
    from aat.generate import GenerateConfig, generate_detect_dataset
    from pathlib import Path
    cfg = GenerateConfig(Path(args.images), Path(args.model), Path(args.out), confidence=args.conf, force=args.force)
    print("AAT: Generating detect dataset...")
    res = generate_detect_dataset(cfg)
    print(f"Done. {res.processed} images, {res.total_detections} detections, {res.failed} failed.")


def _cmd_view(args):
    from aat.viewer import main as viewer_main
    viewer_main()


def _cmd_info(args):
    import aat
    print(f"Arrington Annotation Tool (AAT) v{aat.__version__}")