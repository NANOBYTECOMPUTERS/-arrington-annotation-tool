"""Main CLI entry point for Arrington Annotation Tool (AAT).

Provides the primary user-facing commands for the modular annotation tool:

    aat generate   - Turn a pretrained model/engine into a YOLO detect dataset
    aat view       - Launch the AAT Viewer (with AI engine suggest built-in)
    aat info       - Show version and environment info

This is the recommended way to use AAT from the terminal.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aat",
        description="Arrington Annotation Tool (AAT) — modular AI-assisted YOLO annotation",
        epilog="Examples:\n"
               "  aat generate --model yolov8n.pt --images ./raw --out ./my-dataset --conf 0.2\n"
               "  aat view\n"
               "  aat info",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate subcommand ---
    gen = subparsers.add_parser(
        "generate", aliases=["gen"],
        help="Run a pretrained model/engine over images and write a YOLO detect dataset",
    )
    gen.add_argument("--model", "-m", required=True, help="Path to model (.pt, .engine, or .onnx)")
    gen.add_argument("--images", "-i", required=True, help="Input images folder (flat or YOLO layout)")
    gen.add_argument("--out", "-o", required=True, help="Output dataset root (will contain images/, labels/, data.yaml)")
    gen.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    gen.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    gen.add_argument("--force", action="store_true", help="Overwrite output directory if not empty")
    gen.set_defaults(func=_cmd_generate)

    # --- view subcommand ---
    view = subparsers.add_parser(
        "view", aliases=["viewer"],
        help="Launch the Arrington Annotation Tool (AAT) Viewer",
    )
    view.set_defaults(func=_cmd_view)

    # --- info subcommand ---
    info = subparsers.add_parser("info", help="Show AAT version and environment details")
    info.set_defaults(func=_cmd_info)

    args = parser.parse_args()
    args.func(args)


def _cmd_generate(args: argparse.Namespace) -> None:
    from aat.generate import GenerateConfig, generate_detect_dataset

    cfg = GenerateConfig(
        images_root=args.images,
        model_path=args.model,
        output_root=args.out,
        confidence=args.conf,
        iou=args.iou,
        force=args.force,
    )
    print(f"AAT: Generating detect dataset from {args.model} ...")
    try:
        res = generate_detect_dataset(cfg)
        print(f"Done. Processed {res.processed} images, {res.total_detections} detections, {res.failed} failed.")
        if res.output_root:
            print(f"Output: {res.output_root}")
        if res.errors:
            print("Errors (first 3):")
            for e in res.errors[:3]:
                print("  ", e)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_view(args: argparse.Namespace) -> None:
    from aat.viewer import main as viewer_main
    viewer_main()


def _cmd_info(args: argparse.Namespace) -> None:
    import aat
    from aat.config import get_config

    cfg = get_config()
    print(f"Arrington Annotation Tool (AAT)  v{aat.__version__}")
    print(f"Workspace root: {cfg.workspace_root}")
    print(f"Jobs dir:       {cfg.jobs_dir}")
    print()
    print("Core capabilities:")
    print("  - generate_detect_dataset (engine → YOLO labels)")
    print("  - AAT Viewer with live model suggestions")
    print("  - Pluggable DetectionEngine protocol")
    print()
    print("Entry points: aat, aat-generate, aat-viewer")
    print("Install extras: [ultralytics] for inference, [refine] for SAM post-processing")


if __name__ == "__main__":
    main()
