# Arrington Annotation Tool (AAT)

**Modular, fast, AI-assisted YOLO annotation for computer vision datasets.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Use a pretrained model (`.pt`, `.engine`, or `.onnx`) to automatically generate high-quality YOLO detection labels, then review, correct, and refine them with a powerful desktop viewer that has **built-in AI suggestions**.

## Quick Start

```bash
pip install "arrington-annotation-tool[ultralytics]"

# Generate labels from a strong model
aat generate --model yolov8n.pt --images ./raw-images --out ./my-dataset --conf 0.2

# Launch the viewer with live AI Assist
aat view
```

In the viewer, use the **AI Assist panel** to load any model and get instant suggestions on any image.

## Features

- **Engine → Dataset**: One command turns images + a model into a ready-to-train YOLO detection dataset (with confidence metadata).
- **Live AI in Viewer**: The same engines power "Suggest" buttons directly in the annotation canvas.
- **Fully Modular**: `import aat` gives you clean, testable primitives.
- **Detect-first with extensibility**: Strong foundation for bounding boxes today, clean path to OBB and segmentation.

## Installation

```bash
# Core
pip install arrington-annotation-tool

# Recommended (for running models)
pip install "arrington-annotation-tool[ultralytics]"

# Full features
pip install "arrington-annotation-tool[all]"
```

## CLI

```bash
aat --help
aat generate --model best.pt --images ./raw --out ./dataset
aat view
aat info
```

## Python API Example

```python
from aat import generate_detect_dataset, GenerateConfig, get_engine

cfg = GenerateConfig(
    images_root="raw-photos",
    model_path="best.pt",
    output_root="my-yolo-dataset",
    confidence=0.15
)
result = generate_detect_dataset(cfg)
print(f"Labeled {result.processed} images")
```

## Project Goals

- Best-in-class local tool for bootstrapping and refining YOLO datasets.
- Make AI assistance a delightful, first-class part of manual annotation.
- Stay small, well-tested, and reusable as a Python library.

## Development

```bash
git clone https://github.com/NANOBYTECOMPUTERS/-arrington-annotation-tool.git
cd -arrington-annotation-tool
pip install -e ".[ultralytics,test]"
pytest
aat --help
```

## License

MIT
