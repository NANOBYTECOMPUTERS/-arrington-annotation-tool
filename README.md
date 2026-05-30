# Arrington Annotation Tool (AAT)

**Modular, fast, AI-assisted YOLO annotation.**

Use a pretrained model (`.pt`, `.engine`, or `.onnx`) to automatically generate high-quality detection labels, then review, correct, and refine them with an excellent desktop viewer that has built-in model suggestions.

## Quick Start

```powershell
pip install "arrington-annotation-tool[ultralytics]"

# 1. Bootstrap labels from a strong pretrained model
aat generate --model yolov8n.pt --images C:\data\raw --out C:\data\labeled --conf 0.2

# 2. Launch the modern GUI (recommended)
aat gui

# Or launch the classic powerful viewer
aat view
```

## Two Interfaces (Both First-Class)

### Modern GUI (`aat gui` / `aat-gui`)
- Built with CustomTkinter
- Clean modern look with Light/Dark/System themes
- Integrated dataset generation + easy access to the viewer

### Classic Viewer (`aat view` / `aat-viewer`)
- Full-featured Tkinter viewer
- Advanced editing (auto-trace segmentation, etc.)
- Very powerful for detailed work

Both are equally important and actively maintained.

## Installation

```bash
# Recommended for most users
pip install "arrington-annotation-tool[ultralytics]"

# With modern GUI
pip install "arrington-annotation-tool[gui]"

# Everything
pip install "arrington-annotation-tool[all]"
```

## CLI

```bash
aat --help
aat gui
aat generate --model best.pt --images ./raw --out ./dataset
aat view
aat info
```

## Python API

```python
from aat import generate_detect_dataset, GenerateConfig, get_engine

cfg = GenerateConfig(
    images_root="raw-photos",
    model_path="best.pt",
    output_root="my-yolo-dataset",
    confidence=0.15
)
result = generate_detect_dataset(cfg)
print(result.processed, "images labeled")
```