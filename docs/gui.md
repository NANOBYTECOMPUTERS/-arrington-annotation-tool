# AAT Graphical Interface

AAT offers two graphical interfaces:

## Modern GUI (Recommended)

```bash
aat-gui
# or
aat gui
```

This is built with **CustomTkinter** and provides:

- Clean modern interface with Light / Dark / System theme support
- Integrated dataset generation
- Easy access to the powerful annotation viewer
- Good experience for daily use

**Installation:**
```bash
pip install "arrington-annotation-tool[gui]"
# or
pip install "arrington-annotation-tool[all]"
```

## Classic Viewer

```bash
aat-viewer
# or
aat view
```

This is the original high-performance Tkinter viewer with advanced editing capabilities (segmentation auto-trace, OBB support, etc.).

Both interfaces are considered equally important. Use whichever you prefer.

## Future Plans

- More of the viewer functionality will gradually be ported into the modern CustomTkinter GUI.
- The classic viewer will remain available for users who prefer it or need its advanced features.