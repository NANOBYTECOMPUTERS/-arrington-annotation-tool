# Changelog

All notable changes to the Arrington Annotation Tool (AAT) will be documented in this file.

## [0.2.0] - 2026-05-30

### Added
- Initial modular release of AAT
- `generate_detect_dataset` — bootstrap YOLO detection datasets from pretrained models/engines
- Pluggable `DetectionEngine` protocol with Ultralytics implementation
- Full-featured `AATViewerApp` with live "AI Assist" (model loading + suggest on images)
- Clean I/O layer for YOLO labels and generalized annotation shapes (detect + polygon foundation)
- Unified CLI (`aat generate`, `aat view`, `aat info`)
- Comprehensive test suite for core modules
- GitHub Actions CI for Python 3.11/3.12

### Notes
- This is the clean, standalone modular version focused on the annotation workflow.
- Viewer still uses a small bridge for some advanced editing primitives (planned for further extraction).
