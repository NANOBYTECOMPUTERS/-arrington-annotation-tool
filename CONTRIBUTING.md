# Contributing to Arrington Annotation Tool (AAT)

Thank you for your interest!

## Development Setup

```bash
git clone https://github.com/NANOBYTECOMPUTERS/-arrington-annotation-tool.git
cd -arrington-annotation-tool
python -m pip install -e ".[ultralytics,test]"
pytest
```

## Running the Tool

```bash
aat --help
aat generate ...
aat view
```

## Code Style

- Keep the core `aat` package small and well-tested.
- New features should preferably have corresponding tests.
- The viewer is intentionally pragmatic — heavy extraction can happen incrementally.

## Reporting Issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`.

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Add tests
4. Open a PR using the template

Thanks for helping make YOLO annotation faster and more modular!