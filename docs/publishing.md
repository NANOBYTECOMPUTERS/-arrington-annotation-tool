# Publishing & Release Guide

This project has a complete automated release and publishing pipeline powered by GitHub Actions.

## Available Workflows

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| CI / Tests | `.github/workflows/ci.yml` | Push/PR to main | Run tests on Python 3.11 & 3.12 |
| Release | `.github/workflows/release.yml` | Tag `v*` | Create GitHub Release (with custom notes support) |
| PyPI Publish | `.github/workflows/publish.yml` | Tag `v*` | Build + publish to PyPI (Trusted Publishing) |
| TestPyPI Publish | `.github/workflows/publish-testpypi.yml` | Manual or tag `v*-test*` | Publish to TestPyPI |
| Windows EXE | `.github/workflows/build-exe.yml` | Release published or manual | Build standalone `AAT-Viewer.exe` and attach to release |

## 1. Creating a Release (Recommended Flow)

```bash
git tag -a v0.2.1 -m "Release v0.2.1"
git push origin v0.2.1
```

This will automatically:
- Create a GitHub Release (using notes from `RELEASES/v0.2.1.md` if present, otherwise auto-generated)
- Build and publish the package to PyPI
- Build a Windows executable and attach it to the release

## 2. PyPI / TestPyPI Setup (One-time)

### PyPI Trusted Publishing
1. Go to https://pypi.org/manage/account/publishing/
2. Add publisher for project `arrington-annotation-tool`
3. GitHub owner: `NANOBYTECOMPUTERS`
4. Repository: `-arrington-annotation-tool`
5. Workflow: `publish.yml`

### TestPyPI (optional but recommended)
Repeat the above on https://test.pypi.org for the same project name.

## 3. Custom Release Notes

Place detailed release notes in:
```
RELEASES/v0.2.1.md
```

The release workflow will automatically use this file when the tag is pushed.

## 4. Windows Executable

On every release, a standalone `AAT-Viewer.exe` is built with PyInstaller and attached to the GitHub Release under "Assets".

You can also trigger it manually from the Actions tab.

## Package Information

- **PyPI name**: `arrington-annotation-tool`
- **Import name**: `aat`
- **CLI commands**: `aat`, `aat-generate`, `aat-viewer`

**Note**: The GitHub repository name has a leading dash (`-arrington-annotation-tool`). This is supported but make sure your PyPI project does **not** include the dash.