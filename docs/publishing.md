# Publishing Guide (PyPI + GitHub Releases)

This project uses GitHub Actions for automated releases and publishing.

## 1. GitHub Releases (automatic)

- Pushing a tag `vX.Y.Z` automatically creates a GitHub Release with generated notes.
- Workflow: `.github/workflows/release.yml`

## 2. PyPI Publishing (automatic on tag)

### Setup (one-time)

1. Go to [PyPI Trusted Publishers](https://pypi.org/manage/account/publishing/)
2. Add a new publisher for the project `arrington-annotation-tool`
3. Choose "GitHub" as the publisher
4. Fill in:
   - Owner: `NANOBYTECOMPUTERS`
   - Repository: `-arrington-annotation-tool`
   - Workflow filename: `publish.yml`
   - Environment name: (leave blank or use `pypi`)

5. Repeat for TestPyPI if desired.

### How it works

- When you push a tag (e.g. `git tag v0.2.1 && git push origin v0.2.1`):
  - The `release.yml` workflow creates the GitHub Release.
  - The `publish.yml` workflow builds the package and publishes it to PyPI using OIDC Trusted Publishing (no API tokens stored).

## Manual Release (if needed)

```bash
git tag -a v0.2.1 -m "Release v0.2.1"
git push origin v0.2.1
```

## Package Name

- PyPI package name: `arrington-annotation-tool`
- Import name: `aat`

## Notes

- The leading dash in the GitHub repo name (`-arrington-annotation-tool`) is unusual but supported.
- Make sure the PyPI project name does **not** have the leading dash.