<#
Build script for Arrington Annotation Tool (AAT)

Output: ./build/Arrington Annotation Tool/  (onedir bundle)

The AAT folder is self-contained for building + deploying:
- src/ + tools/ + pyproject.toml   → Python GUI + aat / yolo_annotator packages
- engine/                          → Pre-built C++ TensorRT worker + required DLLs (vendored runtime)
- engine-src/                      → Full source to rebuild the C++ engine (worker/, detector/, .vcxproj, etc.)

The final distributable contains everything needed except your .engine model(s) and dataset.
#>

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$EntryPoint = Join-Path $Root "tools\aat_viewer_app.py"
$BuildOut = Join-Path $Root "build"
$AppName = "Arrington Annotation Tool"
$DistDir = Join-Path $BuildOut $AppName

Write-Host "=== Building $AppName ===" -ForegroundColor Cyan

# ====================== DEPENDENCY CHECK ======================
Write-Host "`n[1/4] Checking and installing build dependencies..." -ForegroundColor Yellow

# Simple Python check with auto-download prompt
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found in PATH." -ForegroundColor Red
    Write-Host "Please download and install Python 3.11 or newer from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add python.exe to PATH' during installation." -ForegroundColor Yellow
    Write-Host "After installing, restart this script." -ForegroundColor Yellow
    exit 1
}

# Upgrade pip
python -m pip install --upgrade pip --quiet

# Install the project and PyInstaller
python -m pip install -e ".[dev]" --quiet

$hasPyInstaller = python -c "import importlib.util; print([bool(importlib.util.find_spec('PyInstaller'))][0])" 2>$null
if ($hasPyInstaller -ne "True") {
    python -m pip install pyinstaller --quiet
}

Write-Host "Dependencies ready.`n" -ForegroundColor Green

# ====================== BUILD ======================
Write-Host "`n[2/4] Preparing build environment..." -ForegroundColor Yellow

# Clean previous build
if (Test-Path $BuildOut) {
    Write-Host "Cleaning previous build..." -ForegroundColor DarkGray
    Remove-Item $BuildOut -Recurse -Force
}

# Ensure src is importable during build (for yolo_annotator legacy pieces)
$env:PYTHONPATH = (Join-Path $Root "src") + ";" + $env:PYTHONPATH

Write-Host "`n[3/4] Running PyInstaller..." -ForegroundColor Yellow

# Run PyInstaller (onedir, windowed, heavy ML libs excluded because inference is C++)
python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name $AppName `
    --exclude-module ultralytics `
    --exclude-module torch `
    --exclude-module torchvision `
    --exclude-module torchaudio `
    --exclude-module cv2 `
    --exclude-module numpy `
    --exclude-module scipy `
    --exclude-module pandas `
    --exclude-module matplotlib `
    --exclude-module onnxruntime `
    --exclude-module tensorrt `
    --hidden-import yolo_annotator `
    --hidden-import yolo_annotator.dataset `
    --hidden-import yolo_annotator.yolo `
    --hidden-import yolo_annotator.classes `
    --hidden-import yolo_annotator.worker_protocol `
    --distpath $BuildOut `
    --workpath (Join-Path $BuildOut "work") `
    --specpath (Join-Path $BuildOut "spec") `
    $EntryPoint

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed"
}

# ====================== BUNDLE THE C++ INFERENCE ENGINE (the "engine") ======================
Write-Host "`n[4/4] Bundling C++ TensorRT worker + runtime DLLs..." -ForegroundColor Yellow

# The engine/ folder (sibling to this script) contains the vendored pre-built
# C++ TensorRT inference runtime.
#
# The companion engine-src/ folder contains the full C++ source + .vcxproj
# so you can modify and rebuild the worker when needed (new kernels, protocol
# changes, updated TensorRT, etc.).
#
# See engine-src/README.md for build instructions and prerequisites.
#
# After rebuilding the worker, copy the resulting .exe + any new/updated DLLs
# back into engine/ so the next run of this build script picks them up.
$EngineSource = Join-Path $Root "engine"
if (-not (Test-Path (Join-Path $EngineSource "yolo_annotation_worker.exe"))) {
    # Fallback only for the original development machine that has the 0BS tree.
    # On any other machine / clean clone this path will not exist and the build
    # will emit a clear warning (you must populate engine/ first).
    $EngineSource = "C:\Users\donar\OneDrive\Desktop\0BS\x64\CUDA"
    Write-Host "Local vendored engine/ not found — falling back to dev path." -ForegroundColor Yellow
}

$WorkerExe = Join-Path $EngineSource "yolo_annotation_worker.exe"

if (Test-Path $WorkerExe) {
    $EngineDest = Join-Path $DistDir "engine"
    New-Item -ItemType Directory -Force -Path $EngineDest | Out-Null

    Copy-Item $WorkerExe $EngineDest -Force
    Write-Host "  + yolo_annotation_worker.exe" -ForegroundColor DarkGray

    $Dlls = Get-ChildItem $EngineSource -Filter "*.dll" -ErrorAction SilentlyContinue
    $copied = 0
    foreach ($dll in $Dlls) {
        Copy-Item $dll.FullName $EngineDest -Force
        $copied++
    }
    Write-Host "  + $copied supporting DLLs (TensorRT, cuDNN, ONNX Runtime, etc.)" -ForegroundColor DarkGray

    Write-Host "Engine runtime included at: $EngineDest" -ForegroundColor Green
} else {
    Write-Host "WARNING: C++ worker not found at $EngineSource" -ForegroundColor Yellow
    Write-Host "Build succeeded for the GUI, but .engine (TensorRT) inference will require you to supply the worker manually." -ForegroundColor Yellow
}

Write-Host "`n[OK] Build complete!" -ForegroundColor Green
Write-Host "Output folder: $DistDir" -ForegroundColor Green
Write-Host ""
Write-Host 'To run:'
Write-Host '  1. The C++ worker + all needed DLLs are already inside the "engine" subfolder.'
Write-Host '  2. Run "Arrington Annotation Tool.exe" from inside the output folder.'
Write-Host '  3. The GUI should auto-detect the worker; just point it at your .engine model and dataset.'
Write-Host ''
Write-Host 'Only your model (.engine) and dataset are required from you. Everything else is in the folder.' -ForegroundColor Yellow
