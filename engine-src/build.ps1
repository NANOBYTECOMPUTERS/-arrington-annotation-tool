# Build the yolo_annotation_worker.exe from source
# Run this from the engine-src directory after setting up CUDA + TensorRT + OpenCV + VS2022

param([string]$Configuration = "CUDA")

$ErrorActionPreference = "Stop"
$msbuild = "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe"
if (-not (Test-Path $msbuild)) {
    $msbuild = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
}

& $msbuild "yolo_annotation_worker.vcxproj" /p:Configuration=$Configuration /p:Platform=x64 /m /v:minimal
exit $LASTEXITCODE
