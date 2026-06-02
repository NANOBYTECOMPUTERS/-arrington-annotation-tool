# yolo_annotation_worker - C++ TensorRT Inference Engine Source

This directory contains the **complete source** needed to build the native inference worker used by the Arrington Annotation Tool (AAT) for fast `.engine` (TensorRT) model inference.

## Contents

- `worker/`          — Core NDJSON protocol + batch detector + main entry point
- `detector/`        — CUDA preprocessing kernel (`cuda_preprocess.cu`) + post-processing
- `include/`         — Small custom headers referenced by the project
- `yolo_annotation_worker.vcxproj` — Visual Studio project (CUDA | x64 configuration)
- `build.ps1`        — Convenience build wrapper
- Original build scripts from the 0BS project

## Prerequisites (on the machine where you build)

- Visual Studio 2022 (or 2019) with **v145** platform toolset (MSVC v143+ usually works)
- CUDA Toolkit (the project is currently configured for CUDA 13.2 / v13.2)
- TensorRT 10.x SDK (the .vcxproj looks for it under `..\modules\TensorRT-...` or a custom `TensorRTDir`)
- OpenCV (built, the project expects a specific install under `..\modules\opencv\...`)
- NVIDIA GPU driver compatible with the CUDA version

## How to Build

1. Open a "x64 Native Tools Command Prompt for VS 2022" (or Developer PowerShell).
2. `cd` into this `engine-src` folder.
3. Run:
   ```powershell
   .\build.ps1
   ```
   or directly:
   ```powershell
   msbuild yolo_annotation_worker.vcxproj /p:Configuration=CUDA /p:Platform=x64 /m
   ```

4. The resulting `yolo_annotation_worker.exe` (and any new DLLs you may have built) will be written to the location defined in the vcxproj (`$(OutDir)` → originally `..\..\x64\CUDA\` relative to the vcxproj).

5. To use the new build with AAT:
   - Copy the new `yolo_annotation_worker.exe` + any updated DLLs into the sibling `../engine/` folder (or wherever your AAT build consumes them).
   - Re-run the main AAT `build.ps1` / `build.bat` if you want a fresh frozen GUI bundle that includes the updated engine runtime.

## Important Notes on Paths

The `.vcxproj` contains several machine-specific paths:

- `CudaToolkitCustomDir`
- `TensorRTDir`
- Include/Library paths pointing at `..\modules\opencv`, `..\modules\TensorRT-...`, etc.

You will almost certainly need to edit the vcxproj (or set the properties in Visual Studio) to point at your local installations of CUDA, TensorRT, and OpenCV before the build will succeed.

The small custom headers in `include/` are included here for convenience.

## Relationship to the rest of AAT

- `../engine/` (sibling folder) contains the **pre-built** worker + DLLs that `build.ps1` (the Python GUI builder) vendors into the final distributable.
- This `engine-src/` folder is what you use when you want to **modify or rebuild** the inference engine itself (new CUDA kernels, protocol changes, different TensorRT version, etc.).

After rebuilding, always copy the fresh artifacts into `../engine/` so the next AAT GUI build picks them up.

## License / Origin

This code was originally developed as part of a larger internal project (0BS). It is included here so the Arrington Annotation Tool can be built and extended in a self-contained way.
