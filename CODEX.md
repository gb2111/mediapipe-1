# Codex Notes (MediaPipe2)

## Current Build Command (Windows, CPU-only)
Run from repo root `D:\Repos\GitHub\mediapipe2`:

```bat
cmd /c "set BAZEL_SH=C:\Progra~1\Git\bin\bash.exe && bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --define tflite_with_xnnpack=false --define tflite_kernel_use_xnnpack=false --repo_env=PYTHON_BIN_PATH=D:\Repos\GitHub\mediapipe2\.python312\python.exe --repo_env=HERMETIC_PYTHON_VERSION=3.12 --shell_executable=C:/Progra~1/Git/bin/bash.exe --cxxopt=/Iexternal/com_google_protobuf/src --copt=/Iexternal/com_google_protobuf/src --host_cxxopt=/Iexternal/com_google_protobuf/src --host_copt=/Iexternal/com_google_protobuf/src --cxxopt=/std:c++20 --host_cxxopt=/std:c++20 //mediapipe/examples/desktop/face_mesh:face_mouth_landmarks_async_cpu"
```

Notes:
- Uses Git Bash for Bazel (`BAZEL_SH`).
- Uses embedded Python at `.python312\python.exe` with `HERMETIC_PYTHON_VERSION=3.12`.
- Extra protobuf include flags are required for MSVC.
- CPU-only, XNNPACK disabled via `--define tflite_with_xnnpack=false --define tflite_kernel_use_xnnpack=false`.

## Last Successful Build
- Date/Time: 2026-04-14 (local run within Codex)
- Duration: ~95s
- Output:
  - `bazel-bin\mediapipe\examples\desktop\face_mesh\face_mouth_landmarks_async_cpu.exe`

## Recent Functional Changes
1. Added `disable_presence_gating` option to face landmarker detector graph:
   - `mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.proto`
   - `mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_detector_graph.cc`
2. Enabled `disable_presence_gating: true` in async ROI runner graph config:
   - `mediapipe/examples/desktop/face_mesh/face_mouth_landmarks_async_main.cc`
3. Runner currently draws **all landmarks** (tesselation + points).
4. ROI handling hardened: clamps to image and preserves last valid ROI during drag.
5. ROI labeling updated to **Head ROI** only; the drawn box is the only ROI and is sent to the model.
6. Blendshapes window added (two columns, name + progress bar + numeric value).
7. Added `download_face_landmarker_task.bat` to fetch the model bundle.

## Useful Files
- Runner: `mediapipe/examples/desktop/face_mesh/face_mouth_landmarks_async_main.cc`
- Face landmarker graph options: `mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.proto`
- Graph logic: `mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_detector_graph.cc`
- Model download script: `download_face_landmarker_task.bat`

