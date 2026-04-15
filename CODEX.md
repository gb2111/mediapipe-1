# Codex Notes (MediaPipe2)

## Maintenance Rule
- On every commit, update `CODEX.md`.
- Update the main sections when the current project state, build flow, runtime behavior, or important defaults have changed.
- Also append a short entry at the bottom in `Change Log` describing what changed in that commit.
- Keep `CODEX.md` as the persistent handoff note for future Codex sessions.

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

## Current Build Command (Windows, Android arm64)
Run from repo root `D:\Repos\GitHub\mediapipe2`:

```bat
cmd /c "set BAZEL_SH=C:\Progra~1\Git\bin\bash.exe && set ANDROID_HOME=C:\Users\gregb\AppData\Local\Android\Sdk && set ANDROID_SDK_ROOT=C:\Users\gregb\AppData\Local\Android\Sdk && set ANDROID_NDK_HOME=C:\Users\gregb\AppData\Local\Android\Sdk\ndk\android-ndk-r28b && set PATH=C:\PROGRA~1\Android\ANDROI~1\jbr\bin;C:\Users\gregb\AppData\Local\Android\Sdk\platform-tools;%PATH% && bazel build --noenable_platform_specific_config -c opt --config=android_arm64 --java_runtime_version=remotejdk_17 --tool_java_runtime_version=remotejdk_17 --repo_env=PYTHON_BIN_PATH=D:\Repos\GitHub\mediapipe2\.python312\python.exe --repo_env=HERMETIC_PYTHON_VERSION=3.12 --shell_executable=C:/Progra~1/Git/bin/bash.exe --copt=-Iexternal/com_google_protobuf/src --cxxopt=-Iexternal/com_google_protobuf/src --host_copt=/Iexternal/com_google_protobuf/src --host_cxxopt=/Iexternal/com_google_protobuf/src --host_cxxopt=/std:c++20 --verbose_failures //mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu:faceheadroigpu"
```

Notes:
- Android Studio path on this machine: `C:\Program Files\Android\Android Studio`
- Android SDK path on this machine: `C:\Users\gregb\AppData\Local\Android\Sdk`
- Android NDK path on this machine: `C:\Users\gregb\AppData\Local\Android\Sdk\ndk\android-ndk-r28b`
- JDK used from Android Studio JBR via `PATH`.
- `BAZEL_SH` must point to Git Bash on Windows.
- `--noenable_platform_specific_config` is required so Windows-only Bazel flags do not leak into Android clang builds.
- Extra protobuf include flags are required for both Android clang and host MSVC.
- Host MSVC also needs explicit `/std:c++20` via `--host_cxxopt=/std:c++20`.
- Local machine workaround currently present in `WORKSPACE`:
  - `android_sdk_repository` points at the local SDK.
  - `android_ndk_repository` points at the local NDK.
  - `bind(name = "android/crosstool", actual = "@androidndk//:toolchain")`

## Last Successful Build
- Date/Time: 2026-04-14 (local run within Codex)
- Duration: ~95s
- Output:
  - `bazel-bin\mediapipe\examples\desktop\face_mesh\face_mouth_landmarks_async_cpu.exe`

## Last Successful Android Build
- Date/Time: 2026-04-15
- Duration: ~1517s total, critical path ~788s
- Output:
  - `bazel-bin\mediapipe\examples\android\src\java\com\google\mediapipe\apps\faceheadroigpu\faceheadroigpu.apk`
  - `bazel-bin\mediapipe\examples\android\src\java\com\google\mediapipe\apps\faceheadroigpu\faceheadroigpu_unsigned.apk`
  - `bazel-bin\mediapipe\examples\android\src\java\com\google\mediapipe\apps\faceheadroigpu\faceheadroigpu_deploy.jar`

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

## Android Head ROI App
1. Added new Android app target:
   - `//mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu:faceheadroigpu`
2. Added new Android graph:
   - `mediapipe/graphs/face_mesh/face_head_roi_mobile_gpu.pbtxt`
3. Added single-ROI adapter calculator:
   - `mediapipe/calculators/util/single_normalized_rect_to_rect_vector_calculator.cc`
4. Android app behavior:
   - one manually drawn **Head ROI**
   - camera preview rendered on-screen
   - overlay draws Head ROI, all face landmark points, blendshape panel, and inference FPS
   - ROI is sent live from Java into the graph as `roi_rect`
   - graph uses `disable_presence_gating: true`
5. Android app code:
   - `mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu/MainActivity.java`
   - `mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu/HeadRoiOverlayView.java`
   - `mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu/BUILD`

## Useful Files
- Runner: `mediapipe/examples/desktop/face_mesh/face_mouth_landmarks_async_main.cc`
- Face landmarker graph options: `mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.proto`
- Graph logic: `mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_detector_graph.cc`
- Model download script: `download_face_landmarker_task.bat`
- Android app target: `mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu/BUILD`
- Android activity: `mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu/MainActivity.java`
- Android overlay: `mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu/HeadRoiOverlayView.java`
- Android graph: `mediapipe/graphs/face_mesh/face_head_roi_mobile_gpu.pbtxt`
- ROI adapter calculator: `mediapipe/calculators/util/single_normalized_rect_to_rect_vector_calculator.cc`

## Change Log
- 2026-04-15: Added `CODEX.md` with the working Windows build command, current runner summary, and key file references.
- 2026-04-15: Updated runner terminology to use Head ROI, kept one ROI path, and confirmed a successful long build (~95s).
- 2026-04-15: Added the rule that every future commit must also update `CODEX.md`, including both the main summary when needed and this changelog.
- 2026-04-15: Added Android Head ROI app target, mobile ROI graph, ROI adapter calculator, Android toolchain notes, and confirmed a successful `faceheadroigpu.apk` build.
