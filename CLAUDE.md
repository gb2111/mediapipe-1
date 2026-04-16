# MediaPipe — CLAUDE.md

## Project Overview

MediaPipe is Google's open-source framework for building on-device ML pipelines. It provides:
- **Framework**: A graph-based dataflow engine where each node is a *calculator*
- **Tasks API**: High-level cross-platform APIs (vision, audio, text)
- **Solutions**: Pre-built modules for face, hand, pose, object detection, etc.
- **Model Maker**: Fine-tuning tooling for on-device models

Supported platforms: Android, iOS, Web, Linux, macOS, Windows, Edge/IoT.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core runtime | C++20 |
| Build system | Bazel 7.4.1 (`.bazelversion`) |
| Serialization | Protocol Buffers (.proto) |
| ML inference | TensorFlow Lite, XNNPACK |
| Image processing | OpenCV |
| GPU | OpenGL ES, Metal (iOS), WebGPU |
| Python bindings | pybind11 + ctypes |
| Web | TypeScript, Rollup, Jasmine |
| Android | Java / Obj-C++ |
| C++ testing | GoogleTest (gtest) |

---

## Key Directories

| Path | Purpose |
|------|---------|
| `mediapipe/calculators/` | 311+ calculator implementations (audio, core, image, tensor, tflite, util, video) |
| `mediapipe/framework/` | Core graph engine — `calculator_graph.h`, `calculator_base.h`, `packet.h` |
| `mediapipe/framework/api2/` | Recommended type-safe Node API (v2) |
| `mediapipe/framework/api3/` | Latest Node API (v3) |
| `mediapipe/framework/formats/` | Proto definitions for images, detections, landmarks, classifications |
| `mediapipe/framework/stream_handler/` | Input synchronization strategies |
| `mediapipe/gpu/` | GPU buffer management and OpenGL/Metal helpers |
| `mediapipe/modules/` | Pre-built subgraph modules (face, hand, pose, iris, holistic, …) |
| `mediapipe/tasks/cc/` | C++ task implementations |
| `mediapipe/tasks/python/` | Python task APIs (vision, audio, text) |
| `mediapipe/tasks/web/` | JS/TS task APIs |
| `mediapipe/python/pybind/` | Low-level pybind11 C++↔Python bindings |
| `mediapipe/python/solutions/` | High-level Python solution APIs |
| `mediapipe/graphs/` | `.pbtxt` graph definitions for desktop examples |
| `third_party/` | External dependency overrides |

Key entry-point headers:
- [mediapipe/framework/calculator_framework.h](mediapipe/framework/calculator_framework.h) — umbrella include for calculator authoring
- [mediapipe/framework/calculator_graph.h](mediapipe/framework/calculator_graph.h) — graph execution engine (~790 lines)
- [mediapipe/framework/calculator_base.h](mediapipe/framework/calculator_base.h) — legacy calculator base class

---

## Build Commands

```bash
# Build a target
bazel build //mediapipe/path/to:target

# Run a binary
bazel run //mediapipe/examples/desktop/face_detection:face_detection_cpu

# Build desktop examples
bash build_desktop_examples.sh

# Build Android examples
bash build_android_examples.sh

# Build Python package
python setup.py install              # builds via Bazel internally

# Disable GPU (pass as env var to Bazel or setup.py)
export MEDIAPIPE_DISABLE_GPU=1
```

Platform configs live in [.bazelrc](.bazelrc). Default: 128 parallel jobs, C++20.

---

## Test Commands

```bash
# Run a single test target
bazel test //mediapipe/calculators/core:pass_through_calculator_test

# Run all tests under a package
bazel test //mediapipe/calculators/...

# Run with verbose output
bazel test //mediapipe/path:target_test --test_output=streamed
```

- C++ tests: `*_test.cc`, colocated with source, use `cc_test()` Bazel rule
- Python tests: `*_test.py`, run via `bazel test`
- JS tests: Jasmine via `@bazel/jasmine`

---

## Android Build — faceheadroigpu (Windows)

Building for Android on Windows requires specific flags to prevent MSVC flags from leaking into the Android ARM64 clang cross-compiler.

**Use `_build_faceheadroigpu.bat`** (committed) or the equivalent bash command:

```bash
export PATH="/c/Program Files/Android/Android Studio/jbr/bin:/c/Users/gregb/AppData/Local/Android/Sdk/platform-tools:$PATH"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" \
bazel build \
  --noenable_platform_specific_config \
  -c opt --config=android_arm64 \
  --java_runtime_version=remotejdk_17 \
  --tool_java_runtime_version=remotejdk_17 \
  "--repo_env=PYTHON_BIN_PATH=D:/Repos/GitHub/mediapipe2/.python312/python.exe" \
  "--repo_env=HERMETIC_PYTHON_VERSION=3.12" \
  --worker_extra_flag="Desugar=--jvm_flag=-Xmx1g" \
  //mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu:faceheadroigpu
```

Key flags explained:
- `--noenable_platform_specific_config` — prevents Windows MSVC copts (`/w`, `/D_USE_MATH_DEFINES`, `/std:c++20`) from leaking into the Android clang cross-compiler
- `--host_cxxopt=/std:c++20`, `--host_copt=/D_USE_MATH_DEFINES` — set in `.user.bazelrc` for host (MSVC) compilation
- `HERMETIC_PYTHON_VERSION=3.12` — TF hermetic Python requires 3.12; Python 3.12 embedded is in `.python312/`
- `--worker_extra_flag="Desugar=--jvm_flag=-Xmx1g"` — limits JVM heap for Android Desugar worker (avoids paging file exhaustion)
- Android Studio JBR must be on PATH for Java toolchain

**Deploy:**
```bash
adb install -r bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu/faceheadroigpu.apk
```

**App sources:** `mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu/`
- `MainActivity.java` — streams ROI rect, receives landmarks + blendshapes packets
- `HeadRoiOverlayView.java` — touch-to-set ROI, renders landmarks, blendshape panel (mouth + jaw) at top of screen

---

## Additional Documentation

Consult these files when working on the relevant area:

| Topic | File |
|-------|------|
| Calculator patterns, graph config, subgraphs, GPU/CPU variants, proto options | [.claude/docs/architectural_patterns.md](.claude/docs/architectural_patterns.md) |
