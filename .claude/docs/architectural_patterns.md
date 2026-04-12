# Architectural Patterns

## 1. Calculator Pattern — Core Component Model

Every processing node in a MediaPipe graph is a **Calculator**. Three API generations exist; prefer API2 or API3 for new work.

### Legacy API (`CalculatorBase`)

Inherit from `CalculatorBase`, implement three lifecycle methods, then register.

- Base class: [mediapipe/framework/calculator_base.h](../../mediapipe/framework/calculator_base.h)
- Canonical example: [mediapipe/calculators/core/pass_through_calculator.cc](../../mediapipe/calculators/core/pass_through_calculator.cc)

Lifecycle order: `GetContract()` → `Open()` → `Process()` (per packet) → `Close()`

`GetContract()` (static) is called by the framework to validate stream types before the graph runs.
Registration macro at end of .cc file: `REGISTER_CALCULATOR(MyCalculatorClass)`.

### API2 — Recommended (type-safe `Node`)

- Header: [mediapipe/framework/api2/node.h](../../mediapipe/framework/api2/node.h)
- Example header: [mediapipe/calculators/core/pass_through_calculator.h:22-33](../../mediapipe/calculators/core/pass_through_calculator.h)

Declare a `struct` inheriting `Node<"CalculatorName">` with a templated `Contract` inner struct. The framework infers I/O types at compile time.

### API3 — Latest

- Directory: [mediapipe/framework/api3/](../../mediapipe/framework/api3/)
- Files: `node.h`, `calculator.h`, `graph.h`

Further improves type safety and ergonomics over API2.

### BUILD rule for calculators

Every calculator .cc file uses `cc_library` with `alwayslink = 1` so the `REGISTER_CALCULATOR` macro runs at startup:

```
cc_library(
    name = "my_calculator",
    srcs = ["my_calculator.cc"],
    deps = ["//mediapipe/framework:calculator_framework"],
    alwayslink = 1,
)
```

---

## 2. Graph Configuration Pattern

Graphs are defined in **Protocol Buffer text format** (`.pbtxt`).

- Example: [mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt](../../mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt)
- Schema proto: [mediapipe/framework/calculator.proto](../../mediapipe/framework/calculator.proto)

Structure of a graph file:
- Top-level `input_stream`/`output_stream` declarations
- One `node {}` block per calculator, with:
  - `calculator:` — name string matching `REGISTER_CALCULATOR`
  - `input_stream:` / `output_stream:` — `"TAG:stream_name"` syntax
  - `node_options:` — typed options via `[type.googleapis.com/mediapipe.XyzOptions] { ... }`
  - Optional `input_stream_info { tag_index: "..." back_edge: true }` for cycles

Stream naming: `"TAG:name"` (tagged) or `"name"` (positional index).

---

## 3. Subgraph / Module Pattern

Reusable multi-node pipelines are packaged as **Subgraphs**, referenced in parent graphs by type name just like calculators.

- Base header: [mediapipe/framework/subgraph.h](../../mediapipe/framework/subgraph.h)
- Registration macro: `REGISTER_MEDIAPIPE_GRAPH(SubgraphName)`
- Module directory: [mediapipe/modules/](../../mediapipe/modules/)

Each domain module (face, hand, pose, iris, holistic, objectron, …) exposes:
- A `.pbtxt` file defining the subgraph's internal graph
- CPU and GPU variants named `*_cpu.pbtxt` / `*_gpu.pbtxt`
- A `type:` field at the top matching the registered name

Example: [mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt](../../mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt)

---

## 4. Calculator Options via Proto Extension

Calculator-specific configuration is expressed as a `.proto` message that **extends** `mediapipe.CalculatorOptions`.

- Base extension proto: [mediapipe/framework/calculator_options.proto](../../mediapipe/framework/calculator_options.proto)
- Pattern example: [mediapipe/calculators/core/gate_calculator.proto:21-43](../../mediapipe/calculators/core/gate_calculator.proto)

```protobuf
message GateCalculatorOptions {
  extend mediapipe.CalculatorOptions {
    optional GateCalculatorOptions ext = 261754847;  // unique extension field number
  }
  optional bool allow = 2 [default = false];
}
```

Each calculator extension has a globally unique field number. In graphs, options are embedded in the `node_options` block using the full type URL.

BUILD macro: `mediapipe_proto_library()` (wraps `proto_library` + `cc_proto_library`).

---

## 5. GPU / CPU Variant Pattern

Calculators with GPU acceleration follow a consistent naming and BUILD convention:

- `*_gpu_calculator.cc` and `*_cpu_calculator.cc` (or a single file with `#ifdef MEDIAPIPE_DISABLE_GPU`)
- GPU helper: [mediapipe/gpu/gl_calculator_helper.h](../../mediapipe/gpu/gl_calculator_helper.h)
- GPU buffer: [mediapipe/gpu/gpu_buffer.h](../../mediapipe/gpu/gpu_buffer.h)
- Metal (iOS): `mediapipe/gpu/MPPMetalHelper`
- Graph-level: separate `.pbtxt` files per variant (`*_gpu.pbtxt`, `*_cpu.pbtxt`) in `mediapipe/modules/`

Build flag: `MEDIAPIPE_DISABLE_GPU=1` disables GPU compilation globally (set as env var before `bazel build` or `python setup.py`).

---

## 6. Python Task API Layering

Python tasks are exposed through a three-layer stack:

```
Python high-level API  (mediapipe/tasks/python/vision/*.py)
        ↓
ctypes C wrapper       (generated from tasks/cc C headers)
        ↓
C++ implementation     (mediapipe/tasks/cc/vision/*)
```

- ctypes binding example: [mediapipe/tasks/python/vision/image_classifier.py:44-129](../../mediapipe/tasks/python/vision/image_classifier.py)
- Pattern: `ctypes.CFUNCTYPE` for callbacks, `CStatusFunction` wrappers for C functions
- All task options use a common `BaseOptions` + task-specific `XyzOptions` dataclass hierarchy

---

## 7. Stream Handler Pattern (Input Synchronization)

By default the framework synchronizes all input streams by timestamp. Override with a `stream_handler` declaration inside a `node {}` block.

- Handlers directory: [mediapipe/framework/stream_handler/](../../mediapipe/framework/stream_handler/)
- Common handlers:
  - `DefaultInputStreamHandler` — waits for one packet on every stream (timestamp-synced)
  - `ImmediateInputStreamHandler` — processes packets as soon as any input arrives
  - `SyncSetInputStreamHandler` — sync subsets of streams independently

Declared in graph `.pbtxt`:
```protobuf
node {
  calculator: "MyCalculator"
  input_stream: "VIDEO:frames"
  input_stream: "DETECTIONS:dets"
  input_stream_handler {
    input_stream_handler: "ImmediateInputStreamHandler"
  }
}
```

---

## 8. Packet System

Every value that flows between calculators is wrapped in a `Packet`.

- Header: [mediapipe/framework/packet.h](../../mediapipe/framework/packet.h)
- Creation: `mediapipe::MakePacket<T>(value)` or `packet.At(timestamp)`
- Access: `packet.Get<T>()` — throws if type doesn't match
- Timestamps: microsecond integers, monotonically increasing per stream; special values `Timestamp::PreStream()`, `Timestamp::PostStream()`

Inside `Process()`, packets are retrieved via `cc->Inputs().Tag("TAG").Get<T>()` and emitted via `cc->Outputs().Tag("TAG").Add(value, timestamp)`.
