# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

licenses(["notice"])

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/face_mesh:face_mesh_tflite
# RUN:
# bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_tflite \
#   --calculator_graph_config_file=<.../face_mesh_desktop.pbtxt> \
#   --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
cc_binary(
    name = "face_mesh_tflite",
    data = ["//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite"],
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/face_mesh:desktop_calculators",
    ],
)

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/face_mesh:face_mesh_cpu
# RUN:
# bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_cpu \
#   --calculator_graph_config_file=<.../face_mesh_desktop_live.pbtxt> \
#   --input_video_path=<input video path> \
#   --output_video_path=<output video path>
cc_binary(
    name = "face_mesh_cpu",
    data = ["//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",
    ],
)

cc_binary(
    name = "face_mouth_landmarks_async_cpu",
    srcs = ["face_mouth_landmarks_async_main.cc"],
    data = [
        "//mediapipe/tasks/testdata/vision:face_landmarker_v2_with_blendshapes_task_public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_roi_graph",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarks_connections",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# Linux only
cc_binary(
    name = "face_mesh_gpu",
    data = ["//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//mediapipe/graphs/face_mesh:desktop_live_gpu_calculators",
    ],
)
