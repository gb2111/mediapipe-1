#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <string>
#include <thread>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

constexpr char kInputStream[] = "input_image";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWorldLandmarksStream[] = "world_landmarks";
constexpr char kWindowName[] = "MediaPipe Pose Async";
constexpr int kPoseLandmarkCount = 33;

#ifdef _WIN32
constexpr char kDefaultCameraBackend[] = "dshow";
#else
constexpr char kDefaultCameraBackend[] = "any";
#endif

ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. If not provided, use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only).");
ABSL_FLAG(int, camera_id, 0,
          "Camera index to open when input_video_path is not provided.");
ABSL_FLAG(std::string, camera_backend, kDefaultCameraBackend,
          "Camera backend: any, dshow, msmf.");
ABSL_FLAG(int, camera_width, 640, "Requested camera width.");
ABSL_FLAG(int, camera_height, 480, "Requested camera height.");
ABSL_FLAG(int, model_complexity, 0,
          "Pose landmark model complexity: 0=lite, 1=full, 2=heavy.");
ABSL_FLAG(bool, smooth_landmarks, true,
          "Whether to smooth landmarks across frames.");
ABSL_FLAG(bool, show_world_z, false,
          "Whether to print world landmark z values in the console.");

namespace {

struct FrameData {
  cv::Mat bgr_frame;
  int64_t timestamp_us = 0;
};

struct SharedState {
  std::mutex frame_mutex;
  std::condition_variable frame_cv;
  bool stop = false;
  bool has_pending_frame = false;
  FrameData pending_frame;

  std::mutex result_mutex;
  bool has_landmarks = false;
  mediapipe::NormalizedLandmarkList landmarks_2d;
  mediapipe::LandmarkList landmarks_3d;
  int64_t latest_timestamp_us = 0;
  double inference_fps = 0.0;
};

const std::array<std::pair<int, int>, 35> kPoseConnections = {{
    {0, 1},   {1, 2},   {2, 3},   {3, 7},   {0, 4},   {4, 5},
    {5, 6},   {6, 8},   {9, 10},  {11, 12}, {11, 13}, {13, 15},
    {15, 17}, {15, 19}, {15, 21}, {17, 19}, {12, 14}, {14, 16},
    {16, 18}, {16, 20}, {16, 22}, {18, 20}, {11, 23}, {12, 24},
    {23, 24}, {23, 25}, {24, 26}, {25, 27}, {26, 28}, {27, 29},
    {28, 30}, {29, 31}, {30, 32}, {27, 31}, {28, 32},
}};

int GetCameraBackend() {
  const std::string backend = absl::GetFlag(FLAGS_camera_backend);
  if (backend == "any") {
    return cv::CAP_ANY;
  }
#ifdef _WIN32
  if (backend == "dshow") {
    return cv::CAP_DSHOW;
  }
  if (backend == "msmf") {
    return cv::CAP_MSMF;
  }
#endif
  ABSL_LOG(WARNING) << "Unknown camera backend: " << backend
                    << ". Falling back to CAP_ANY.";
  return cv::CAP_ANY;
}

void DrawFps(cv::Mat& frame, double display_fps, double inference_fps) {
  char fps_text[64];
  std::snprintf(fps_text, sizeof(fps_text), "Display FPS: %.1f  Infer FPS: %.1f",
                display_fps, inference_fps);
  cv::putText(frame, fps_text, cv::Point(16, 32), cv::FONT_HERSHEY_SIMPLEX,
              0.7, cv::Scalar(0, 0, 0), 4);
  cv::putText(frame, fps_text, cv::Point(16, 32), cv::FONT_HERSHEY_SIMPLEX,
              0.7, cv::Scalar(0, 255, 0), 2);
}

void DrawPoseLandmarks(const mediapipe::NormalizedLandmarkList& landmarks,
                       cv::Mat& frame) {
  if (landmarks.landmark_size() < kPoseLandmarkCount) {
    return;
  }

  const auto ToPoint = [&](int index) {
    const auto& landmark = landmarks.landmark(index);
    return cv::Point(static_cast<int>(landmark.x() * frame.cols),
                     static_cast<int>(landmark.y() * frame.rows));
  };

  for (const auto& connection : kPoseConnections) {
    const auto& start = landmarks.landmark(connection.first);
    const auto& end = landmarks.landmark(connection.second);
    if ((start.has_visibility() && start.visibility() < 0.2f) ||
        (end.has_visibility() && end.visibility() < 0.2f)) {
      continue;
    }
    cv::line(frame, ToPoint(connection.first), ToPoint(connection.second),
             cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
  }

  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& landmark = landmarks.landmark(i);
    if (landmark.has_visibility() && landmark.visibility() < 0.2f) {
      continue;
    }
    cv::circle(frame, ToPoint(i), 4, cv::Scalar(255, 128, 0), cv::FILLED,
               cv::LINE_AA);
  }
}

mediapipe::CalculatorGraphConfig CreateGraphConfig() {
  return mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
    input_stream: "input_image"
    output_stream: "world_landmarks"
    output_stream: "landmarks"
    node {
      calculator: "PoseLandmarkCpu"
      input_stream: "IMAGE:input_image"
      input_side_packet: "MODEL_COMPLEXITY:model_complexity"
      input_side_packet: "SMOOTH_LANDMARKS:smooth_landmarks"
      output_stream: "LANDMARKS:landmarks"
      output_stream: "WORLD_LANDMARKS:world_landmarks"
    }
  )pb");
}

absl::Status RunAsyncPose() {
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(
      CreateGraphConfig(),
      {
          {"model_complexity",
           mediapipe::MakePacket<int>(absl::GetFlag(FLAGS_model_complexity))},
          {"smooth_landmarks",
           mediapipe::MakePacket<bool>(absl::GetFlag(FLAGS_smooth_landmarks))},
      }));

  MP_ASSIGN_OR_RETURN(auto landmarks_poller,
                      graph.AddOutputStreamPoller(kLandmarksStream));
  MP_ASSIGN_OR_RETURN(auto world_landmarks_poller,
                      graph.AddOutputStreamPoller(kWorldLandmarksStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  cv::VideoCapture capture;
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    const int camera_id = absl::GetFlag(FLAGS_camera_id);
    const int backend = GetCameraBackend();
    ABSL_LOG(INFO) << "Opening camera_id=" << camera_id
                   << " with backend=" << absl::GetFlag(FLAGS_camera_backend);
    capture.open(camera_id, backend);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, absl::GetFlag(FLAGS_camera_width));
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, absl::GetFlag(FLAGS_camera_height));
    capture.set(cv::CAP_PROP_FPS, 30);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, 1);
  }

  SharedState state;
  std::thread inference_thread([&]() {
    using Clock = std::chrono::steady_clock;
    auto last_inference_time = Clock::now();
    double smoothed_inference_fps = 0.0;

    while (true) {
      FrameData frame;
      {
        std::unique_lock<std::mutex> lock(state.frame_mutex);
        state.frame_cv.wait(lock,
                            [&]() { return state.stop || state.has_pending_frame; });
        if (state.stop && !state.has_pending_frame) {
          break;
        }
        frame = std::move(state.pending_frame);
        state.has_pending_frame = false;
      }

      cv::Mat rgb_frame;
      cv::cvtColor(frame.bgr_frame, rgb_frame, cv::COLOR_BGR2RGB);
      auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
          mediapipe::ImageFormat::SRGB, rgb_frame.cols, rgb_frame.rows,
          mediapipe::ImageFrame::kDefaultAlignmentBoundary);
      cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
      rgb_frame.copyTo(input_frame_mat);

      auto status = graph.AddPacketToInputStream(
          kInputStream, mediapipe::Adopt(input_frame.release())
                            .At(mediapipe::Timestamp(frame.timestamp_us)));
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to add frame to graph: " << status.message();
        continue;
      }

      mediapipe::Packet landmarks_packet;
      mediapipe::Packet world_landmarks_packet;
      if (!landmarks_poller.Next(&landmarks_packet) ||
          !world_landmarks_poller.Next(&world_landmarks_packet)) {
        break;
      }

      const auto current_inference_time = Clock::now();
      const auto inference_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              current_inference_time - last_inference_time)
              .count();
      if (inference_ms > 0) {
        const double instant_fps = 1000.0 / inference_ms;
        smoothed_inference_fps =
            smoothed_inference_fps == 0.0
                ? instant_fps
                : smoothed_inference_fps * 0.9 + instant_fps * 0.1;
      }
      last_inference_time = current_inference_time;

      std::lock_guard<std::mutex> lock(state.result_mutex);
      state.latest_timestamp_us = frame.timestamp_us;
      state.inference_fps = smoothed_inference_fps;
      state.has_landmarks =
          !landmarks_packet.IsEmpty() && !world_landmarks_packet.IsEmpty();
      if (state.has_landmarks) {
        state.landmarks_2d =
            landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();
        state.landmarks_3d = world_landmarks_packet.Get<mediapipe::LandmarkList>();
        if (absl::GetFlag(FLAGS_show_world_z) &&
            state.landmarks_3d.landmark_size() > 0) {
          ABSL_LOG(INFO) << "Hip center z="
                         << state.landmarks_3d.landmark(23).z();
        }
      }
    }
  });

  using Clock = std::chrono::steady_clock;
  auto last_display_time = Clock::now();
  double display_fps = 0.0;
  int64_t frame_counter = 0;
  bool grab_frames = true;
  while (grab_frames) {
    cv::Mat camera_frame;
    capture >> camera_frame;
    if (camera_frame.empty()) {
      if (!load_video) {
        continue;
      }
      break;
    }
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, 1);
    }

    const auto current_display_time = Clock::now();
    const auto display_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            current_display_time - last_display_time)
            .count();
    if (display_ms > 0) {
      const double instant_fps = 1000.0 / display_ms;
      display_fps =
          display_fps == 0.0 ? instant_fps : display_fps * 0.9 + instant_fps * 0.1;
    }
    last_display_time = current_display_time;

    FrameData frame_data;
    frame_data.bgr_frame = camera_frame.clone();
    frame_data.timestamp_us = ++frame_counter * 33333;
    {
      std::lock_guard<std::mutex> lock(state.frame_mutex);
      state.pending_frame = std::move(frame_data);
      state.has_pending_frame = true;
    }
    state.frame_cv.notify_one();

    mediapipe::NormalizedLandmarkList landmarks_2d;
    double inference_fps = 0.0;
    bool has_landmarks = false;
    {
      std::lock_guard<std::mutex> lock(state.result_mutex);
      has_landmarks = state.has_landmarks;
      inference_fps = state.inference_fps;
      if (has_landmarks) {
        landmarks_2d = state.landmarks_2d;
      }
    }

    if (has_landmarks) {
      DrawPoseLandmarks(landmarks_2d, camera_frame);
    }
    DrawFps(camera_frame, display_fps, inference_fps);

    if (save_video) {
      if (!writer.isOpened()) {
        writer.open(absl::GetFlag(FLAGS_output_video_path),
                    mediapipe::fourcc('a', 'v', 'c', '1'),
                    30.0, camera_frame.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(camera_frame);
    } else {
      cv::imshow(kWindowName, camera_frame);
      const int pressed_key = cv::waitKey(1);
      if (pressed_key >= 0 && pressed_key != 255) {
        grab_frames = false;
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(state.frame_mutex);
    state.stop = true;
  }
  state.frame_cv.notify_one();
  inference_thread.join();

  if (writer.isOpened()) {
    writer.release();
  }
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  const absl::Status status = RunAsyncPose();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run async pose graph: " << status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
