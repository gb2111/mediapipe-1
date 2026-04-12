#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
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
constexpr char kPoseLandmarksStream[] = "pose_landmarks";
constexpr char kPoseWorldLandmarksStream[] = "pose_world_landmarks";
constexpr char kFaceLandmarksStream[] = "face_landmarks";
constexpr char kLeftHandLandmarksStream[] = "left_hand_landmarks";
constexpr char kRightHandLandmarksStream[] = "right_hand_landmarks";
constexpr char kWindowName[] = "MediaPipe Holistic Async";
constexpr int kPoseLandmarkCount = 33;
constexpr int kHandLandmarkCount = 21;

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
          "Holistic model complexity: 0=lite, 1=full, 2=heavy for pose.");
ABSL_FLAG(bool, smooth_landmarks, true,
          "Whether to smooth landmarks across frames.");
ABSL_FLAG(bool, refine_face_landmarks, false,
          "Whether to use the attention-based face landmark model.");
ABSL_FLAG(bool, use_prev_landmarks, true,
          "Whether to use previous landmarks to localize the next frame.");
ABSL_FLAG(bool, show_world_z, false,
          "Whether to print pose world landmark z values in the console.");

namespace {

struct FrameData {
  cv::Mat bgr_frame;
  int64_t timestamp_us = 0;
};

struct HolisticResult {
  std::mutex frame_mutex;
  std::condition_variable frame_cv;
  bool stop = false;
  bool has_pending_frame = false;
  FrameData pending_frame;

  std::mutex result_mutex;
  bool has_pose_2d = false;
  bool has_pose_3d = false;
  bool has_face = false;
  bool has_left_hand = false;
  bool has_right_hand = false;
  mediapipe::NormalizedLandmarkList pose_landmarks_2d;
  mediapipe::LandmarkList pose_landmarks_3d;
  mediapipe::NormalizedLandmarkList face_landmarks;
  mediapipe::NormalizedLandmarkList left_hand_landmarks;
  mediapipe::NormalizedLandmarkList right_hand_landmarks;
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

const std::array<std::pair<int, int>, 20> kHandConnections = {{
    {0, 1},   {1, 2},   {2, 3},   {3, 4},   {0, 5},   {5, 6},  {6, 7},
    {7, 8},   {5, 9},   {9, 10},  {10, 11}, {11, 12}, {9, 13}, {13, 14},
    {14, 15}, {15, 16}, {13, 17}, {0, 17},  {17, 18}, {18, 19},
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

cv::Point LandmarkToPoint(const mediapipe::NormalizedLandmark& landmark,
                          const cv::Mat& frame) {
  return cv::Point(static_cast<int>(landmark.x() * frame.cols),
                   static_cast<int>(landmark.y() * frame.rows));
}

void DrawPoseLandmarks(const mediapipe::NormalizedLandmarkList& landmarks,
                       cv::Mat& frame) {
  if (landmarks.landmark_size() < kPoseLandmarkCount) {
    return;
  }

  for (const auto& connection : kPoseConnections) {
    const auto& start = landmarks.landmark(connection.first);
    const auto& end = landmarks.landmark(connection.second);
    if ((start.has_visibility() && start.visibility() < 0.2f) ||
        (end.has_visibility() && end.visibility() < 0.2f)) {
      continue;
    }
    cv::line(frame, LandmarkToPoint(start, frame), LandmarkToPoint(end, frame),
             cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
  }

  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& landmark = landmarks.landmark(i);
    if (landmark.has_visibility() && landmark.visibility() < 0.2f) {
      continue;
    }
    cv::circle(frame, LandmarkToPoint(landmark, frame), 4,
               cv::Scalar(255, 128, 0), cv::FILLED, cv::LINE_AA);
  }
}

void DrawHandLandmarks(const mediapipe::NormalizedLandmarkList& landmarks,
                       const cv::Scalar& color, cv::Mat& frame) {
  if (landmarks.landmark_size() < kHandLandmarkCount) {
    return;
  }

  for (const auto& connection : kHandConnections) {
    const auto& start = landmarks.landmark(connection.first);
    const auto& end = landmarks.landmark(connection.second);
    cv::line(frame, LandmarkToPoint(start, frame), LandmarkToPoint(end, frame),
             color, 2, cv::LINE_AA);
  }

  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    cv::circle(frame, LandmarkToPoint(landmarks.landmark(i), frame), 3, color,
               cv::FILLED, cv::LINE_AA);
  }
}

void DrawFaceLandmarks(const mediapipe::NormalizedLandmarkList& landmarks,
                       cv::Mat& frame) {
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    cv::circle(frame, LandmarkToPoint(landmarks.landmark(i), frame), 1,
               cv::Scalar(80, 255, 120), cv::FILLED, cv::LINE_AA);
  }
}

mediapipe::CalculatorGraphConfig CreateGraphConfig() {
  return mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
    input_stream: "input_image"
    output_stream: "pose_landmarks"
    output_stream: "pose_world_landmarks"
    output_stream: "face_landmarks"
    output_stream: "left_hand_landmarks"
    output_stream: "right_hand_landmarks"
    node {
      calculator: "HolisticLandmarkCpu"
      input_stream: "IMAGE:input_image"
      input_side_packet: "MODEL_COMPLEXITY:model_complexity"
      input_side_packet: "SMOOTH_LANDMARKS:smooth_landmarks"
      input_side_packet: "REFINE_FACE_LANDMARKS:refine_face_landmarks"
      input_side_packet: "USE_PREV_LANDMARKS:use_prev_landmarks"
      output_stream: "POSE_LANDMARKS:pose_landmarks"
      output_stream: "WORLD_LANDMARKS:pose_world_landmarks"
      output_stream: "FACE_LANDMARKS:face_landmarks"
      output_stream: "LEFT_HAND_LANDMARKS:left_hand_landmarks"
      output_stream: "RIGHT_HAND_LANDMARKS:right_hand_landmarks"
    }
  )pb");
}

absl::Status RunAsyncHolistic() {
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(
      CreateGraphConfig(),
      {
          {"model_complexity",
           mediapipe::MakePacket<int>(absl::GetFlag(FLAGS_model_complexity))},
          {"smooth_landmarks",
           mediapipe::MakePacket<bool>(absl::GetFlag(FLAGS_smooth_landmarks))},
          {"refine_face_landmarks",
           mediapipe::MakePacket<bool>(
               absl::GetFlag(FLAGS_refine_face_landmarks))},
          {"use_prev_landmarks",
           mediapipe::MakePacket<bool>(absl::GetFlag(FLAGS_use_prev_landmarks))},
      }));

  HolisticResult state;
  using Clock = std::chrono::steady_clock;
  auto last_pose_time = Clock::now();
  double smoothed_inference_fps = 0.0;

  MP_RETURN_IF_ERROR(graph.ObserveOutputStream(
      kPoseLandmarksStream,
      [&](const mediapipe::Packet& packet) -> absl::Status {
        const auto current_time = Clock::now();
        const auto inference_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_pose_time)
                .count();
        if (inference_ms > 0) {
          const double instant_fps = 1000.0 / inference_ms;
          smoothed_inference_fps =
              smoothed_inference_fps == 0.0
                  ? instant_fps
                  : smoothed_inference_fps * 0.9 + instant_fps * 0.1;
        }
        last_pose_time = current_time;

        std::lock_guard<std::mutex> lock(state.result_mutex);
        state.latest_timestamp_us = packet.Timestamp().Value();
        state.inference_fps = smoothed_inference_fps;
        state.has_pose_2d = !packet.IsEmpty();
        if (state.has_pose_2d) {
          state.pose_landmarks_2d =
              packet.Get<mediapipe::NormalizedLandmarkList>();
        }
        return absl::OkStatus();
      }));
  MP_RETURN_IF_ERROR(graph.ObserveOutputStream(
      kPoseWorldLandmarksStream,
      [&](const mediapipe::Packet& packet) -> absl::Status {
        std::lock_guard<std::mutex> lock(state.result_mutex);
        state.has_pose_3d = !packet.IsEmpty();
        if (state.has_pose_3d) {
          state.pose_landmarks_3d = packet.Get<mediapipe::LandmarkList>();
          if (absl::GetFlag(FLAGS_show_world_z) &&
              state.pose_landmarks_3d.landmark_size() > 23) {
            ABSL_LOG(INFO) << "Hip center z="
                           << state.pose_landmarks_3d.landmark(23).z();
          }
        }
        return absl::OkStatus();
      }));
  MP_RETURN_IF_ERROR(graph.ObserveOutputStream(
      kFaceLandmarksStream,
      [&](const mediapipe::Packet& packet) -> absl::Status {
        std::lock_guard<std::mutex> lock(state.result_mutex);
        state.has_face = !packet.IsEmpty();
        if (state.has_face) {
          state.face_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
        }
        return absl::OkStatus();
      }));
  MP_RETURN_IF_ERROR(graph.ObserveOutputStream(
      kLeftHandLandmarksStream,
      [&](const mediapipe::Packet& packet) -> absl::Status {
        std::lock_guard<std::mutex> lock(state.result_mutex);
        state.has_left_hand = !packet.IsEmpty();
        if (state.has_left_hand) {
          state.left_hand_landmarks =
              packet.Get<mediapipe::NormalizedLandmarkList>();
        }
        return absl::OkStatus();
      }));
  MP_RETURN_IF_ERROR(graph.ObserveOutputStream(
      kRightHandLandmarksStream,
      [&](const mediapipe::Packet& packet) -> absl::Status {
        std::lock_guard<std::mutex> lock(state.result_mutex);
        state.has_right_hand = !packet.IsEmpty();
        if (state.has_right_hand) {
          state.right_hand_landmarks =
              packet.Get<mediapipe::NormalizedLandmarkList>();
        }
        return absl::OkStatus();
      }));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  cv::VideoCapture capture;
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    const int camera_id = absl::GetFlag(FLAGS_camera_id);
    ABSL_LOG(INFO) << "Opening camera_id=" << camera_id
                   << " with backend=" << absl::GetFlag(FLAGS_camera_backend);
    capture.open(camera_id, GetCameraBackend());
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

  std::thread inference_thread([&]() {
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
      }
    }
  });

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

    mediapipe::NormalizedLandmarkList pose_landmarks_2d;
    mediapipe::NormalizedLandmarkList face_landmarks;
    mediapipe::NormalizedLandmarkList left_hand_landmarks;
    mediapipe::NormalizedLandmarkList right_hand_landmarks;
    bool has_pose = false;
    bool has_face = false;
    bool has_left_hand = false;
    bool has_right_hand = false;
    double inference_fps = 0.0;
    {
      std::lock_guard<std::mutex> lock(state.result_mutex);
      has_pose = state.has_pose_2d;
      has_face = state.has_face;
      has_left_hand = state.has_left_hand;
      has_right_hand = state.has_right_hand;
      inference_fps = state.inference_fps;
      if (has_pose) {
        pose_landmarks_2d = state.pose_landmarks_2d;
      }
      if (has_face) {
        face_landmarks = state.face_landmarks;
      }
      if (has_left_hand) {
        left_hand_landmarks = state.left_hand_landmarks;
      }
      if (has_right_hand) {
        right_hand_landmarks = state.right_hand_landmarks;
      }
    }

    if (has_face) {
      DrawFaceLandmarks(face_landmarks, camera_frame);
    }
    if (has_left_hand) {
      DrawHandLandmarks(left_hand_landmarks, cv::Scalar(255, 80, 80),
                        camera_frame);
    }
    if (has_right_hand) {
      DrawHandLandmarks(right_hand_landmarks, cv::Scalar(80, 180, 255),
                        camera_frame);
    }
    if (has_pose) {
      DrawPoseLandmarks(pose_landmarks_2d, camera_frame);
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
  const absl::Status status = RunAsyncHolistic();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run async holistic graph: "
                    << status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
