#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_connections.h"

constexpr char kInputStream[] = "input_image";
constexpr char kRoiRectsStream[] = "roi_rects";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kBlendshapesStream[] = "blendshapes";
constexpr char kWindowName[] = "MediaPipe Mouth ROI";
constexpr char kCropWindowName[] = "MediaPipe Mouth ROI (Crop)";

#ifdef _WIN32
constexpr char kDefaultCameraBackend[] = "dshow";
#else
constexpr char kDefaultCameraBackend[] = "any";
#endif

ABSL_FLAG(int, camera_id, 0, "Camera index to open.");
ABSL_FLAG(std::string, camera_backend, kDefaultCameraBackend,
          "Camera backend: any, dshow, msmf.");
ABSL_FLAG(int, camera_width, 640, "Requested camera width.");
ABSL_FLAG(int, camera_height, 480, "Requested camera height.");
ABSL_FLAG(std::string, model_path,
          "mediapipe/tasks/testdata/vision/"
          "face_landmarker_v2_with_blendshapes.task",
          "Path to face landmarker .task model bundle.");
ABSL_FLAG(std::string, roi_mode, "mouse",
          "ROI mode: mouse or full. Mouse uses click-drag selection.");
ABSL_FLAG(float, mouth_scale_x, 2.4f,
          "Scale factor to expand mouth ROI into a virtual face ROI (width).");
ABSL_FLAG(float, mouth_scale_y, 3.0f,
          "Scale factor to expand mouth ROI into a virtual face ROI (height).");
ABSL_FLAG(float, mouth_shift_y, 1.0f,
          "Shift the virtual face ROI upward by this *mouth height* amount.");

namespace {

struct FrameData {
  cv::Mat bgr_frame;
  cv::Rect roi_px;
  std::vector<mediapipe::NormalizedRect> roi_rects;
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
  mediapipe::NormalizedLandmarkList landmarks;
  bool has_blendshapes = false;
  mediapipe::ClassificationList blendshapes;
  double inference_fps = 0.0;
};

struct RoiSelection {
  bool dragging = false;
  bool has_roi = false;
  cv::Point start;
  cv::Point current;
  cv::Rect roi;
};

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
                          const cv::Size& full_size,
                          const cv::Point& offset) {
  return cv::Point(
      static_cast<int>(landmark.x() * full_size.width) + offset.x,
      static_cast<int>(landmark.y() * full_size.height) + offset.y);
}

void DrawAllLandmarks(const mediapipe::NormalizedLandmarkList& landmarks,
                      cv::Mat& frame, const cv::Size& full_size,
                      const cv::Point& offset) {
  if (landmarks.landmark_size() == 0) {
    return;
  }

  for (const auto& connection :
       mediapipe::tasks::vision::face_landmarker::FaceLandmarksConnections::
           kFaceLandmarksTesselation) {
    if (connection[0] >= landmarks.landmark_size() ||
        connection[1] >= landmarks.landmark_size()) {
      continue;
    }
    const auto& start = landmarks.landmark(connection[0]);
    const auto& end = landmarks.landmark(connection[1]);
    cv::line(frame, LandmarkToPoint(start, full_size, offset),
             LandmarkToPoint(end, full_size, offset), cv::Scalar(200, 200, 200),
             1, cv::LINE_AA);
  }

  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& landmark = landmarks.landmark(i);
    cv::circle(frame, LandmarkToPoint(landmark, full_size, offset), 2,
               cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
  }
}

mediapipe::NormalizedRect MakeNormalizedRect(const cv::Rect& roi,
                                             const cv::Size& size) {
  mediapipe::NormalizedRect rect;
  rect.set_rotation(0.0f);
  rect.set_x_center((roi.x + roi.width * 0.5f) / size.width);
  rect.set_y_center((roi.y + roi.height * 0.5f) / size.height);
  rect.set_width(roi.width / static_cast<float>(size.width));
  rect.set_height(roi.height / static_cast<float>(size.height));
  rect.set_x_center(std::min(1.0f, std::max(0.0f, rect.x_center())));
  rect.set_y_center(std::min(1.0f, std::max(0.0f, rect.y_center())));
  rect.set_width(std::min(1.0f, std::max(0.0f, rect.width())));
  rect.set_height(std::min(1.0f, std::max(0.0f, rect.height())));
  return rect;
}

cv::Rect ExpandMouthToFaceRect(const cv::Rect& mouth_roi,
                               const cv::Size& size) {
  const float scale_x = std::max(1.0f, absl::GetFlag(FLAGS_mouth_scale_x));
  const float scale_y = std::max(1.0f, absl::GetFlag(FLAGS_mouth_scale_y));
  const float shift_y = absl::GetFlag(FLAGS_mouth_shift_y);

  const float mouth_cx = mouth_roi.x + mouth_roi.width * 0.5f;
  const float mouth_cy = mouth_roi.y + mouth_roi.height * 0.5f;

  const float face_w = mouth_roi.width * scale_x;
  const float face_h = mouth_roi.height * scale_y;
  const float face_cx = mouth_cx;
  const float face_cy = mouth_cy - mouth_roi.height * shift_y;

  int x0 = static_cast<int>(std::round(face_cx - face_w * 0.5f));
  int y0 = static_cast<int>(std::round(face_cy - face_h * 0.5f));
  int x1 = static_cast<int>(std::round(face_cx + face_w * 0.5f));
  int y1 = static_cast<int>(std::round(face_cy + face_h * 0.5f));

  cv::Rect face_roi(cv::Point(x0, y0), cv::Point(x1, y1));
  face_roi &= cv::Rect(0, 0, size.width, size.height);
  if (face_roi.width < 1 || face_roi.height < 1) {
    return cv::Rect(0, 0, size.width, size.height);
  }
  return face_roi;
}

mediapipe::NormalizedRect MakeFullNormalizedRect() {
  mediapipe::NormalizedRect rect;
  rect.set_rotation(0.0f);
  rect.set_x_center(0.5f);
  rect.set_y_center(0.5f);
  rect.set_width(1.0f);
  rect.set_height(1.0f);
  return rect;
}

void UpdateRoiFromMouse(int event, int x, int y, int flags, void* userdata) {
  auto* roi_state = static_cast<RoiSelection*>(userdata);
  if (!roi_state) {
    return;
  }
  const cv::Point point(x, y);
  if (event == cv::EVENT_LBUTTONDOWN) {
    roi_state->dragging = true;
    roi_state->start = point;
    roi_state->current = point;
  } else if (event == cv::EVENT_MOUSEMOVE && roi_state->dragging) {
    roi_state->current = point;
  } else if (event == cv::EVENT_LBUTTONUP) {
    roi_state->dragging = false;
    roi_state->current = point;
    const int x0 = std::min(roi_state->start.x, roi_state->current.x);
    const int y0 = std::min(roi_state->start.y, roi_state->current.y);
    const int x1 = std::max(roi_state->start.x, roi_state->current.x);
    const int y1 = std::max(roi_state->start.y, roi_state->current.y);
    roi_state->roi = cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
    roi_state->has_roi = roi_state->roi.width > 4 && roi_state->roi.height > 4;
  }
}

mediapipe::CalculatorGraphConfig CreateGraphConfig(
    const std::string& model_path) {
  const std::string config_text = absl::StrFormat(
      R"pb(
        input_stream: "IMAGE:input_image"
        input_stream: "NORM_RECT:roi_rects"
        output_stream: "NORM_LANDMARKS:landmarks"
        output_stream: "BLENDSHAPES:blendshapes"
        node {
          calculator: "mediapipe.tasks.vision.face_landmarker.FaceLandmarkerRoiGraph"
          input_stream: "IMAGE:input_image"
          input_stream: "NORM_RECT:roi_rects"
          output_stream: "NORM_LANDMARKS:landmarks"
          output_stream: "BLENDSHAPES:blendshapes"
          options {
            [mediapipe.tasks.vision.face_landmarker.proto.FaceLandmarkerGraphOptions.ext] {
              base_options {
                model_asset { file_name: "%s" }
                use_stream_mode: true
              }
              face_detector_graph_options { num_faces: 1 }
              face_landmarks_detector_graph_options {
                min_detection_confidence: 0.0
                disable_presence_gating: true
              }
            }
          }
        }
      )pb",
      model_path);
  return mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
      config_text);
}

absl::Status RunAsyncMouthLandmarks() {
  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(CreateGraphConfig(model_path)));

  MP_ASSIGN_OR_RETURN(auto landmarks_poller,
                      graph.AddOutputStreamPoller(kLandmarksStream));
  MP_ASSIGN_OR_RETURN(auto blendshapes_poller,
                      graph.AddOutputStreamPoller(kBlendshapesStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  cv::VideoCapture capture;
  const int camera_id = absl::GetFlag(FLAGS_camera_id);
  const int backend = GetCameraBackend();
  ABSL_LOG(INFO) << "Opening camera_id=" << camera_id
                 << " with backend=" << absl::GetFlag(FLAGS_camera_backend);
  capture.open(camera_id, backend);
  capture.set(cv::CAP_PROP_FRAME_WIDTH, absl::GetFlag(FLAGS_camera_width));
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, absl::GetFlag(FLAGS_camera_height));
  capture.set(cv::CAP_PROP_FPS, 30);
  RET_CHECK(capture.isOpened());

  cv::namedWindow(kWindowName, 1);
  cv::namedWindow(kCropWindowName, 1);

  RoiSelection roi_state;
  cv::setMouseCallback(kWindowName, UpdateRoiFromMouse, &roi_state);

  SharedState state;
  std::thread inference_thread([&]() {
    using Clock = std::chrono::steady_clock;
    auto last_inference_time = Clock::now();
    auto last_blendshape_log = Clock::now();
    double smoothed_inference_fps = 0.0;

    while (true) {
      FrameData frame;
      {
        std::unique_lock<std::mutex> lock(state.frame_mutex);
        state.frame_cv.wait(lock, [&]() {
          return state.stop || state.has_pending_frame;
        });
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
      std::shared_ptr<mediapipe::ImageFrame> shared_frame(input_frame.release());
      mediapipe::Image image(shared_frame);

      const auto timestamp = mediapipe::Timestamp(frame.timestamp_us);
      auto status = graph.AddPacketToInputStream(
          kRoiRectsStream,
          mediapipe::MakePacket<std::vector<mediapipe::NormalizedRect>>(
              frame.roi_rects)
              .At(timestamp));
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to add ROI packet: " << status.message();
        continue;
      }
      status = graph.AddPacketToInputStream(
          kInputStream, mediapipe::MakePacket<mediapipe::Image>(image)
                            .At(timestamp));
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to add frame to graph: " << status.message();
        continue;
      }

      mediapipe::Packet landmarks_packet;
      mediapipe::Packet blendshapes_packet;
      if (!landmarks_poller.Next(&landmarks_packet) ||
          !blendshapes_poller.Next(&blendshapes_packet)) {
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
      state.inference_fps = smoothed_inference_fps;
      state.has_landmarks = !landmarks_packet.IsEmpty();
      if (state.has_landmarks) {
        const auto& lists =
            landmarks_packet
                .Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        if (!lists.empty()) {
          state.landmarks = lists[0];
        } else {
          state.has_landmarks = false;
        }
      }
      state.has_blendshapes = !blendshapes_packet.IsEmpty();
      if (state.has_blendshapes) {
        const auto& lists =
            blendshapes_packet.Get<std::vector<mediapipe::ClassificationList>>();
        if (!lists.empty()) {
          state.blendshapes = lists[0];
        } else {
          state.has_blendshapes = false;
        }
      }

      if (state.has_blendshapes) {
        const auto now = Clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_blendshape_log)
                .count();
        if (elapsed > 1000) {
          last_blendshape_log = now;
          std::string mouth_report = "Mouth blendshapes:";
          for (const auto& classification :
               state.blendshapes.classification()) {
            if (!classification.has_label()) {
              continue;
            }
            if (classification.label().rfind("mouth", 0) != 0) {
              continue;
            }
            absl::StrAppend(&mouth_report, " ", classification.label(), "=",
                            absl::StrFormat("%.2f", classification.score()));
          }
          ABSL_LOG(INFO) << mouth_report;
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
      continue;
    }
    cv::flip(camera_frame, camera_frame, 1);

    cv::Rect roi_px;
    if (absl::GetFlag(FLAGS_roi_mode) == "full") {
      roi_px = cv::Rect(0, 0, camera_frame.cols, camera_frame.rows);
    } else if (roi_state.has_roi) {
      roi_px = roi_state.roi & cv::Rect(0, 0, camera_frame.cols,
                                        camera_frame.rows);
    } else if (roi_state.dragging) {
      const int x0 = std::min(roi_state.start.x, roi_state.current.x);
      const int y0 = std::min(roi_state.start.y, roi_state.current.y);
      const int x1 = std::max(roi_state.start.x, roi_state.current.x);
      const int y1 = std::max(roi_state.start.y, roi_state.current.y);
      roi_px = cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
    } else {
      roi_px = cv::Rect(0, 0, camera_frame.cols, camera_frame.rows);
    }

    if (roi_px.width < 1 || roi_px.height < 1) {
      roi_px = cv::Rect(0, 0, camera_frame.cols, camera_frame.rows);
    }

    cv::Rect face_roi_px = roi_px;
    if (absl::GetFlag(FLAGS_roi_mode) == "mouse") {
      face_roi_px = ExpandMouthToFaceRect(roi_px, camera_frame.size());
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
    frame_data.roi_px = roi_px;
    frame_data.roi_rects = {MakeNormalizedRect(face_roi_px, camera_frame.size())};
    frame_data.timestamp_us = ++frame_counter * 33333;
    {
      std::lock_guard<std::mutex> lock(state.frame_mutex);
      state.pending_frame = std::move(frame_data);
      state.has_pending_frame = true;
    }
    state.frame_cv.notify_one();

    mediapipe::NormalizedLandmarkList landmarks;
    double inference_fps = 0.0;
    bool has_landmarks = false;
    {
      std::lock_guard<std::mutex> lock(state.result_mutex);
      has_landmarks = state.has_landmarks;
      inference_fps = state.inference_fps;
      if (has_landmarks) {
        landmarks = state.landmarks;
      }
    }

    if (has_landmarks) {
      DrawAllLandmarks(landmarks, camera_frame, camera_frame.size(),
                       cv::Point(0, 0));
    }
    if (absl::GetFlag(FLAGS_roi_mode) == "mouse") {
      cv::rectangle(camera_frame, face_roi_px, cv::Scalar(0, 255, 0), 2);
      cv::rectangle(camera_frame, roi_px, cv::Scalar(255, 128, 0), 2);
    } else {
      cv::rectangle(camera_frame, roi_px, cv::Scalar(0, 255, 0), 2);
    }
    DrawFps(camera_frame, display_fps, inference_fps);
    cv::imshow(kWindowName, camera_frame);

    cv::Mat crop_frame;
    if (roi_px.width > 0 && roi_px.height > 0) {
      crop_frame = camera_frame(roi_px).clone();
      if (has_landmarks) {
        DrawAllLandmarks(landmarks, crop_frame, camera_frame.size(),
                         cv::Point(-roi_px.x, -roi_px.y));
      }
    } else {
      crop_frame = cv::Mat(200, 200, CV_8UC3, cv::Scalar(10, 10, 10));
      cv::putText(crop_frame, "Select ROI", cv::Point(12, 110),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 2);
    }
    cv::imshow(kCropWindowName, crop_frame);

    const int pressed_key = cv::waitKey(1);
    if (pressed_key >= 0 && pressed_key != 255) {
      grab_frames = false;
    }
  }

  {
    std::lock_guard<std::mutex> lock(state.frame_mutex);
    state.stop = true;
  }
  state.frame_cv.notify_one();
  inference_thread.join();

  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kRoiRectsStream));
  return graph.WaitUntilDone();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const absl::Status status = RunAsyncMouthLandmarks();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << status;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
