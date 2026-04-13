/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/core/clip_vector_size_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_blendshapes_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/util/graph_builder_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_landmarker {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::core::ModelAssetBundleResources;
using ::mediapipe::tasks::metadata::SetExternalFile;
using ::mediapipe::tasks::vision::face_detector::proto::
    FaceDetectorGraphOptions;
using ::mediapipe::tasks::vision::face_landmarker::proto::
    FaceLandmarkerGraphOptions;
using ::mediapipe::tasks::vision::face_landmarker::proto::
    FaceLandmarksDetectorGraphOptions;

constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kBlendshapesTag[] = "BLENDSHAPES";
constexpr char kFaceDetectorTFLiteName[] = "face_detector.tflite";
constexpr char kFaceLandmarksDetectorTFLiteName[] =
    "face_landmarks_detector.tflite";
constexpr char kFaceBlendshapeTFLiteName[] = "face_blendshapes.tflite";

struct FaceLandmarkerRoiOutputs {
  Source<std::vector<NormalizedLandmarkList>> landmark_lists;
  std::optional<Source<std::vector<ClassificationList>>> face_blendshapes;
};

absl::Status SetSubTaskBaseOptions(const ModelAssetBundleResources& resources,
                                   FaceLandmarkerGraphOptions* options,
                                   bool is_copy) {
  auto* face_detector_graph_options =
      options->mutable_face_detector_graph_options();
  if (!face_detector_graph_options->base_options().has_model_asset()) {
    MP_ASSIGN_OR_RETURN(const auto face_detector_file,
                        resources.GetFile(kFaceDetectorTFLiteName));
    SetExternalFile(face_detector_file,
                    face_detector_graph_options->mutable_base_options()
                        ->mutable_model_asset(),
                    is_copy);
  }
  face_detector_graph_options->mutable_base_options()
      ->mutable_acceleration()
      ->CopyFrom(options->base_options().acceleration());
  face_detector_graph_options->mutable_base_options()->set_use_stream_mode(
      options->base_options().use_stream_mode());
  face_detector_graph_options->mutable_base_options()->set_gpu_origin(
      options->base_options().gpu_origin());

  auto* face_landmarks_detector_graph_options =
      options->mutable_face_landmarks_detector_graph_options();
  if (!face_landmarks_detector_graph_options->base_options()
           .has_model_asset()) {
    MP_ASSIGN_OR_RETURN(const auto face_landmarks_detector_file,
                        resources.GetFile(kFaceLandmarksDetectorTFLiteName));
    SetExternalFile(
        face_landmarks_detector_file,
        face_landmarks_detector_graph_options->mutable_base_options()
            ->mutable_model_asset(),
        is_copy);
  }
  face_landmarks_detector_graph_options->mutable_base_options()
      ->mutable_acceleration()
      ->CopyFrom(options->base_options().acceleration());
  face_landmarks_detector_graph_options->mutable_base_options()
      ->set_use_stream_mode(options->base_options().use_stream_mode());
  face_landmarks_detector_graph_options->mutable_base_options()->set_gpu_origin(
      options->base_options().gpu_origin());

  absl::StatusOr<absl::string_view> face_blendshape_model =
      resources.GetFile(kFaceBlendshapeTFLiteName);
  if (face_blendshape_model.ok()) {
    SetExternalFile(*face_blendshape_model,
                    face_landmarks_detector_graph_options
                        ->mutable_face_blendshapes_graph_options()
                        ->mutable_base_options()
                        ->mutable_model_asset(),
                    is_copy);
    face_landmarks_detector_graph_options
        ->mutable_face_blendshapes_graph_options()
        ->mutable_base_options()
        ->mutable_acceleration()
        ->mutable_xnnpack();
    ABSL_LOG(WARNING) << "Sets FaceBlendshapesGraph acceleration to xnnpack "
                      << "by default.";
  }

  return absl::OkStatus();
}

}  // namespace

// A "mediapipe.tasks.vision.face_landmarker.FaceLandmarkerRoiGraph" performs
// face landmarks detection using caller-provided RoIs.
//
// Inputs:
//   IMAGE - Image
//     Image to perform face landmarks detection on.
//   NORM_RECT - std::vector<NormalizedRect>
//     Vector of RoIs to run the face landmarks detector on. The graph does not
//     perform face detection.
//
// Outputs:
//   NORM_LANDMARKS - std::vector<NormalizedLandmarkList>
//     Vector of detected face landmarks.
//   BLENDSHAPES - std::vector<ClassificationList> @optional
//     Blendshape classification, available when the given model asset contains
//     a blendshapes model.
class FaceLandmarkerRoiGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    if (sc->Options<FaceLandmarkerGraphOptions>()
            .base_options()
            .has_model_asset()) {
      MP_ASSIGN_OR_RETURN(
          const auto* model_asset_bundle_resources,
          CreateModelAssetBundleResources<FaceLandmarkerGraphOptions>(sc));
      MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
          *model_asset_bundle_resources,
          sc->MutableOptions<FaceLandmarkerGraphOptions>(),
          !sc->Service(::mediapipe::tasks::core::kModelResourcesCacheService)
               .IsAvailable()));
    }

    bool output_blendshapes = HasOutput(sc->OriginalNode(), kBlendshapesTag);
    if (output_blendshapes && !sc->Options<FaceLandmarkerGraphOptions>()
                                   .face_landmarks_detector_graph_options()
                                   .has_face_blendshapes_graph_options()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "BLENDSHAPES Tag and blendshapes model must be both set. Get "
          "BLENDSHAPES is set: %v, blendshapes model is set: %v",
          output_blendshapes,
          sc->Options<FaceLandmarkerGraphOptions>()
              .face_landmarks_detector_graph_options()
              .has_face_blendshapes_graph_options()));
    }

    MP_ASSIGN_OR_RETURN(
        auto outs,
        BuildFaceLandmarkerRoiGraph(
            *sc->MutableOptions<FaceLandmarkerGraphOptions>(),
            graph[Input<Image>(kImageTag)],
            graph[Input<std::vector<NormalizedRect>>(kNormRectTag)],
            output_blendshapes, graph));
    outs.landmark_lists >>
        graph[Output<std::vector<NormalizedLandmarkList>>(kNormLandmarksTag)];
    if (outs.face_blendshapes) {
      *outs.face_blendshapes >>
          graph[Output<std::vector<ClassificationList>>(kBlendshapesTag)];
    }

    return graph.GetConfig();
  }

 private:
  absl::StatusOr<FaceLandmarkerRoiOutputs> BuildFaceLandmarkerRoiGraph(
      FaceLandmarkerGraphOptions& tasks_options, Source<Image> image_in,
      Source<std::vector<NormalizedRect>> norm_rects_in, bool output_blendshapes,
      Graph& graph) {
    int max_num_faces = tasks_options.face_detector_graph_options().num_faces();
    if (max_num_faces <= 0) {
      max_num_faces = 1;
    }

    auto& clip_face_rects =
        graph.AddNode("ClipNormalizedRectVectorSizeCalculator");
    clip_face_rects.GetOptions<ClipVectorSizeCalculatorOptions>()
        .set_max_vec_size(max_num_faces);
    norm_rects_in >> clip_face_rects.In("");
    auto clipped_face_rects = clip_face_rects.Out("");

    auto& face_landmarks_detector_graph = graph.AddNode(
        "mediapipe.tasks.vision.face_landmarker."
        "MultiFaceLandmarksDetectorGraph");
    face_landmarks_detector_graph
        .GetOptions<FaceLandmarksDetectorGraphOptions>()
        .Swap(tasks_options.mutable_face_landmarks_detector_graph_options());
    if (tasks_options.base_options().use_stream_mode() && max_num_faces == 1) {
      face_landmarks_detector_graph
          .GetOptions<FaceLandmarksDetectorGraphOptions>()
          .set_smooth_landmarks(true);
    }
    image_in >> face_landmarks_detector_graph.In(kImageTag);
    clipped_face_rects >> face_landmarks_detector_graph.In(kNormRectTag);

    std::optional<Source<std::vector<ClassificationList>>> blendshapes;
    if (output_blendshapes) {
      blendshapes = std::make_optional<>(
          face_landmarks_detector_graph.Out(kBlendshapesTag)
              .Cast<std::vector<ClassificationList>>());
    }

    return {FaceLandmarkerRoiOutputs{
        /* landmark_lists= */
        face_landmarks_detector_graph.Out(kNormLandmarksTag)
            .Cast<std::vector<NormalizedLandmarkList>>(),
        /* face_blendshapes= */ blendshapes,
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerRoiGraph);

}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
