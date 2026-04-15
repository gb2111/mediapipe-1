// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

class SingleNormalizedRectToRectVectorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<NormalizedRect>();
    cc->Outputs().Index(0).Set<std::vector<NormalizedRect>>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    auto output = absl::make_unique<std::vector<NormalizedRect>>();
    output->push_back(cc->Inputs().Index(0).Get<NormalizedRect>());
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(SingleNormalizedRectToRectVectorCalculator);

}  // namespace mediapipe
