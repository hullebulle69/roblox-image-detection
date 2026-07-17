#pragma once

#include <vector>

#include "common/types.hpp"

namespace rcd {

// Draws detection boxes with "label score%" tags into a BGRA frame.
// Box color is graded by confidence (green / yellow / orange).
void annotate(Frame& frame, const std::vector<Detection>& detections);

}  // namespace rcd
