#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace rcd {

// Axis-aligned box in source-image pixel coordinates.
struct BBox {
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const {
        return std::max(0.f, width()) * std::max(0.f, height());
    }
};

inline float iou(const BBox& a, const BBox& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float inter =
        std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    const float uni = a.area() + b.area() - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

struct Detection {
    BBox box;
    float score = 0.f;  // confidence in [0, 1]
    int class_id = 0;
    std::string label;
};

// A CPU-side 8-bit BGRA frame. `stride` is bytes per row (>= width * 4).
struct Frame {
    int width = 0;
    int height = 0;
    int stride = 0;
    std::vector<uint8_t> bgra;
    uint64_t timestamp_us = 0;

    void resize(int w, int h) {
        width = w;
        height = h;
        stride = w * 4;
        bgra.resize(static_cast<size_t>(stride) * h);
    }
    bool empty() const { return width <= 0 || height <= 0 || bgra.empty(); }
};

}  // namespace rcd
