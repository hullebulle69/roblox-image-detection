#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/types.hpp"

namespace rcd {

// Which ONNX Runtime execution provider to use. `Auto` walks the list
// TensorRT -> CUDA -> DirectML -> CPU and keeps the first one that loads.
enum class Backend { Auto, TensorRT, CUDA, DirectML, CPU };

Backend backend_from_string(const std::string& name);
const char* backend_name(Backend b);

struct DetectorOptions {
    std::string model_path;
    float conf_threshold = 0.35f;
    float iou_threshold = 0.45f;
    int max_detections = 100;
    Backend backend = Backend::Auto;
    int device_id = 0;
    // Used only if the model has dynamic spatial dims.
    int fallback_input_size = 640;
    // Restrict results to these class names (empty = all classes).
    std::vector<std::string> class_filter;
    // Override class names (empty = read from model metadata).
    std::vector<std::string> class_names;
    int intra_op_threads = 0;  // 0 = ONNX Runtime default
};

struct DetectorStats {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms() const {
        return preprocess_ms + inference_ms + postprocess_ms;
    }
};

// YOLO-family single-image detector on ONNX Runtime.
//
// Supports the two mainstream exported output layouts:
//   * YOLOv8/v11:      [1, 4+nc, N]   (no objectness, attributes-first)
//   * YOLOv5/v7-style: [1, N, 5+nc]   (objectness * class score)
// Boxes are decoded from letterboxed input space back to source pixels and
// filtered with class-aware greedy NMS.
//
// Thread safety: detect() may be called from one thread at a time.
class YoloDetector {
public:
    explicit YoloDetector(const DetectorOptions& options);
    ~YoloDetector();
    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;

    std::vector<Detection> detect(const Frame& frame);

    // Runs one inference on a blank frame so first-frame latency (JIT,
    // engine build, allocation) is paid up front.
    void warmup(int width, int height);

    const std::string& active_provider() const;
    const std::vector<std::string>& class_names() const;
    const DetectorStats& last_stats() const;
    int input_width() const;
    int input_height() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rcd
