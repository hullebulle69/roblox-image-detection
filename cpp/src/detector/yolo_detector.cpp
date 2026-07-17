#include "detector/yolo_detector.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <windows.h>
// dml_provider_factory.h transitively needs DirectML.h (Windows SDK 1903+),
// which MinGW does not ship — require both before enabling the DML path.
#if __has_include(<dml_provider_factory.h>) && __has_include(<DirectML.h>)
#define RCD_HAS_DML 1
#include <dml_provider_factory.h>
#endif
#endif

namespace rcd {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0)
        .count();
}

std::basic_string<ORTCHAR_T> to_ort_path(const std::string& utf8) {
#ifdef _WIN32
    if (utf8.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(),
                                      static_cast<int>(utf8.size()),
                                      nullptr, 0);
    std::wstring wide(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(),
                        static_cast<int>(utf8.size()), wide.data(), n);
    return wide;
#else
    return utf8;
#endif
}

// Parses ultralytics-style class-name metadata:  {0: 'person', 1: "bus"}
std::vector<std::string> parse_names_metadata(const std::string& raw) {
    std::vector<std::string> names;
    size_t i = 0;
    while (i < raw.size()) {
        while (i < raw.size() && !std::isdigit(static_cast<unsigned char>(raw[i])))
            ++i;
        if (i >= raw.size()) break;
        size_t j = i;
        while (j < raw.size() && std::isdigit(static_cast<unsigned char>(raw[j])))
            ++j;
        const int id = std::stoi(raw.substr(i, j - i));
        i = j;
        while (i < raw.size() && (raw[i] == ' ' || raw[i] == ':')) ++i;
        if (i >= raw.size() || (raw[i] != '\'' && raw[i] != '"')) continue;
        const char quote = raw[i++];
        std::string name;
        while (i < raw.size() && raw[i] != quote) {
            if (raw[i] == '\\' && i + 1 < raw.size()) ++i;
            name += raw[i++];
        }
        ++i;  // closing quote
        if (id >= 0) {
            if (names.size() <= static_cast<size_t>(id))
                names.resize(static_cast<size_t>(id) + 1);
            names[static_cast<size_t>(id)] = name;
        }
    }
    return names;
}

// Runs f(begin_row, end_row) on `threads` slices of [0, rows).
template <typename F>
void parallel_rows(int rows, F&& f) {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int threads = std::clamp(std::min(hw, rows / 64), 1, 8);
    if (threads <= 1) {
        f(0, rows);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads) - 1);
    const int chunk = (rows + threads - 1) / threads;
    for (int t = 1; t < threads; ++t) {
        const int b = t * chunk;
        const int e = std::min(rows, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&f, b, e] { f(b, e); });
    }
    f(0, std::min(rows, chunk));
    for (auto& th : pool) th.join();
}

struct LetterboxParams {
    float scale = 1.f;
    int pad_x = 0;
    int pad_y = 0;
};

// BGRA frame -> normalized RGB CHW float tensor with ultralytics-compatible
// letterboxing (center placement, 114-gray padding, bilinear resample).
LetterboxParams letterbox_to_tensor(const Frame& frame, int in_w, int in_h,
                                    std::vector<float>& out) {
    LetterboxParams lb;
    lb.scale = std::min(static_cast<float>(in_w) / frame.width,
                        static_cast<float>(in_h) / frame.height);
    const int new_w = static_cast<int>(std::lround(frame.width * lb.scale));
    const int new_h = static_cast<int>(std::lround(frame.height * lb.scale));
    const float dw = (in_w - new_w) / 2.f;
    const float dh = (in_h - new_h) / 2.f;
    lb.pad_x = static_cast<int>(std::lround(dw - 0.1f));
    lb.pad_y = static_cast<int>(std::lround(dh - 0.1f));

    const size_t plane = static_cast<size_t>(in_w) * in_h;
    out.assign(plane * 3, 114.f / 255.f);

    const float inv_scale_x = static_cast<float>(frame.width) / new_w;
    const float inv_scale_y = static_cast<float>(frame.height) / new_h;
    const uint8_t* src = frame.bgra.data();
    const int src_stride = frame.stride;
    const int src_w = frame.width;
    const int src_h = frame.height;
    float* r_plane = out.data();
    float* g_plane = out.data() + plane;
    float* b_plane = out.data() + 2 * plane;
    const int pad_x = lb.pad_x;
    const int pad_y = lb.pad_y;

    parallel_rows(new_h, [&](int row_begin, int row_end) {
        for (int y = row_begin; y < row_end; ++y) {
            const float sy = (y + 0.5f) * inv_scale_y - 0.5f;
            const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, src_h - 1);
            const int y1 = std::min(y0 + 1, src_h - 1);
            const float fy = std::clamp(sy - y0, 0.f, 1.f);
            const uint8_t* row0 = src + static_cast<size_t>(y0) * src_stride;
            const uint8_t* row1 = src + static_cast<size_t>(y1) * src_stride;
            const size_t dst_row = static_cast<size_t>(y + pad_y) * in_w;
            for (int x = 0; x < new_w; ++x) {
                const float sx = (x + 0.5f) * inv_scale_x - 0.5f;
                const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, src_w - 1);
                const int x1 = std::min(x0 + 1, src_w - 1);
                const float fx = std::clamp(sx - x0, 0.f, 1.f);
                const float w00 = (1.f - fx) * (1.f - fy);
                const float w10 = fx * (1.f - fy);
                const float w01 = (1.f - fx) * fy;
                const float w11 = fx * fy;
                const uint8_t* p00 = row0 + static_cast<size_t>(x0) * 4;
                const uint8_t* p10 = row0 + static_cast<size_t>(x1) * 4;
                const uint8_t* p01 = row1 + static_cast<size_t>(x0) * 4;
                const uint8_t* p11 = row1 + static_cast<size_t>(x1) * 4;
                const float b = w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0];
                const float g = w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1];
                const float r = w00 * p00[2] + w10 * p10[2] + w01 * p01[2] + w11 * p11[2];
                const size_t di = dst_row + static_cast<size_t>(x + pad_x);
                r_plane[di] = r * (1.f / 255.f);
                g_plane[di] = g * (1.f / 255.f);
                b_plane[di] = b * (1.f / 255.f);
            }
        }
    });
    return lb;
}

std::vector<Detection> nms(std::vector<Detection>& candidates,
                           float iou_threshold, int max_detections) {
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });
    std::vector<Detection> kept;
    std::vector<bool> suppressed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(candidates[i]);
        if (static_cast<int>(kept.size()) >= max_detections) break;
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            if (candidates[j].class_id != candidates[i].class_id) continue;
            if (iou(candidates[i].box, candidates[j].box) > iou_threshold)
                suppressed[j] = true;
        }
    }
    return kept;
}

}  // namespace

Backend backend_from_string(const std::string& name) {
    std::string s;
    for (char c : name) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "auto") return Backend::Auto;
    if (s == "tensorrt" || s == "trt") return Backend::TensorRT;
    if (s == "cuda") return Backend::CUDA;
    if (s == "directml" || s == "dml") return Backend::DirectML;
    if (s == "cpu") return Backend::CPU;
    throw std::invalid_argument("unknown backend: " + name +
                                " (use auto|tensorrt|cuda|directml|cpu)");
}

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::Auto: return "auto";
        case Backend::TensorRT: return "TensorRT";
        case Backend::CUDA: return "CUDA";
        case Backend::DirectML: return "DirectML";
        case Backend::CPU: return "CPU";
    }
    return "?";
}

struct YoloDetector::Impl {
    DetectorOptions opts;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "rcd"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::string input_name, output_name;
    int input_w = 640, input_h = 640;
    std::string provider = "CPU";
    std::vector<std::string> names;
    std::vector<char> class_allowed;  // by id; empty = all allowed
    std::vector<float> input_buffer;
    DetectorStats stats;

    explicit Impl(const DetectorOptions& options) : opts(options) {
        configure_session_options();
        session = std::make_unique<Ort::Session>(
            env, to_ort_path(opts.model_path).c_str(), session_options);
        read_model_io();
        resolve_class_names();
    }

    void configure_session_options() {
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (opts.intra_op_threads > 0)
            session_options.SetIntraOpNumThreads(opts.intra_op_threads);

        const bool want_auto = opts.backend == Backend::Auto;
        auto want = [&](Backend b) {
            return want_auto || opts.backend == b;
        };
        std::string tried;

        if (want(Backend::TensorRT) && try_tensorrt(tried)) return;
        if (want(Backend::CUDA) && try_cuda(tried)) return;
        if (want(Backend::DirectML) && try_dml(tried)) return;

        if (!want_auto && opts.backend != Backend::CPU) {
            throw std::runtime_error(
                std::string("requested backend '") + backend_name(opts.backend) +
                "' is unavailable in this ONNX Runtime build/machine" +
                (tried.empty() ? "" : " (" + tried + ")"));
        }
        provider = "CPU";
        if (!tried.empty())
            std::cerr << "[rcd] GPU providers unavailable (" << tried
                      << "); falling back to CPU.\n";
    }

    bool try_tensorrt(std::string& tried) {
        try {
            OrtTensorRTProviderOptions trt{};
            trt.device_id = opts.device_id;
            trt.trt_fp16_enable = 1;
            trt.trt_engine_cache_enable = 1;
            trt.trt_engine_cache_path = "trt_engine_cache";
            trt.trt_max_workspace_size = 2ULL << 30;
            session_options.AppendExecutionProvider_TensorRT(trt);
            provider = "TensorRT";
            return true;
        } catch (const Ort::Exception& e) {
            tried += std::string(tried.empty() ? "" : "; ") + "TensorRT: " + e.what();
            return false;
        }
    }

    bool try_cuda(std::string& tried) {
        try {
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = opts.device_id;
            session_options.AppendExecutionProvider_CUDA(cuda);
            provider = "CUDA";
            return true;
        } catch (const Ort::Exception& e) {
            tried += std::string(tried.empty() ? "" : "; ") + "CUDA: " + e.what();
            return false;
        }
    }

    bool try_dml(std::string& tried) {
#ifdef RCD_HAS_DML
        try {
            // DirectML requires sequential execution and no memory pattern.
            session_options.DisableMemPattern();
            session_options.SetExecutionMode(ORT_SEQUENTIAL);
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(
                session_options, opts.device_id));
            provider = "DirectML";
            return true;
        } catch (const Ort::Exception& e) {
            tried += std::string(tried.empty() ? "" : "; ") + "DirectML: " + e.what();
            return false;
        }
#else
        tried += std::string(tried.empty() ? "" : "; ") +
                 "DirectML: not compiled in";
        return false;
#endif
    }

    void read_model_io() {
        Ort::AllocatorWithDefaultOptions alloc;
        if (session->GetInputCount() < 1 || session->GetOutputCount() < 1)
            throw std::runtime_error("model must have >=1 input and output");
        input_name = session->GetInputNameAllocated(0, alloc).get();
        output_name = session->GetOutputNameAllocated(0, alloc).get();

        // Keep the TypeInfo alive: GetTensorTypeAndShapeInfo() returns a
        // non-owning view into it.
        const Ort::TypeInfo type_info = session->GetInputTypeInfo(0);
        const auto info = type_info.GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (shape.size() != 4)
            throw std::runtime_error("expected NCHW image input, got rank " +
                                     std::to_string(shape.size()));
        input_h = shape[2] > 0 ? static_cast<int>(shape[2])
                               : opts.fallback_input_size;
        input_w = shape[3] > 0 ? static_cast<int>(shape[3])
                               : opts.fallback_input_size;
    }

    void resolve_class_names() {
        if (!opts.class_names.empty()) {
            names = opts.class_names;
        } else {
            Ort::AllocatorWithDefaultOptions alloc;
            const Ort::ModelMetadata meta = session->GetModelMetadata();
            if (auto raw = meta.LookupCustomMetadataMapAllocated("names", alloc))
                names = parse_names_metadata(raw.get());
        }
        if (!opts.class_filter.empty()) {
            class_allowed.assign(std::max<size_t>(names.size(), 1), 0);
            for (const auto& want : opts.class_filter) {
                bool found = false;
                for (size_t i = 0; i < names.size(); ++i) {
                    if (names[i] == want) {
                        class_allowed[i] = 1;
                        found = true;
                    }
                }
                if (!found)
                    std::cerr << "[rcd] warning: class filter '" << want
                              << "' does not match any model class\n";
            }
        }
    }

    std::string label_for(int class_id) const {
        if (class_id >= 0 && static_cast<size_t>(class_id) < names.size() &&
            !names[static_cast<size_t>(class_id)].empty())
            return names[static_cast<size_t>(class_id)];
        return "class" + std::to_string(class_id);
    }

    bool allowed(int class_id) const {
        if (class_allowed.empty()) return true;
        return class_id >= 0 &&
               static_cast<size_t>(class_id) < class_allowed.size() &&
               class_allowed[static_cast<size_t>(class_id)];
    }

    std::vector<Detection> detect(const Frame& frame) {
        if (frame.empty()) return {};

        auto t0 = Clock::now();
        const LetterboxParams lb =
            letterbox_to_tensor(frame, input_w, input_h, input_buffer);
        stats.preprocess_ms = ms_since(t0);

        t0 = Clock::now();
        const std::array<int64_t, 4> in_shape{1, 3, input_h, input_w};
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory_info, input_buffer.data(), input_buffer.size(),
            in_shape.data(), in_shape.size());
        const char* in_names[] = {input_name.c_str()};
        const char* out_names[] = {output_name.c_str()};
        auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, &input,
                                    1, out_names, 1);
        stats.inference_ms = ms_since(t0);

        t0 = Clock::now();
        auto detections = decode(outputs[0], frame, lb);
        stats.postprocess_ms = ms_since(t0);
        return detections;
    }

    std::vector<Detection> decode(const Ort::Value& output, const Frame& frame,
                                  const LetterboxParams& lb) const {
        const auto info = output.GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (shape.size() != 3 || shape[0] != 1)
            throw std::runtime_error("unexpected output shape (want [1,A,B])");
        const float* data = output.GetTensorData<float>();

        const int64_t dim1 = shape[1];
        const int64_t dim2 = shape[2];
        // YOLOv8/11: [1, 4+nc, anchors] with attributes down dim1;
        // YOLOv5/7:  [1, anchors, 5+nc] with objectness.
        const bool attrs_first = dim1 < dim2;
        const int64_t num_anchors = attrs_first ? dim2 : dim1;
        const int64_t num_attrs = attrs_first ? dim1 : dim2;
        const bool has_objectness = !attrs_first;
        const int num_classes =
            static_cast<int>(num_attrs - (has_objectness ? 5 : 4));
        if (num_classes < 1)
            throw std::runtime_error("output has no class scores");

        auto attr = [&](int64_t anchor, int64_t a) {
            return attrs_first ? data[a * num_anchors + anchor]
                               : data[anchor * num_attrs + a];
        };

        std::vector<Detection> candidates;
        for (int64_t i = 0; i < num_anchors; ++i) {
            float obj = 1.f;
            if (has_objectness) {
                obj = attr(i, 4);
                if (obj < opts.conf_threshold) continue;
            }
            int best_class = -1;
            float best_score = 0.f;
            const int64_t class_base = has_objectness ? 5 : 4;
            for (int c = 0; c < num_classes; ++c) {
                const float s = attr(i, class_base + c);
                if (s > best_score) {
                    best_score = s;
                    best_class = c;
                }
            }
            const float score = obj * best_score;
            if (score < opts.conf_threshold || !allowed(best_class)) continue;

            const float cx = attr(i, 0);
            const float cy = attr(i, 1);
            const float w = attr(i, 2);
            const float h = attr(i, 3);

            Detection det;
            det.box.x1 = (cx - w * 0.5f - lb.pad_x) / lb.scale;
            det.box.y1 = (cy - h * 0.5f - lb.pad_y) / lb.scale;
            det.box.x2 = (cx + w * 0.5f - lb.pad_x) / lb.scale;
            det.box.y2 = (cy + h * 0.5f - lb.pad_y) / lb.scale;
            det.box.x1 = std::clamp(det.box.x1, 0.f, static_cast<float>(frame.width));
            det.box.y1 = std::clamp(det.box.y1, 0.f, static_cast<float>(frame.height));
            det.box.x2 = std::clamp(det.box.x2, 0.f, static_cast<float>(frame.width));
            det.box.y2 = std::clamp(det.box.y2, 0.f, static_cast<float>(frame.height));
            if (det.box.width() < 1.f || det.box.height() < 1.f) continue;
            det.score = score;
            det.class_id = best_class;
            det.label = label_for(best_class);
            candidates.push_back(std::move(det));
        }
        return nms(candidates, opts.iou_threshold, opts.max_detections);
    }
};

YoloDetector::YoloDetector(const DetectorOptions& options)
    : impl_(std::make_unique<Impl>(options)) {}

YoloDetector::~YoloDetector() = default;

std::vector<Detection> YoloDetector::detect(const Frame& frame) {
    return impl_->detect(frame);
}

void YoloDetector::warmup(int width, int height) {
    Frame blank;
    blank.resize(std::max(width, 8), std::max(height, 8));
    (void)impl_->detect(blank);
}

const std::string& YoloDetector::active_provider() const {
    return impl_->provider;
}

const std::vector<std::string>& YoloDetector::class_names() const {
    return impl_->names;
}

const DetectorStats& YoloDetector::last_stats() const { return impl_->stats; }

int YoloDetector::input_width() const { return impl_->input_w; }
int YoloDetector::input_height() const { return impl_->input_h; }

}  // namespace rcd
