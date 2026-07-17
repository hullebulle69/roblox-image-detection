// rcd_cli — run the Roblox character detector on images from disk.
//
// This shares the exact detector core used by the live overlay tool, so it
// doubles as the way to validate a model + backend before going live, and
// as a batch annotator.
//
//   rcd_cli shot.png --model roblox_yolo.onnx
//   rcd_cli shot.png --model m.onnx --out annotated.png --json result.json
//   rcd_cli shot.png --model m.onnx --backend cuda --bench 100

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common/types.hpp"
#include "detector/yolo_detector.hpp"
#include "render/draw.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace {

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " <image> --model <model.onnx> [options]\n"
        << "  --model PATH       YOLO ONNX model (required)\n"
        << "  --out PATH         write annotated copy (default: <image>.det.png)\n"
        << "  --no-out           skip writing the annotated image\n"
        << "  --json PATH        write detections as JSON ('-' for stdout)\n"
        << "  --conf F           confidence threshold (default 0.35)\n"
        << "  --iou F            NMS IoU threshold (default 0.45)\n"
        << "  --backend NAME     auto|tensorrt|cuda|directml|cpu (default auto)\n"
        << "  --device N         GPU device index (default 0)\n"
        << "  --classes a,b,c    only report these class names\n"
        << "  --max-det N        cap detections (default 100)\n"
        << "  --bench N          run N timed iterations and report latency/FPS\n";
}

bool load_frame(const std::string& path, rcd::Frame& frame) {
    int w = 0, h = 0, comp = 0;
    unsigned char* rgba = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!rgba) {
        std::cerr << "error: cannot load image '" << path
                  << "': " << stbi_failure_reason() << "\n";
        return false;
    }
    frame.resize(w, h);
    for (int y = 0; y < h; ++y) {
        const unsigned char* src = rgba + static_cast<size_t>(y) * w * 4;
        uint8_t* dst = frame.bgra.data() + static_cast<size_t>(y) * frame.stride;
        for (int x = 0; x < w; ++x) {  // RGBA -> BGRA
            dst[x * 4 + 0] = src[x * 4 + 2];
            dst[x * 4 + 1] = src[x * 4 + 1];
            dst[x * 4 + 2] = src[x * 4 + 0];
            dst[x * 4 + 3] = src[x * 4 + 3];
        }
    }
    stbi_image_free(rgba);
    return true;
}

bool save_frame_png(const std::string& path, const rcd::Frame& frame) {
    std::vector<unsigned char> rgba(static_cast<size_t>(frame.width) *
                                    frame.height * 4);
    for (int y = 0; y < frame.height; ++y) {
        const uint8_t* src =
            frame.bgra.data() + static_cast<size_t>(y) * frame.stride;
        unsigned char* dst =
            rgba.data() + static_cast<size_t>(y) * frame.width * 4;
        for (int x = 0; x < frame.width; ++x) {  // BGRA -> RGBA
            dst[x * 4 + 0] = src[x * 4 + 2];
            dst[x * 4 + 1] = src[x * 4 + 1];
            dst[x * 4 + 2] = src[x * 4 + 0];
            dst[x * 4 + 3] = 255;
        }
    }
    return stbi_write_png(path.c_str(), frame.width, frame.height, 4,
                          rgba.data(), frame.width * 4) != 0;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out += c;
        }
    }
    return out;
}

std::string detections_to_json(const std::string& image,
                               const std::vector<rcd::Detection>& dets,
                               const rcd::YoloDetector& detector) {
    std::string j = "{\n  \"image\": \"" + json_escape(image) + "\",\n";
    j += "  \"provider\": \"" + json_escape(detector.active_provider()) +
         "\",\n  \"detections\": [\n";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        char nums[160];
        std::snprintf(nums, sizeof(nums),
                      "\"score\": %.4f, \"box\": [%.1f, %.1f, %.1f, %.1f]",
                      d.score, d.box.x1, d.box.y1, d.box.x2, d.box.y2);
        j += "    {\"label\": \"" + json_escape(d.label) + "\", " + nums +
             "}" + (i + 1 < dets.size() ? "," : "") + "\n";
    }
    j += "  ]\n}\n";
    return j;
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        if (comma > start) out.push_back(s.substr(start, comma - start));
        start = comma + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string image_path, out_path, json_path;
    bool write_out = true;
    int bench = 0;
    rcd::DetectorOptions opts;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "error: " << what << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        try {
            if (arg == "--model") opts.model_path = next("--model");
            else if (arg == "--out") out_path = next("--out");
            else if (arg == "--no-out") write_out = false;
            else if (arg == "--json") json_path = next("--json");
            else if (arg == "--conf") opts.conf_threshold = std::stof(next("--conf"));
            else if (arg == "--iou") opts.iou_threshold = std::stof(next("--iou"));
            else if (arg == "--backend")
                opts.backend = rcd::backend_from_string(next("--backend"));
            else if (arg == "--device") opts.device_id = std::stoi(next("--device"));
            else if (arg == "--classes") opts.class_filter = split_csv(next("--classes"));
            else if (arg == "--max-det") opts.max_detections = std::stoi(next("--max-det"));
            else if (arg == "--bench") bench = std::stoi(next("--bench"));
            else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            } else if (!arg.empty() && arg[0] == '-') {
                std::cerr << "error: unknown option " << arg << "\n";
                usage(argv[0]);
                return 2;
            } else if (image_path.empty()) {
                image_path = arg;
            } else {
                std::cerr << "error: multiple images given; run once per image\n";
                return 2;
            }
        } catch (const std::exception& e) {
            std::cerr << "error: bad value for " << arg << ": " << e.what()
                      << "\n";
            return 2;
        }
    }
    if (image_path.empty() || opts.model_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    rcd::Frame frame;
    if (!load_frame(image_path, frame)) return 1;

    try {
        rcd::YoloDetector detector(opts);
        std::cerr << "[rcd] provider: " << detector.active_provider()
                  << ", model input " << detector.input_width() << "x"
                  << detector.input_height() << "\n";

        auto detections = detector.detect(frame);

        // Keep stdout clean for machine consumption when JSON goes there.
        FILE* table = json_path == "-" ? stderr : stdout;

        if (bench > 0) {
            detector.warmup(frame.width, frame.height);
            double pre = 0, inf = 0, post = 0;
            for (int i = 0; i < bench; ++i) {
                (void)detector.detect(frame);
                const auto& st = detector.last_stats();
                pre += st.preprocess_ms;
                inf += st.inference_ms;
                post += st.postprocess_ms;
            }
            const double total = (pre + inf + post) / bench;
            std::fprintf(
                table,
                "bench (%d iters): pre %.2f ms | infer %.2f ms | post %.2f ms "
                "| total %.2f ms | %.1f FPS\n",
                bench, pre / bench, inf / bench, post / bench, total,
                1000.0 / total);
        }
        for (const auto& d : detections)
            std::fprintf(table, "%-24s %5.1f%%  [%7.1f, %7.1f, %7.1f, %7.1f]\n",
                         d.label.c_str(), d.score * 100.f, d.box.x1, d.box.y1,
                         d.box.x2, d.box.y2);
        if (detections.empty())
            std::fprintf(table, "no detections above %.2f confidence\n",
                         opts.conf_threshold);

        if (!json_path.empty()) {
            const std::string j =
                detections_to_json(image_path, detections, detector);
            if (json_path == "-") {
                std::cout << j;
            } else {
                std::ofstream f(json_path);
                f << j;
                if (!f) {
                    std::cerr << "error: cannot write " << json_path << "\n";
                    return 1;
                }
            }
        }

        if (write_out) {
            if (out_path.empty()) out_path = image_path + ".det.png";
            rcd::annotate(frame, detections);
            if (!save_frame_png(out_path, frame)) {
                std::cerr << "error: cannot write " << out_path << "\n";
                return 1;
            }
            std::cerr << "[rcd] annotated image: " << out_path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
