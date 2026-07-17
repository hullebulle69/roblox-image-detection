// rcd_live — real-time Roblox character detection overlay for Windows.
//
// Pipeline: DXGI desktop duplication captures the monitor -> YOLO ONNX
// inference on the GPU (TensorRT / CUDA / DirectML, CPU fallback) -> a
// click-through DirectComposition overlay draws confidence-scored boxes.
//
// This tool only *visualizes* what is already on your screen. It does not
// read game memory, send input, aim, or click — and it never will.
//
//   rcd_live --model roblox_yolo.onnx
//   rcd_live --model m.onnx --backend directml --conf 0.4 --monitor 1

#ifdef _WIN32

#include <windows.h>

#include <atomic>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/types.hpp"
#include "detector/yolo_detector.hpp"
#include "win/dxgi_capture.hpp"
#include "win/overlay_window.hpp"

namespace {

constexpr int kHotkeyToggle = 1;  // F8
constexpr int kHotkeyPause = 2;   // F9
constexpr int kHotkeyQuit = 3;    // F10

struct SharedResults {
    std::mutex mutex;
    std::vector<rcd::Detection> detections;
    double inference_ms = 0.0;
    double fps = 0.0;
    uint64_t seq = 0;
    ULONGLONG updated_at_ms = 0;
};

struct AppFlags {
    std::atomic<bool> running{true};
    std::atomic<bool> paused{false};
    std::atomic<bool> failed{false};
};

void usage() {
    std::cerr
        << "Usage: rcd_live --model <model.onnx> [options]\n"
        << "  --model PATH     YOLO ONNX model trained for Roblox characters\n"
        << "  --conf F         confidence threshold (default 0.35)\n"
        << "  --iou F          NMS IoU threshold (default 0.45)\n"
        << "  --backend NAME   auto|tensorrt|cuda|directml|cpu (default auto)\n"
        << "  --device N       GPU device index (default 0)\n"
        << "  --monitor N      monitor index to watch (default 0 = primary)\n"
        << "  --classes a,b    only draw these class names\n"
        << "  --max-det N      cap detections per frame (default 100)\n"
        << "\nHotkeys: F8 = show/hide overlay, F9 = pause, F10 = quit\n";
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

void enable_per_monitor_dpi() {
    // Physical-pixel coordinates everywhere, so overlay boxes line up with
    // captured pixels on scaled displays. Loaded dynamically to keep the
    // binary working on older Windows 10 builds.
    if (HMODULE user32 = GetModuleHandleW(L"user32.dll")) {
        using Fn = BOOL(WINAPI*)(DPI_AWARENESS_CONTEXT);
        if (auto fn = reinterpret_cast<Fn>(reinterpret_cast<void*>(
                GetProcAddress(user32, "SetProcessDpiAwarenessContext")))) {
            if (fn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) return;
        }
    }
    SetProcessDPIAware();
}

void detection_loop(rcd::DxgiScreenCapture& capture,
                    rcd::YoloDetector& detector, SharedResults& shared,
                    AppFlags& flags) {
    rcd::Frame frame;
    double fps_ema = 0.0;
    ULONGLONG last_frame_ms = GetTickCount64();
    int consecutive_errors = 0;
    const int overlay_w = capture.width();
    const int overlay_h = capture.height();
    bool warned_size_change = false;

    while (flags.running.load(std::memory_order_relaxed)) {
        if (flags.paused.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        const rcd::CaptureStatus status = capture.grab(frame);
        if (status == rcd::CaptureStatus::Timeout) {
            // Static screen: nothing changed, so the last detections are
            // still exactly right — keep them from going "stale".
            std::lock_guard<std::mutex> lock(shared.mutex);
            if (shared.updated_at_ms != 0)
                shared.updated_at_ms = GetTickCount64();
            continue;
        }
        if (status == rcd::CaptureStatus::Reinit) {
            if (!warned_size_change &&
                (capture.width() != overlay_w || capture.height() != overlay_h)) {
                std::cerr << "[rcd] display mode changed (" << overlay_w << "x"
                          << overlay_h << " -> " << capture.width() << "x"
                          << capture.height()
                          << "); boxes may misalign — restart rcd_live\n";
                warned_size_change = true;
            }
            // Covers lock screen / UAC too, which can last minutes.
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            continue;
        }
        if (status == rcd::CaptureStatus::Error) {
            if (++consecutive_errors >= 10) {
                std::cerr << "[rcd] capture failed: " << capture.last_error()
                          << "\n";
                flags.failed.store(true);
                flags.running.store(false);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        consecutive_errors = 0;

        std::vector<rcd::Detection> detections;
        try {
            detections = detector.detect(frame);
        } catch (const std::exception& e) {
            std::cerr << "[rcd] inference failed: " << e.what() << "\n";
            flags.failed.store(true);
            flags.running.store(false);
            return;
        }

        const ULONGLONG now = GetTickCount64();
        const double dt = std::max(1.0, static_cast<double>(now - last_frame_ms));
        last_frame_ms = now;
        const double inst_fps = 1000.0 / dt;
        fps_ema = fps_ema <= 0.0 ? inst_fps : fps_ema * 0.9 + inst_fps * 0.1;

        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.detections = std::move(detections);
            shared.inference_ms = detector.last_stats().total_ms();
            shared.fps = fps_ema;
            shared.seq++;
            shared.updated_at_ms = now;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    rcd::DetectorOptions det_opts;
    rcd::DxgiScreenCapture::Options cap_opts;

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
            if (arg == "--model") det_opts.model_path = next("--model");
            else if (arg == "--conf") det_opts.conf_threshold = std::stof(next("--conf"));
            else if (arg == "--iou") det_opts.iou_threshold = std::stof(next("--iou"));
            else if (arg == "--backend")
                det_opts.backend = rcd::backend_from_string(next("--backend"));
            else if (arg == "--device") det_opts.device_id = std::stoi(next("--device"));
            else if (arg == "--monitor") cap_opts.output_index = std::stoi(next("--monitor"));
            else if (arg == "--classes") det_opts.class_filter = split_csv(next("--classes"));
            else if (arg == "--max-det") det_opts.max_detections = std::stoi(next("--max-det"));
            else if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            } else {
                std::cerr << "error: unknown option " << arg << "\n";
                usage();
                return 2;
            }
        } catch (const std::exception& e) {
            std::cerr << "error: bad value for " << arg << ": " << e.what()
                      << "\n";
            return 2;
        }
    }
    if (det_opts.model_path.empty()) {
        usage();
        return 2;
    }

    enable_per_monitor_dpi();

    rcd::DxgiScreenCapture capture(cap_opts);
    if (!capture.initialize()) {
        std::cerr << "error: screen capture init failed: "
                  << capture.last_error() << "\n";
        return 1;
    }
    std::cerr << "[rcd] capturing monitor " << cap_opts.output_index << " ("
              << capture.width() << "x" << capture.height() << ")\n";

    std::unique_ptr<rcd::YoloDetector> detector;
    try {
        detector = std::make_unique<rcd::YoloDetector>(det_opts);
        std::cerr << "[rcd] provider: " << detector->active_provider()
                  << ", model input " << detector->input_width() << "x"
                  << detector->input_height() << "\n";
        if (detector->active_provider() == "TensorRT")
            std::cerr << "[rcd] first TensorRT run builds an engine; this can "
                         "take a few minutes (cached afterwards)\n";
        detector->warmup(capture.width(), capture.height());
    } catch (const std::exception& e) {
        std::cerr << "error: detector init failed: " << e.what() << "\n";
        return 1;
    }

    rcd::OverlayWindow overlay;
    std::string err;
    if (!overlay.create(capture.desktop_rect(), err)) {
        std::cerr << "error: overlay init failed: " << err << "\n";
        return 1;
    }

    if (!RegisterHotKey(nullptr, kHotkeyToggle, 0, VK_F8) |
        !RegisterHotKey(nullptr, kHotkeyPause, 0, VK_F9) |
        !RegisterHotKey(nullptr, kHotkeyQuit, 0, VK_F10)) {
        std::cerr << "[rcd] warning: some hotkeys are taken by another app; "
                     "use Ctrl+C in this console to quit\n";
    }
    std::cerr << "[rcd] running — F8 show/hide, F9 pause, F10 quit\n"
              << "[rcd] tip: run Roblox in windowed or borderless mode; "
                 "exclusive fullscreen hides overlays\n";

    SharedResults shared;
    AppFlags flags;
    std::thread worker(detection_loop, std::ref(capture), std::ref(*detector),
                       std::ref(shared), std::ref(flags));

    std::vector<rcd::Detection> to_draw;
    while (flags.running.load(std::memory_order_relaxed)) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) flags.running.store(false);
            if (msg.message == WM_HOTKEY) {
                switch (msg.wParam) {
                    case kHotkeyToggle:
                        overlay.set_visible(!overlay.visible());
                        break;
                    case kHotkeyPause:
                        flags.paused.store(!flags.paused.load());
                        break;
                    case kHotkeyQuit:
                        flags.running.store(false);
                        break;
                }
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!flags.running.load(std::memory_order_relaxed)) break;

        if (!overlay.visible()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        rcd::OverlayWindow::Hud hud;
        hud.provider = detector->active_provider();
        hud.paused = flags.paused.load(std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            const ULONGLONG age = GetTickCount64() - shared.updated_at_ms;
            // Drop stale boxes so a stalled detector doesn't leave ghosts.
            if (!hud.paused && (shared.updated_at_ms == 0 || age > 1200)) {
                to_draw.clear();
                hud.waiting = true;
            } else {
                to_draw = shared.detections;
            }
            hud.inference_ms = shared.inference_ms;
            hud.fps = shared.fps;
            hud.character_count = static_cast<int>(to_draw.size());
        }
        // render() presents with vsync, which paces this loop to the
        // monitor's refresh rate.
        if (!overlay.render(to_draw, hud)) {
            std::cerr << "[rcd] overlay device lost; recreating\n";
            overlay.destroy();
            if (!overlay.create(capture.desktop_rect(), err)) {
                std::cerr << "error: overlay recreate failed: " << err << "\n";
                flags.failed.store(true);
                flags.running.store(false);
            }
        }
    }

    flags.running.store(false);
    worker.join();
    UnregisterHotKey(nullptr, kHotkeyToggle);
    UnregisterHotKey(nullptr, kHotkeyPause);
    UnregisterHotKey(nullptr, kHotkeyQuit);
    overlay.destroy();
    return flags.failed.load() ? 1 : 0;
}

#else
#include <cstdio>
int main() {
    std::fprintf(stderr,
                 "rcd_live requires Windows (DXGI desktop duplication). "
                 "Use rcd_cli on this platform.\n");
    return 1;
}
#endif
