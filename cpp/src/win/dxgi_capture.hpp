#pragma once
#ifdef _WIN32

#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <string>

#include "common/types.hpp"

namespace rcd {

enum class CaptureStatus {
    Ok,        // a new frame was written to `out`
    Timeout,   // nothing changed on screen within the timeout
    Reinit,    // duplication was lost and recreated; call grab() again
    Error,     // unrecoverable failure; see last_error()
};

// Zero-copy-ish desktop capture via the DXGI Desktop Duplication API.
// The GPU hands us the composited desktop image; we copy it through a
// staging texture into a CPU BGRA frame for preprocessing.
class DxgiScreenCapture {
public:
    struct Options {
        int output_index = 0;  // monitor index (0 = primary-adapter order)
        int timeout_ms = 100;
    };

    explicit DxgiScreenCapture(const Options& options);
    ~DxgiScreenCapture();

    bool initialize();
    CaptureStatus grab(Frame& out);

    int width() const { return width_; }
    int height() const { return height_; }
    // Desktop coordinates of the captured output (virtual-screen space);
    // the overlay window is placed on this rect.
    RECT desktop_rect() const { return desktop_rect_; }
    const std::string& last_error() const { return last_error_; }

private:
    bool create_duplication();
    void release_duplication();

    Options opts_;
    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
    Microsoft::WRL::ComPtr<IDXGIOutput1> output1_;
    Microsoft::WRL::ComPtr<IDXGIOutputDuplication> duplication_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging_;
    int width_ = 0;
    int height_ = 0;
    RECT desktop_rect_{};
    std::string last_error_;
};

}  // namespace rcd

#endif  // _WIN32
