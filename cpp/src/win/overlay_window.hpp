#pragma once
#ifdef _WIN32

#include <d2d1_1.h>
#include <d3d11.h>
#include <dcomp.h>
#include <dwrite.h>
#include <dxgi1_3.h>
#include <wrl/client.h>

#include <string>
#include <vector>

#include "common/types.hpp"

namespace rcd {

// Click-through, always-on-top, per-pixel-alpha overlay rendered on the GPU
// (D3D11 flip-model swap chain composed by DirectComposition, drawn with
// Direct2D/DirectWrite). The window never takes focus or input, and is
// excluded from screen capture so the detector never sees its own boxes.
class OverlayWindow {
public:
    struct Hud {
        std::string provider;
        double inference_ms = 0.0;
        double fps = 0.0;
        int character_count = 0;
        bool paused = false;
        bool waiting = false;  // no fresh frames from the detector
    };

    OverlayWindow() = default;
    ~OverlayWindow();
    OverlayWindow(const OverlayWindow&) = delete;
    OverlayWindow& operator=(const OverlayWindow&) = delete;

    // `rect` is the monitor rect in virtual-screen coordinates; detection
    // boxes are interpreted relative to its top-left corner.
    bool create(const RECT& rect, std::string& error);
    void destroy();

    // Draws one overlay frame and presents with vsync. Must be called from
    // the thread that created the window. Returns false if the D3D device
    // was lost and the overlay needs create() again.
    bool render(const std::vector<Detection>& detections, const Hud& hud);

    void set_visible(bool visible);
    bool visible() const { return visible_; }
    HWND hwnd() const { return hwnd_; }

private:
    bool init_graphics(std::string& error);
    static LRESULT CALLBACK wnd_proc(HWND, UINT, WPARAM, LPARAM);

    HWND hwnd_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    bool visible_ = true;
    HMODULE dcomp_lib_ = nullptr;

    Microsoft::WRL::ComPtr<ID3D11Device> d3d_;
    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapchain_;
    Microsoft::WRL::ComPtr<IDCompositionDevice> dcomp_;
    Microsoft::WRL::ComPtr<IDCompositionTarget> dcomp_target_;
    Microsoft::WRL::ComPtr<IDCompositionVisual> dcomp_visual_;
    Microsoft::WRL::ComPtr<ID2D1DeviceContext> d2d_ctx_;
    Microsoft::WRL::ComPtr<ID2D1Bitmap1> d2d_target_;
    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> brush_;
    Microsoft::WRL::ComPtr<IDWriteFactory> dwrite_;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> label_format_;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> hud_format_;
};

}  // namespace rcd

#endif  // _WIN32
