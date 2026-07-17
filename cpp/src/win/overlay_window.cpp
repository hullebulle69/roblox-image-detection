#ifdef _WIN32

#include "win/overlay_window.hpp"

#include <cstdio>
#include <cwchar>

#pragma comment(lib, "d3d11")
#pragma comment(lib, "dxgi")
#pragma comment(lib, "d2d1")
#pragma comment(lib, "dwrite")

using Microsoft::WRL::ComPtr;

namespace rcd {
namespace {

constexpr wchar_t kWindowClass[] = L"RcdOverlayWindow";

// WDA_EXCLUDEFROMCAPTURE needs a Win10 2004 SDK; define for older ones.
#ifndef WDA_EXCLUDEFROMCAPTURE
#define WDA_EXCLUDEFROMCAPTURE 0x00000011
#endif

D2D1_COLOR_F score_color(float score) {
    if (score >= 0.75f) return D2D1::ColorF(0.24f, 0.86f, 0.31f);
    if (score >= 0.50f) return D2D1::ColorF(0.98f, 0.82f, 0.16f);
    return D2D1::ColorF(1.0f, 0.51f, 0.12f);
}

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(),
                                      static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()),
                        w.data(), n);
    return w;
}

}  // namespace

OverlayWindow::~OverlayWindow() { destroy(); }

LRESULT CALLBACK OverlayWindow::wnd_proc(HWND hwnd, UINT msg, WPARAM wp,
                                         LPARAM lp) {
    // Note: no PostQuitMessage on WM_DESTROY — the overlay is destroyed and
    // recreated on device loss, and app lifetime is owned by the main loop.
    switch (msg) {
        case WM_NCHITTEST:
            return HTTRANSPARENT;  // belt-and-braces click-through
        default:
            return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

bool OverlayWindow::create(const RECT& rect, std::string& error) {
    width_ = rect.right - rect.left;
    height_ = rect.bottom - rect.top;
    if (width_ <= 0 || height_ <= 0) {
        error = "overlay rect is empty";
        return false;
    }

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = kWindowClass;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);  // idempotent; ERROR_CLASS_ALREADY_EXISTS is fine

    hwnd_ = CreateWindowExW(
        WS_EX_NOREDIRECTIONBITMAP | WS_EX_LAYERED | WS_EX_TRANSPARENT |
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        kWindowClass, L"Roblox Character Detector Overlay", WS_POPUP,
        rect.left, rect.top, width_, height_, nullptr, nullptr, wc.hInstance,
        nullptr);
    if (!hwnd_) {
        error = "CreateWindowExW failed";
        return false;
    }
    SetLayeredWindowAttributes(hwnd_, 0, 255, LWA_ALPHA);

    // Keep our own boxes out of the DXGI capture: without this the detector
    // would see (and be confused by) the previous frame's overlay.
    if (!SetWindowDisplayAffinity(hwnd_, WDA_EXCLUDEFROMCAPTURE)) {
        fprintf(stderr,
                "[rcd] warning: SetWindowDisplayAffinity failed (needs "
                "Windows 10 2004+); overlay will be visible to capture. "
                "Boxes are drawn without labels' feedback protection.\n");
    }

    if (!init_graphics(error)) {
        destroy();
        return false;
    }
    ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
    visible_ = true;
    return true;
}

bool OverlayWindow::init_graphics(std::string& error) {
    auto fail = [&](const char* what, HRESULT hr) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s failed (hr=0x%08lX)", what,
                      static_cast<unsigned long>(hr));
        error = buf;
        return false;
    };

    static const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1,
                                               D3D_FEATURE_LEVEL_11_0,
                                               D3D_FEATURE_LEVEL_10_1};
    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels, ARRAYSIZE(levels),
        D3D11_SDK_VERSION, &d3d_, nullptr, nullptr);
    if (FAILED(hr)) return fail("D3D11CreateDevice(overlay)", hr);

    ComPtr<IDXGIDevice> dxgi_device;
    hr = d3d_.As(&dxgi_device);
    if (FAILED(hr)) return fail("IDXGIDevice query", hr);
    ComPtr<IDXGIAdapter> adapter;
    hr = dxgi_device->GetAdapter(&adapter);
    if (FAILED(hr)) return fail("GetAdapter", hr);
    ComPtr<IDXGIFactory2> factory;
    hr = adapter->GetParent(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return fail("IDXGIFactory2 query", hr);

    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width = static_cast<UINT>(width_);
    sd.Height = static_cast<UINT>(height_);
    sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
    hr = factory->CreateSwapChainForComposition(d3d_.Get(), &sd, nullptr,
                                                &swapchain_);
    if (FAILED(hr)) return fail("CreateSwapChainForComposition", hr);

    // DirectComposition is loaded dynamically so the binary starts on
    // systems where dcomp.dll is unavailable and can report a clean error.
    dcomp_lib_ = LoadLibraryW(L"dcomp.dll");
    if (!dcomp_lib_) return fail("LoadLibrary(dcomp.dll)", E_FAIL);
    using PFN_DCompositionCreateDevice =
        HRESULT(WINAPI*)(IDXGIDevice*, REFIID, void**);
    auto create_dcomp = reinterpret_cast<PFN_DCompositionCreateDevice>(
        reinterpret_cast<void*>(
            GetProcAddress(dcomp_lib_, "DCompositionCreateDevice")));
    if (!create_dcomp) return fail("GetProcAddress(DCompositionCreateDevice)", E_FAIL);
    hr = create_dcomp(dxgi_device.Get(), IID_PPV_ARGS(&dcomp_));
    if (FAILED(hr)) return fail("DCompositionCreateDevice", hr);
    hr = dcomp_->CreateTargetForHwnd(hwnd_, TRUE, &dcomp_target_);
    if (FAILED(hr)) return fail("CreateTargetForHwnd", hr);
    hr = dcomp_->CreateVisual(&dcomp_visual_);
    if (FAILED(hr)) return fail("CreateVisual", hr);
    dcomp_visual_->SetContent(swapchain_.Get());
    dcomp_target_->SetRoot(dcomp_visual_.Get());
    hr = dcomp_->Commit();
    if (FAILED(hr)) return fail("IDCompositionDevice::Commit", hr);

    ComPtr<ID2D1Factory1> d2d_factory;
    D2D1_FACTORY_OPTIONS fo{};
    hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED,
                           __uuidof(ID2D1Factory1), &fo,
                           reinterpret_cast<void**>(d2d_factory.GetAddressOf()));
    if (FAILED(hr)) return fail("D2D1CreateFactory", hr);
    ComPtr<ID2D1Device> d2d_device;
    hr = d2d_factory->CreateDevice(dxgi_device.Get(), &d2d_device);
    if (FAILED(hr)) return fail("ID2D1Factory1::CreateDevice", hr);
    hr = d2d_device->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE,
                                         &d2d_ctx_);
    if (FAILED(hr)) return fail("CreateDeviceContext", hr);

    ComPtr<IDXGISurface> surface;
    hr = swapchain_->GetBuffer(0, IID_PPV_ARGS(&surface));
    if (FAILED(hr)) return fail("GetBuffer(0)", hr);
    const D2D1_BITMAP_PROPERTIES1 bp = D2D1::BitmapProperties1(
        D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM,
                          D2D1_ALPHA_MODE_PREMULTIPLIED),
        96.f, 96.f);
    hr = d2d_ctx_->CreateBitmapFromDxgiSurface(surface.Get(), &bp,
                                               &d2d_target_);
    if (FAILED(hr)) return fail("CreateBitmapFromDxgiSurface", hr);
    d2d_ctx_->SetTarget(d2d_target_.Get());
    // 96 DPI => one DIP == one physical pixel == one capture pixel.
    d2d_ctx_->SetDpi(96.f, 96.f);

    hr = d2d_ctx_->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White),
                                         &brush_);
    if (FAILED(hr)) return fail("CreateSolidColorBrush", hr);

    hr = DWriteCreateFactory(
        DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
        reinterpret_cast<IUnknown**>(dwrite_.GetAddressOf()));
    if (FAILED(hr)) return fail("DWriteCreateFactory", hr);
    hr = dwrite_->CreateTextFormat(L"Segoe UI", nullptr,
                                   DWRITE_FONT_WEIGHT_SEMI_BOLD,
                                   DWRITE_FONT_STYLE_NORMAL,
                                   DWRITE_FONT_STRETCH_NORMAL, 15.f, L"",
                                   &label_format_);
    if (FAILED(hr)) return fail("CreateTextFormat(label)", hr);
    hr = dwrite_->CreateTextFormat(L"Segoe UI", nullptr,
                                   DWRITE_FONT_WEIGHT_NORMAL,
                                   DWRITE_FONT_STYLE_NORMAL,
                                   DWRITE_FONT_STRETCH_NORMAL, 14.f, L"",
                                   &hud_format_);
    if (FAILED(hr)) return fail("CreateTextFormat(hud)", hr);
    return true;
}

bool OverlayWindow::render(const std::vector<Detection>& detections,
                           const Hud& hud) {
    if (!d2d_ctx_) return false;

    d2d_ctx_->BeginDraw();
    d2d_ctx_->Clear(D2D1::ColorF(0, 0, 0, 0));

    for (const auto& det : detections) {
        const D2D1_RECT_F box =
            D2D1::RectF(det.box.x1, det.box.y1, det.box.x2, det.box.y2);
        const D2D1_COLOR_F color = score_color(det.score);

        brush_->SetColor(color);
        d2d_ctx_->DrawRoundedRectangle(
            D2D1::RoundedRect(box, 4.f, 4.f), brush_.Get(), 2.5f);

        wchar_t text[128];
        std::swprintf(text, 128, L"%ls %.0f%%", widen(det.label).c_str(),
                      det.score * 100.f);
        ComPtr<IDWriteTextLayout> layout;
        if (FAILED(dwrite_->CreateTextLayout(
                text, static_cast<UINT32>(wcslen(text)), label_format_.Get(),
                static_cast<float>(width_), 40.f, &layout)))
            continue;
        DWRITE_TEXT_METRICS tm{};
        layout->GetMetrics(&tm);

        float ty = det.box.y1 - tm.height - 8.f;
        if (ty < 0.f) ty = det.box.y1 + 4.f;
        const D2D1_RECT_F tag = D2D1::RectF(
            det.box.x1, ty, det.box.x1 + tm.width + 12.f, ty + tm.height + 6.f);
        brush_->SetColor(D2D1::ColorF(0.06f, 0.06f, 0.08f, 0.82f));
        d2d_ctx_->FillRoundedRectangle(D2D1::RoundedRect(tag, 3.f, 3.f),
                                       brush_.Get());
        brush_->SetColor(color);
        d2d_ctx_->DrawTextLayout(D2D1::Point2F(tag.left + 6.f, tag.top + 3.f),
                                 layout.Get(), brush_.Get(),
                                 D2D1_DRAW_TEXT_OPTIONS_NONE);
    }

    // HUD strip, top-left of the monitor.
    {
        wchar_t text[256];
        if (hud.paused) {
            std::swprintf(text, 256,
                          L"Roblox Character Detector — PAUSED — F9 resume, "
                          L"F8 hide, F10 quit");
        } else if (hud.waiting) {
            std::swprintf(text, 256,
                          L"Roblox Character Detector — %ls — waiting for "
                          L"frames… — F8 hide, F9 pause, F10 quit",
                          widen(hud.provider).c_str());
        } else {
            std::swprintf(text, 256,
                          L"Roblox Character Detector — %ls — %.0f FPS — "
                          L"%.1f ms — %d character%ls — F8 hide, F9 pause, "
                          L"F10 quit",
                          widen(hud.provider).c_str(), hud.fps,
                          hud.inference_ms, hud.character_count,
                          hud.character_count == 1 ? L"" : L"s");
        }
        ComPtr<IDWriteTextLayout> layout;
        if (SUCCEEDED(dwrite_->CreateTextLayout(
                text, static_cast<UINT32>(wcslen(text)), hud_format_.Get(),
                static_cast<float>(width_), 40.f, &layout))) {
            DWRITE_TEXT_METRICS tm{};
            layout->GetMetrics(&tm);
            const D2D1_RECT_F bg =
                D2D1::RectF(12.f, 12.f, 12.f + tm.width + 16.f,
                            12.f + tm.height + 10.f);
            brush_->SetColor(D2D1::ColorF(0.06f, 0.06f, 0.08f, 0.72f));
            d2d_ctx_->FillRoundedRectangle(D2D1::RoundedRect(bg, 4.f, 4.f),
                                           brush_.Get());
            brush_->SetColor(hud.paused
                                 ? D2D1::ColorF(0.98f, 0.82f, 0.16f)
                                 : D2D1::ColorF(0.92f, 0.94f, 0.96f));
            d2d_ctx_->DrawTextLayout(D2D1::Point2F(bg.left + 8.f, bg.top + 5.f),
                                     layout.Get(), brush_.Get(),
                                     D2D1_DRAW_TEXT_OPTIONS_NONE);
        }
    }

    const HRESULT hr = d2d_ctx_->EndDraw();
    if (hr == static_cast<HRESULT>(D2DERR_RECREATE_TARGET)) return false;
    if (FAILED(swapchain_->Present(1, 0))) return false;
    return true;
}

void OverlayWindow::set_visible(bool visible) {
    if (!hwnd_ || visible == visible_) return;
    ShowWindow(hwnd_, visible ? SW_SHOWNOACTIVATE : SW_HIDE);
    visible_ = visible;
}

void OverlayWindow::destroy() {
    d2d_target_.Reset();
    d2d_ctx_.Reset();
    brush_.Reset();
    label_format_.Reset();
    hud_format_.Reset();
    dwrite_.Reset();
    dcomp_visual_.Reset();
    dcomp_target_.Reset();
    dcomp_.Reset();
    swapchain_.Reset();
    d3d_.Reset();
    if (hwnd_) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
    if (dcomp_lib_) {
        FreeLibrary(dcomp_lib_);
        dcomp_lib_ = nullptr;
    }
}

}  // namespace rcd

#endif  // _WIN32
