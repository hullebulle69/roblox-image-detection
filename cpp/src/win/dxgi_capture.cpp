#ifdef _WIN32

#include "win/dxgi_capture.hpp"

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace rcd {
namespace {

std::string hr_message(const char* what, HRESULT hr) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s failed (hr=0x%08lX)", what,
                  static_cast<unsigned long>(hr));
    return buf;
}

uint64_t now_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace

DxgiScreenCapture::DxgiScreenCapture(const Options& options)
    : opts_(options) {}

DxgiScreenCapture::~DxgiScreenCapture() { release_duplication(); }

bool DxgiScreenCapture::initialize() {
    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        last_error_ = hr_message("CreateDXGIFactory1", hr);
        return false;
    }

    // Find the adapter+output pair for the requested monitor index,
    // counting outputs across adapters in enumeration order.
    ComPtr<IDXGIAdapter1> chosen_adapter;
    ComPtr<IDXGIOutput> chosen_output;
    int seen = 0;
    for (UINT a = 0;; ++a) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(a, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        for (UINT o = 0;; ++o) {
            ComPtr<IDXGIOutput> output;
            if (adapter->EnumOutputs(o, &output) == DXGI_ERROR_NOT_FOUND)
                break;
            if (seen++ == opts_.output_index) {
                chosen_adapter = adapter;
                chosen_output = output;
            }
        }
    }
    if (!chosen_output) {
        last_error_ = "monitor index " + std::to_string(opts_.output_index) +
                      " not found (" + std::to_string(seen) + " outputs)";
        return false;
    }

    DXGI_OUTPUT_DESC out_desc{};
    chosen_output->GetDesc(&out_desc);
    desktop_rect_ = out_desc.DesktopCoordinates;
    width_ = desktop_rect_.right - desktop_rect_.left;
    height_ = desktop_rect_.bottom - desktop_rect_.top;
    if (out_desc.Rotation != DXGI_MODE_ROTATION_IDENTITY &&
        out_desc.Rotation != DXGI_MODE_ROTATION_UNSPECIFIED) {
        last_error_ =
            "rotated displays are not supported yet; set the monitor to "
            "landscape orientation";
        return false;
    }

    // The device must be created on the adapter that owns the output.
    static const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1,
                                               D3D_FEATURE_LEVEL_11_0,
                                               D3D_FEATURE_LEVEL_10_1};
    hr = D3D11CreateDevice(chosen_adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN,
                           nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels,
                           ARRAYSIZE(levels), D3D11_SDK_VERSION, &device_,
                           nullptr, &context_);
    if (FAILED(hr)) {
        last_error_ = hr_message("D3D11CreateDevice", hr);
        return false;
    }

    hr = chosen_output.As(&output1_);
    if (FAILED(hr)) {
        last_error_ = hr_message("IDXGIOutput1 query", hr);
        return false;
    }
    return create_duplication();
}

bool DxgiScreenCapture::create_duplication() {
    release_duplication();
    HRESULT hr = S_OK;
    // DuplicateOutput transiently fails during mode switches / UAC prompts.
    for (int attempt = 0; attempt < 10; ++attempt) {
        hr = output1_->DuplicateOutput(device_.Get(), &duplication_);
        if (SUCCEEDED(hr)) break;
        if (hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE) {
            last_error_ =
                "desktop duplication not available (too many apps are "
                "already duplicating this output)";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (FAILED(hr)) {
        last_error_ = hr_message("DuplicateOutput", hr);
        return false;
    }

    DXGI_OUTDUPL_DESC dup_desc{};
    duplication_->GetDesc(&dup_desc);
    width_ = static_cast<int>(dup_desc.ModeDesc.Width);
    height_ = static_cast<int>(dup_desc.ModeDesc.Height);

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = dup_desc.ModeDesc.Width;
    desc.Height = dup_desc.ModeDesc.Height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    HRESULT shr = device_->CreateTexture2D(&desc, nullptr, &staging_);
    if (FAILED(shr)) {
        last_error_ = hr_message("CreateTexture2D(staging)", shr);
        return false;
    }
    return true;
}

void DxgiScreenCapture::release_duplication() {
    if (duplication_) {
        duplication_->ReleaseFrame();  // harmless if no frame held
        duplication_.Reset();
    }
    staging_.Reset();
}

CaptureStatus DxgiScreenCapture::grab(Frame& out) {
    if (!duplication_) {
        if (!create_duplication()) return CaptureStatus::Error;
        return CaptureStatus::Reinit;
    }

    ComPtr<IDXGIResource> resource;
    DXGI_OUTDUPL_FRAME_INFO info{};
    HRESULT hr = duplication_->AcquireNextFrame(
        static_cast<UINT>(opts_.timeout_ms), &info, &resource);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) return CaptureStatus::Timeout;
    if (hr == DXGI_ERROR_ACCESS_LOST) {
        // Mode change, fullscreen transition, or secure desktop; rebuild.
        if (!create_duplication()) return CaptureStatus::Error;
        return CaptureStatus::Reinit;
    }
    if (FAILED(hr)) {
        last_error_ = hr_message("AcquireNextFrame", hr);
        return CaptureStatus::Error;
    }

    ComPtr<ID3D11Texture2D> texture;
    hr = resource.As(&texture);
    if (FAILED(hr)) {
        duplication_->ReleaseFrame();
        last_error_ = hr_message("frame texture query", hr);
        return CaptureStatus::Error;
    }
    context_->CopyResource(staging_.Get(), texture.Get());
    duplication_->ReleaseFrame();

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = context_->Map(staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        last_error_ = hr_message("Map(staging)", hr);
        return CaptureStatus::Error;
    }
    out.resize(width_, height_);
    const uint8_t* src = static_cast<const uint8_t*>(mapped.pData);
    const size_t row_bytes = static_cast<size_t>(width_) * 4;
    for (int y = 0; y < height_; ++y) {
        std::memcpy(out.bgra.data() + static_cast<size_t>(y) * out.stride,
                    src + static_cast<size_t>(y) * mapped.RowPitch, row_bytes);
    }
    context_->Unmap(staging_.Get(), 0);
    out.timestamp_us = now_us();
    return CaptureStatus::Ok;
}

}  // namespace rcd

#endif  // _WIN32
