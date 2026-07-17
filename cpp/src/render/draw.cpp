#include "render/draw.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

extern "C" {
#include "font8x8_basic.h"  // public-domain 8x8 bitmap font
}

namespace rcd {
namespace {

struct Color {
    uint8_t b, g, r, a;
};

Color score_color(float score) {
    if (score >= 0.75f) return {80, 220, 60, 255};    // green
    if (score >= 0.50f) return {40, 210, 250, 255};   // yellow
    return {30, 130, 255, 255};                       // orange
}

void put_px(Frame& f, int x, int y, Color c) {
    if (x < 0 || y < 0 || x >= f.width || y >= f.height) return;
    uint8_t* p = f.bgra.data() + static_cast<size_t>(y) * f.stride +
                 static_cast<size_t>(x) * 4;
    p[0] = c.b;
    p[1] = c.g;
    p[2] = c.r;
    p[3] = c.a;
}

void fill_rect(Frame& f, int x1, int y1, int x2, int y2, Color c) {
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(f.width, x2);
    y2 = std::min(f.height, y2);
    for (int y = y1; y < y2; ++y)
        for (int x = x1; x < x2; ++x) put_px(f, x, y, c);
}

void stroke_rect(Frame& f, int x1, int y1, int x2, int y2, int t, Color c) {
    fill_rect(f, x1, y1, x2, y1 + t, c);
    fill_rect(f, x1, y2 - t, x2, y2, c);
    fill_rect(f, x1, y1, x1 + t, y2, c);
    fill_rect(f, x2 - t, y1, x2, y2, c);
}

void draw_text(Frame& f, int x, int y, const std::string& text, int scale,
               Color c) {
    int cx = x;
    for (char ch : text) {
        const unsigned char u = static_cast<unsigned char>(ch);
        if (u >= 128) {
            cx += 8 * scale;
            continue;
        }
        const unsigned char* glyph = font8x8_basic[u];
        for (int gy = 0; gy < 8; ++gy) {
            const unsigned char bits = glyph[gy];
            for (int gx = 0; gx < 8; ++gx) {
                if (!(bits & (1 << gx))) continue;
                for (int sy = 0; sy < scale; ++sy)
                    for (int sx = 0; sx < scale; ++sx)
                        put_px(f, cx + gx * scale + sx, y + gy * scale + sy, c);
            }
        }
        cx += 8 * scale;
    }
}

}  // namespace

void annotate(Frame& frame, const std::vector<Detection>& detections) {
    // Scale line thickness / text with image size so annotations stay legible
    // on both thumbnails and 4K screenshots.
    const int dim = std::max(frame.width, frame.height);
    const int thickness = std::clamp(dim / 400, 2, 8);
    const int text_scale = std::clamp(dim / 640, 1, 4);

    for (const auto& det : detections) {
        const Color c = score_color(det.score);
        const int x1 = static_cast<int>(std::lround(det.box.x1));
        const int y1 = static_cast<int>(std::lround(det.box.y1));
        const int x2 = static_cast<int>(std::lround(det.box.x2));
        const int y2 = static_cast<int>(std::lround(det.box.y2));
        stroke_rect(frame, x1, y1, x2, y2, thickness, c);

        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s %.0f%%", det.label.c_str(),
                      det.score * 100.f);
        const int tw = static_cast<int>(std::strlen(buf)) * 8 * text_scale;
        const int th = 8 * text_scale;
        int ty = y1 - th - 4;
        if (ty < 0) ty = y1 + thickness + 2;
        fill_rect(frame, x1, ty - 2, x1 + tw + 6, ty + th + 2,
                  {20, 20, 20, 220});
        draw_text(frame, x1 + 3, ty, buf, text_scale, {255, 255, 255, 255});
    }
}

}  // namespace rcd
