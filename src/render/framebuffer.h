#pragma once
#include <vector>
#include <cstring>
#include <mutex>
#include "core/vec3.h"

class Framebuffer {
public:
    Framebuffer(int w, int h)
        : width(w), height(h), pixels(w * h * 3, 0.0f) {}

    void set(int x, int y, const color& c) {
        int base = (y * width + x) * 3;
        pixels[base + 0] = static_cast<float>(c.x());
        pixels[base + 1] = static_cast<float>(c.y());
        pixels[base + 2] = static_cast<float>(c.z());
    }

    color get(int x, int y) const {
        int base = (y * width + x) * 3;
        return color(pixels[base], pixels[base+1], pixels[base+2]);
    }

    void set_bulk(const float* src, float inv_scale) {
        int n = width * height * 3;
        if (inv_scale == 1.0f) {
            std::memcpy(pixels.data(), src, n * sizeof(float));
        } else {
            for (int i = 0; i < n; ++i)
                pixels[i] = src[i] * inv_scale;
        }
    }

    const float* raw_data() const { return pixels.data(); }

    int get_width()  const { return width;  }
    int get_height() const { return height; }

    mutable std::mutex mtx;

private:
    int width, height;
    std::vector<float> pixels;
};