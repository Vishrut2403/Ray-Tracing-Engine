#pragma once
#include <vector>
#include <mutex>
#include "core/vec3.h"

class Framebuffer {
public:
    Framebuffer(int w, int h)
        : width(w), height(h), pixels(w * h, color(0,0,0)) {}

    void set(int x, int y, const color& c) {
        pixels[y * width + x] = c;
    }

    const color& get(int x, int y) const {
        return pixels[y * width + x];
    }

    const float* raw_data() const {
        return reinterpret_cast<const float*>(pixels.data());
    }

    int get_width() const { return width; }
    int get_height() const { return height; }

    mutable std::mutex mtx;   // ← ADD THIS BACK

private:
    int width, height;
    std::vector<color> pixels;
};