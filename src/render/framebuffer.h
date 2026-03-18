#pragma once
#include <vector>
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
        return color(
            static_cast<double>(pixels[base + 0]),
            static_cast<double>(pixels[base + 1]),
            static_cast<double>(pixels[base + 2])
        );
    }

    const float* raw_data() const {
        return pixels.data();
    }

    int get_width()  const { return width;  }
    int get_height() const { return height; }

    mutable std::mutex mtx;

private:
    int width, height;
    std::vector<float> pixels; 
};