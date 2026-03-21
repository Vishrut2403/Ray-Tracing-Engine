#pragma once
#include <string>
#include "core/vec3.h"
#include "render/framebuffer.h"

class ImageWriter {
public:
    static void write_ppm(
        const std::string& path,
        const Framebuffer& fb,
        int samples_per_pixel
    );

    static void write_exr(
        const std::string& path,
        const Framebuffer& fb
    );
};