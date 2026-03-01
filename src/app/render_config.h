#pragma once

#include <string>
#include "core/vec3.h"

struct RenderConfig {
    std::string output_path = "renders/output.ppm";
    std::string feature     = "cornell";

    int width  = 600;
    int height = 600;

    int samples   = 400;
    int max_depth = 40;
    int tile_size = 32;

    color background = color(0,0,0);
};