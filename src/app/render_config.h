#pragma once

#include <string>
#include "core/vec3.h"

struct RenderConfig {
    std::string output_path = "renders/output.ppm";
    std::string feature     = "cornell";

    int width     = 400;
    int height    = 400;
    int samples   = 64;
    int max_depth = 10;
    int tile_size = 32;

    color background = color(0, 0, 0);
};

inline void apply_furnace_preset(RenderConfig& cfg) {
    cfg.width      = 200;
    cfg.height     = 200;
    cfg.samples    = 256;
    cfg.max_depth  = 10;
    cfg.background = color(1, 1, 1);
}