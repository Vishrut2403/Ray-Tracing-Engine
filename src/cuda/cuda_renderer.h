#pragma once

#include "render/framebuffer.h"
#include "core/camera.h"
#include "scenes/cornell_scene.h"
#include "core/vec3.h"

void cuda_render(const Scene& scene,
                 Framebuffer& fb,
                 const camera& cam,
                 const color& background,
                 int spp,
                 int max_depth);
