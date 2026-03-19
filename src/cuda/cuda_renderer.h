#pragma once

#include "render/framebuffer.h"
#include "core/camera.h"
#include "scenes/cornell_scene.h"
#include "core/vec3.h"
#include "viewer/preview_window.h"

void cuda_render(const Scene& scene,
                 Framebuffer& fb,
                 const camera& cam,
                 const color& background,
                 int spp,
                 int max_depth,
                 PreviewWindow* preview = nullptr);