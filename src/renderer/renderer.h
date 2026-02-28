#pragma once

#include <memory>
#include <ostream>

#include "hittable.h"
#include "camera.h"
#include "vec3.h"
#include "integrator/path_integrator.h"

class Renderer {
public:
    void render(
        std::ostream& out,
        const hittable& world,
        const std::shared_ptr<hittable>& lights,
        const camera& cam,
        const color& background,
        const PathIntegrator& integrator,
        int image_width,
        int image_height,
        int samples_per_pixel,
        int max_depth
    ) const;
};