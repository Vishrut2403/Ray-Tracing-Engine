#pragma once

#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include <memory>

class PathIntegrator {
public:
    color Li(
        const ray& r,
        const color& background,
        const hittable& world,
        const std::shared_ptr<hittable>& lights,
        int depth
    ) const;
};