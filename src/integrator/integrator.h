#pragma once

#include "core/ray.h"
#include "core/vec3.h"
#include "hittables/hittable.h"

class Integrator {
public:
    virtual ~Integrator() = default;

    virtual vec3 Li(
        const ray& r,
        const hittable& world,
        int max_depth
    ) const = 0;
};