#pragma once

#include <memory>
#include "core/rtweekend.h"
#include "core/ray.h"
#include "hittables/hittable.h"
#include "hittables/hittable_list.h"

color Li(
    const ray& r,
    const color& background,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<hittable_list>& lights,
    int depth
);