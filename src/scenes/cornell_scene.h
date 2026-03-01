#pragma once
#include <memory>
#include "hittables/hittable.h"
#include "hittables/hittable_list.h"

struct Scene {
    std::shared_ptr<hittable> world;
    std::shared_ptr<hittable_list> lights;
};

Scene build_cornell_volume_scene();