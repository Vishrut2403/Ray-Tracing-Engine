#pragma once
#include <memory>
#include "hittables/hittable.h"
#include "hittables/hittable_list.h"

class env_light;

struct Scene {
    std::shared_ptr<hittable>      world;
    std::shared_ptr<hittable_list> lights;
    std::shared_ptr<env_light>     env = nullptr;
};

Scene build_cornell_volume_scene();