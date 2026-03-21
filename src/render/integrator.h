#pragma once

#include <memory>
#include <vector>
#include "core/rtweekend.h"
#include "core/ray.h"
#include "core/camera.h"
#include "hittables/hittable.h"
#include "hittables/hittable_list.h"
#include "lights/env_light.h"
#include "bdpt/path_vertex.h"

class Framebuffer;

int trace_camera_path(
    const ray& r,
    const color& initial_beta,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<env_light>& env,
    int max_depth,
    std::vector<PathVertex>& path
);

int trace_light_path(
    const std::shared_ptr<hittable_list>& lights,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<env_light>& env,
    int max_depth,
    std::vector<PathVertex>& path
);

color connect(
    const std::vector<PathVertex>& camera_path, int t,
    const std::vector<PathVertex>& light_path,  int s,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<env_light>& env,
    const color& background
);

double mis_weight(
    const std::vector<PathVertex>& camera_path, int t,
    const std::vector<PathVertex>& light_path,  int s,
    const PathVertex& sampled
);

color bdpt_Li(
    const ray& camera_ray,
    const color& background,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<hittable_list>& lights,
    const std::shared_ptr<env_light>& env,
    int max_depth
);

color Li(
    const ray& r,
    const color& background,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<hittable_list>& lights,
    int max_depth,
    const std::shared_ptr<env_light>& env = nullptr
);