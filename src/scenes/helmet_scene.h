#pragma once

#include "scenes/cornell_scene.h"
#include "hittables/hittable_list.h"
#include "hittables/xz_rect.h"
#include "hittables/flip_face.h"
#include "materials/material.h"
#include "textures/hdr_texture.h"
#include "lights/env_light.h"
#include "geometry/gltf_mesh.h"
#include "acceleration/bvh.h"

inline Scene build_helmet_scene(
    const std::string& gltf_path = "models/damaged_helmet/DamagedHelmet.gltf",
    const std::string& hdr_path  = "textures/quarry_01_1k.hdr"
) {
    Scene scene;
    hittable_list world;
    auto lights = std::make_shared<hittable_list>();

    auto hdr = std::make_shared<hdr_texture>(hdr_path.c_str());
    auto env  = std::make_shared<env_light>(hdr);
    lights->add(env);

    auto floor_mat = std::make_shared<lambertian>(color(0.05, 0.05, 0.05));
    world.add(std::make_shared<xz_rect>(-5, 5, -5, 5, -1.2, floor_mat));

    auto helmet = load_gltf(gltf_path, 1.0, vec3(0, 0.2, 0));
    if (helmet) {
        world.add(helmet);
    } else {
        std::cerr << "[helmet_scene] glTF load failed\n";
        auto mat = std::make_shared<ggx>(color(0.8, 0.3, 0.1), 0.3, 0.9);
        world.add(std::make_shared<sphere>(point3(0, 0, 0), 1.0, mat));
    }

    scene.world  = std::make_shared<bvh_node>(
        world.objects, 0, world.objects.size(), 0.0, 1.0);
    scene.lights = lights;
    scene.env    = env;
    return scene;
}