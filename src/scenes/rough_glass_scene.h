#pragma once

#include "scenes/cornell_scene.h"
#include "hittables/sphere.h"
#include "hittables/hittable_list.h"
#include "hittables/xz_rect.h"
#include "hittables/flip_face.h"
#include "materials/material.h"
#include "acceleration/bvh.h"

inline Scene build_rough_glass_scene() {
    Scene scene;
    hittable_list world;
    auto lights = std::make_shared<hittable_list>();

    auto floor_mat = std::make_shared<lambertian>(color(0.1, 0.1, 0.12));
    auto light_mat = std::make_shared<diffuse_light>(color(8, 7, 6));

    auto ceiling_light = std::make_shared<xz_rect>(-8, 8, -3, 3, 6, light_mat);
    world.add(std::make_shared<flip_face>(ceiling_light));
    lights->add(ceiling_light);

    world.add(std::make_shared<xz_rect>(-10, 10, -6, 6, -1.1, floor_mat));

    auto backdrop = std::make_shared<lambertian>(color(0.08, 0.25, 0.6));
    world.add(std::make_shared<sphere>(point3(0, 3, -5), 4.0, backdrop));

    double roughness_steps[] = { 0.001, 0.05, 0.1, 0.2, 0.3 };

    for (int i = 0; i < 5; ++i) {
        double r = roughness_steps[i];
        double x = (i - 2) * 2.8;
        auto mat = std::make_shared<rough_dielectric>(color(1.0, 1.0, 1.0), r, 1.5);
        world.add(std::make_shared<sphere>(point3(x, 0.0, 0), 1.0, mat));
    }

    scene.world  = std::make_shared<bvh_node>(
        world.objects, 0, world.objects.size(), 0.0, 1.0);
    scene.lights = lights;
    return scene;
}