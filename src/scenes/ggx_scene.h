#pragma once
#include "scenes/cornell_scene.h"
#include "hittables/sphere.h"
#include "hittables/hittable_list.h"
#include "hittables/xz_rect.h"
#include "hittables/flip_face.h"
#include "materials/material.h"
#include "acceleration/bvh.h"

inline Scene build_ggx_scene() {
    Scene scene;
    hittable_list world;
    auto lights = std::make_shared<hittable_list>();

    auto light_mat = std::make_shared<diffuse_light>(color(12, 12, 12));
    auto floor_mat = std::make_shared<lambertian>(color(0.4, 0.4, 0.4));

    // Wide ceiling light
    auto ceiling_light = std::make_shared<xz_rect>(-6, 6, -2, 2, 5, light_mat);
    world.add(std::make_shared<flip_face>(ceiling_light));
    lights->add(ceiling_light);

    // Floor
    world.add(std::make_shared<xz_rect>(-8, 8, -4, 4, -1.0, floor_mat));

    // Roughness values: 0.025 -> 1.0
    double roughness_steps[] = { 0.025, 0.25, 0.5, 0.75, 1.0 };

    for (int i = 0; i < 5; ++i) {
        double r = roughness_steps[i];
        double x = (i - 2) * 2.5;

        // Top row — metallic gold
        world.add(std::make_shared<sphere>(
            point3(x, 1.1, 0), 0.9,
            std::make_shared<ggx>(color(1.0, 0.76, 0.33), r, 1.0)
        ));

        // Bottom row — dielectric blue
        world.add(std::make_shared<sphere>(
            point3(x, -0.55, 0), 0.9,
            std::make_shared<ggx>(color(0.05, 0.2, 0.8), r, 0.0)
        ));
    }

    scene.world  = std::make_shared<bvh_node>(
        world.objects, 0, world.objects.size(), 0.0, 1.0);
    scene.lights = lights;
    return scene;
}