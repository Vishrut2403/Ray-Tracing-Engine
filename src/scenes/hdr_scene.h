#pragma once

#include "scenes/cornell_scene.h"
#include "hittables/sphere.h"
#include "hittables/hittable_list.h"
#include "hittables/xz_rect.h"
#include "materials/material.h"
#include "textures/hdr_texture.h"
#include "lights/env_light.h"
#include "acceleration/bvh.h"

inline Scene build_hdr_scene(const std::string& hdr_path = "textures/quarry_01_1k.hdr") {
	Scene scene;
	hittable_list world;
	auto lights = std::make_shared<hittable_list>();

	auto hdr = std::make_shared<hdr_texture>(hdr_path.c_str());
	auto env  = std::make_shared<env_light>(hdr);
	lights->add(env);

	auto floor_mat = std::make_shared<lambertian>(color(0.4, 0.4, 0.4));
	world.add(std::make_shared<xz_rect>(-8, 8, -4, 4, -1.0, floor_mat));

	float roughness_steps[] = { 0.025f, 0.25f, 0.5f, 0.75f, 1.0f };

	for (int i = 0; i < 5; ++i) {
		float r = roughness_steps[i];
		float x = (i - 2) * 2.5f;

		world.add(std::make_shared<sphere>(
			point3(x, 1.1, 0), 0.9,
			std::make_shared<ggx>(color(1.0, 0.76, 0.33), r, 1.0)
		));

		world.add(std::make_shared<sphere>(
			point3(x, -0.55, 0), 0.9,
			std::make_shared<ggx>(color(0.05, 0.2, 0.8), r, 0.0)
		));
	}

	scene.world  = std::make_shared<bvh_node>(
		world.objects, 0, world.objects.size(), 0.0, 1.0);
	scene.lights = lights;
	scene.env    = env;
	return scene;
}