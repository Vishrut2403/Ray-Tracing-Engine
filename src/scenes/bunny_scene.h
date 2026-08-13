#pragma once

#include "scenes/cornell_scene.h"
#include "hittables/sphere.h"
#include "hittables/hittable_list.h"
#include "hittables/xz_rect.h"
#include "hittables/flip_face.h"
#include "materials/material.h"
#include "geometry/mesh.h"
#include "acceleration/bvh.h"

inline Scene build_bunny_scene(
	const std::string& obj_path  = "models/bunny.obj",
	const std::string& hdr_path  = "" 
) {
	Scene scene;
	hittable_list world;
	auto lights = std::make_shared<hittable_list>();

	auto floor_mat  = std::make_shared<lambertian>(color(0.6, 0.6, 0.6));
	auto light_mat  = std::make_shared<diffuse_light>(color(12, 12, 12));
	auto bunny_mat  = std::make_shared<ggx>(color(0.8, 0.6, 0.2), 0.2, 0.9);

	auto ceiling_light = std::make_shared<xz_rect>(-2, 2, -1, 1, 4, light_mat);
	world.add(std::make_shared<flip_face>(ceiling_light));
	lights->add(ceiling_light);

	world.add(std::make_shared<xz_rect>(-5, 5, -5, 5, 0, floor_mat));

	auto bunny = load_obj(obj_path, bunny_mat, 8.0, vec3(0, 0.15, 0));
	if (bunny) {
		world.add(bunny);
	} else {
		std::cerr << "[bunny_scene] OBJ load failed, using sphere fallback\n";
		world.add(std::make_shared<sphere>(
			point3(0, 1, 0), 0.8, bunny_mat));
	}

	scene.world  = std::make_shared<bvh_node>(
		world.objects, 0, world.objects.size(), 0.0, 1.0);
	scene.lights = lights;
	return scene;
}