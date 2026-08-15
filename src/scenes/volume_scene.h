#pragma once

#include "scenes/cornell_scene.h"
#include "hittables/hittable_list.h"
#include "hittables/xz_rect.h"
#include "hittables/yz_rect.h"
#include "hittables/xy_rect.h"
#include "hittables/flip_face.h"
#include "hittables/box.h"
#include "hittables/translate.h"
#include "hittables/rotate_y.h"
#include "hittables/constant_medium.h"
#include "materials/material.h"
#include "acceleration/bvh.h"

// Cornell box with a participating medium. Both backends model the same one:
// sigma_t = 0.003, scattering albedo 0.9, Henyey-Greenstein g = 0.2, bounded by
// the 0..555 box. Keep the two definitions in step or they stop matching.
inline Scene build_volume_scene() {
	Scene scene;
	hittable_list world;
	auto lights = std::make_shared<hittable_list>();

	auto red   = std::make_shared<lambertian>(color(0.65, 0.05, 0.05));
	auto white = std::make_shared<lambertian>(color(0.73, 0.73, 0.73));
	auto green = std::make_shared<lambertian>(color(0.12, 0.45, 0.15));
	auto light = std::make_shared<diffuse_light>(color(30, 28, 25));

	world.add(std::make_shared<yz_rect>(0, 555, 0, 555, 555, green));
	world.add(std::make_shared<flip_face>(
		std::make_shared<yz_rect>(0, 555, 0, 555, 0, red)));
	world.add(std::make_shared<xz_rect>(0, 555, 0, 555, 0, white));
	world.add(std::make_shared<flip_face>(
		std::make_shared<xz_rect>(0, 555, 0, 555, 555, white)));
	world.add(std::make_shared<flip_face>(
		std::make_shared<xy_rect>(0, 555, 0, 555, 555, white)));

	auto ceiling_light =
		std::make_shared<xz_rect>(213, 343, 227, 332, 554, light);
	world.add(std::make_shared<flip_face>(ceiling_light));
	lights->add(ceiling_light);

	std::shared_ptr<hittable> box1 =
		std::make_shared<box>(point3(0,0,0), point3(165,330,165), white);
	box1 = std::make_shared<rotate_y>(box1, 15);
	box1 = std::make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	std::shared_ptr<hittable> box2 =
		std::make_shared<box>(point3(0,0,0), point3(165,165,165), white);
	box2 = std::make_shared<rotate_y>(box2, -18);
	box2 = std::make_shared<translate>(box2, vec3(130,0,65));
	world.add(box2);

	// sigma_t = 0.003 with a scattering albedo of 0.9, matching the GPU medium.
	auto fog_bounds =
		std::make_shared<box>(point3(0,0,0), point3(555,555,555), white);
	world.add(std::make_shared<constant_medium>(
		fog_bounds, 0.003, color(0.9, 0.9, 0.9), 0.2));

	scene.world  = std::make_shared<bvh_node>(
		world.objects, 0, world.objects.size(), 0.0, 1.0);
	scene.lights = lights;
	return scene;
}
