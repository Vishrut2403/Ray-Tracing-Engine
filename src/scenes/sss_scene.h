#pragma once

#include "scenes/cornell_scene.h"
#include "hittables/sphere.h"
#include "hittables/hittable_list.h"
#include "hittables/xz_rect.h"
#include "hittables/yz_rect.h"
#include "hittables/xy_rect.h"
#include "hittables/flip_face.h"
#include "materials/material.h"
#include "acceleration/bvh.h"

// Subsurface scattering demo — three spheres in a Cornell box.
// Mirrors SceneUploader::build_sss so both backends render the same scene.
inline Scene build_sss_scene() {
	Scene scene;
	hittable_list world;
	auto lights = std::make_shared<hittable_list>();

	auto red   = std::make_shared<lambertian>(color(0.65, 0.05, 0.05));
	auto white = std::make_shared<lambertian>(color(0.73, 0.73, 0.73));
	auto green = std::make_shared<lambertian>(color(0.12, 0.45, 0.15));
	auto light = std::make_shared<diffuse_light>(color(60, 56, 50));

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

	// Skin — warm salmon, short mean free path
	world.add(std::make_shared<sphere>(
		point3(160, 100, 180), 60,
		std::make_shared<subsurface>(color(0.85, 0.40, 0.25), 0.08, 1.4)));

	// Marble — near-white, long mean free path, very translucent
	world.add(std::make_shared<sphere>(
		point3(390, 120, 350), 70,
		std::make_shared<subsurface>(color(0.92, 0.89, 0.82), 0.20, 1.3)));

	// Jade — deep green, moderate mean free path
	world.add(std::make_shared<sphere>(
		point3(390, 310, 120), 60,
		std::make_shared<subsurface>(color(0.18, 0.55, 0.34), 0.10, 1.5)));

	scene.world  = std::make_shared<bvh_node>(
		world.objects, 0, world.objects.size(), 0.0, 1.0);
	scene.lights = lights;
	return scene;
}
