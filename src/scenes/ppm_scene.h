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

inline Scene build_ppm_scene() {
	Scene scene;
	hittable_list world;
	auto lights = std::make_shared<hittable_list>();

	auto red   = std::make_shared<lambertian>(color(0.65,0.05,0.05));
	auto white = std::make_shared<lambertian>(color(0.73,0.73,0.73));
	auto green = std::make_shared<lambertian>(color(0.12,0.45,0.15));
	auto light = std::make_shared<diffuse_light>(color(30,25,20));
	auto glass = std::make_shared<dielectric>(1.5);
	auto gold  = std::make_shared<ggx>(color(0.9,0.7,0.1), 0.6, 0.9);

	world.add(std::make_shared<yz_rect>(0,555,0,555,555,green));
	world.add(std::make_shared<yz_rect>(0,555,0,555,0,  red));
	world.add(std::make_shared<xz_rect>(0,555,0,555,0,  white));
	world.add(std::make_shared<xz_rect>(0,555,0,555,555,white));
	world.add(std::make_shared<xy_rect>(0,555,0,555,555,white));

	auto ceil_light = std::make_shared<xz_rect>(213,343,227,332,554,light);
	world.add(std::make_shared<flip_face>(ceil_light));
	lights->add(ceil_light);

	world.add(std::make_shared<sphere>(point3(278,180,278), 120, glass));
	world.add(std::make_shared<sphere>(point3(150,100,150),  80, gold));

	scene.world    = std::make_shared<bvh_node>(
		world.objects, 0, world.objects.size(), 0.0, 1.0);
	scene.lights   = lights;
	scene.use_bdpt = false;
	return scene;
}