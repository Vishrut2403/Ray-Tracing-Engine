#pragma once

#include "scenes/cornell_scene.h"
#include "hittables/sphere.h"
#include "hittables/hittable_list.h"
#include "materials/material.h"

inline Scene build_furnace_scene()
{
	Scene scene;

	auto world  = std::make_shared<hittable_list>();
	auto lights = std::make_shared<hittable_list>();

	auto mat = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));

	world->add(std::make_shared<sphere>(point3(0, 0, 0), 1.0, mat));

	scene.world  = world;
	scene.lights = lights;

	return scene;
}