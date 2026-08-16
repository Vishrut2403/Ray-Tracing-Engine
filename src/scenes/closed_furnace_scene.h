#pragma once

#include "scenes/cornell_scene.h"
#include "hittables/hittable_list.h"
#include "hittables/xz_rect.h"
#include "hittables/yz_rect.h"
#include "hittables/xy_rect.h"
#include "materials/material.h"
#include "acceleration/bvh.h"

// A sealed enclosure whose walls both emit and reflect. Every surface sees the
// same radiance from every direction, so equilibrium is exact:
//     L = Le + rho * L   =>   L = Le / (1 - rho)
// With rho = 0.5 and Le = 0.5 the whole image must be exactly 1.0. Unlike the
// open furnace (a convex sphere, one bounce) this exercises the full transport
// chain: interreflection, Russian roulette, depth truncation and MIS.
class emissive_lambertian : public lambertian {
public:
	color Le;
	emissive_lambertian(const color& albedo, const color& e)
		: lambertian(albedo), Le(e) {}

	virtual color emitted(const ray&, const hit_record&,
						   double, double, const point3&) const override {
		return Le;
	}
};

inline Scene build_closed_furnace_scene() {
	Scene scene;
	hittable_list world;

	const double rho = 0.5, le = 0.5;   // => L = 1.0
	auto wall = std::make_shared<emissive_lambertian>(
		color(rho, rho, rho), color(le, le, le));

	world.add(std::make_shared<yz_rect>(0, 555, 0, 555,   0, wall));
	world.add(std::make_shared<yz_rect>(0, 555, 0, 555, 555, wall));
	world.add(std::make_shared<xz_rect>(0, 555, 0, 555,   0, wall));
	world.add(std::make_shared<xz_rect>(0, 555, 0, 555, 555, wall));
	world.add(std::make_shared<xy_rect>(0, 555, 0, 555,   0, wall));
	world.add(std::make_shared<xy_rect>(0, 555, 0, 555, 555, wall));

	scene.world  = std::make_shared<bvh_node>(
		world.objects, 0, world.objects.size(), 0.0, 1.0);
	// Left empty on purpose: with no explicit lights the estimator falls back to
	// pure BSDF sampling, so this measures transport without NEE or MIS.
	scene.lights = std::make_shared<hittable_list>();
	return scene;
}
