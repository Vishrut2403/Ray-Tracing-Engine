#pragma once

#include <memory>
#include "hittable.h"

class flip_face : public hittable {
public:
	flip_face(std::shared_ptr<hittable> p)
		: ptr(p) {}

	virtual bool hit(
		const ray& r,
		const interval& ray_t,
		hit_record& rec
	) const override {

		if (!ptr->hit(r, ray_t, rec))
			return false;

		rec.front_face = !rec.front_face;
		return true;
	}

	virtual bool bounding_box(
		real time0,
		real time1,
		aabb& output_box
	) const override {

		return ptr->bounding_box(time0, time1, output_box);
	}

	virtual real pdf_value(const point3& o, const vec3& d) const override {
		return ptr->pdf_value(o, d);
	}
	virtual vec3 random(const point3& o) const override { return ptr->random(o); }

	// Normal stays as the wrapped shape reports it; callers pick the emitting side.
	virtual real area() const override { return ptr->area(); }
	virtual point3 sample_area(real u1, real u2, vec3& ng) const override {
		return ptr->sample_area(u1, u2, ng);
	}
	virtual bool contains_point(const point3& p) const override {
		return ptr->contains_point(p);
	}

public:
	std::shared_ptr<hittable> ptr;
};