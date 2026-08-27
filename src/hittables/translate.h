#pragma once

#include <memory>
#include "hittable.h"

class translate : public hittable {
public:
	translate(std::shared_ptr<hittable> p,
			  const vec3& displacement)
		: ptr(p), offset(displacement) {}

	virtual bool hit(
		const ray& r,
		const interval& ray_t,
		hit_record& rec
	) const override {

		ray moved_r(
			r.origin() - offset,
			r.direction(),
			r.time()
		);

		if (!ptr->hit(moved_r, ray_t, rec))
			return false;

		rec.p += offset;

		rec.set_face_normal(moved_r, rec.normal);

		return true;
	}

	virtual bool bounding_box(
		real time0,
		real time1,
		aabb& output_box
	) const override {

		if (!ptr->bounding_box(time0, time1, output_box))
			return false;

		output_box = aabb(
			interval(output_box.x.min + offset.x(),
					output_box.x.max + offset.x()),
			interval(output_box.y.min + offset.y(),
					output_box.y.max + offset.y()),
			interval(output_box.z.min + offset.z(),
					output_box.z.max + offset.z())
		);

		return true;
	}
	virtual void tessellate(TriSoup& out) const override {
		size_t begin = out.size();
		ptr->tessellate(out);
		for (size_t i = begin; i < out.size(); ++i) {
			out[i].v0 += offset; out[i].v1 += offset; out[i].v2 += offset;
		}
	}

private:
	std::shared_ptr<hittable> ptr;
	vec3 offset;
};