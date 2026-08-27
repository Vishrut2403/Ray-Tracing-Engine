#ifndef BVH_H
#define BVH_H

#include "core/rtweekend.h"
#include "hittables/hittable.h"
#include "hittables/hittable_list.h"
#include "aabb.h"
#include <memory>
#include <algorithm>

class bvh_node : public hittable {
public:
	bvh_node() {}

	bvh_node(
		std::vector<std::shared_ptr<hittable>>& objects,
		size_t start,
		size_t end,
		real time0,
		real time1
	) {
		// Longest axis of the span, not a random one. A random axis reseeded
		// per process made the tree differ between runs, and traversal order is
		// not neutral here: constant_medium draws its scattering distance
		// inside hit(), so a different order shifts every later sampler
		// dimension along that path.
		int axis = longest_axis(objects, start, end, time0, time1);
		auto comparator = (axis == 0) ? box_x_compare
						: (axis == 1) ? box_y_compare
									  : box_z_compare;

		size_t object_span = end - start;

		if (object_span == 1) {
			left = right = objects[start];
		}
		else if (object_span == 2) {
			if (comparator(objects[start], objects[start+1])) {
				left  = objects[start];
				right = objects[start+1];
			} else {
				left  = objects[start+1];
				right = objects[start];
			}
		}
		else {
			std::sort(objects.begin() + start,
					  objects.begin() + end,
					  comparator);

			auto mid = start + object_span / 2;

			left  = std::make_shared<bvh_node>(
				objects, start, mid, time0, time1);

			right = std::make_shared<bvh_node>(
				objects, mid, end, time0, time1);
		}

		aabb box_left, box_right;

		if (!left->bounding_box(time0, time1, box_left)
		 || !right->bounding_box(time0, time1, box_right))
			std::cerr << "No bounding box in BVH constructor.\n";

		box = surrounding_box(box_left, box_right);
	}

	virtual bool hit(
		const ray& r,
		const interval& ray_t,
		hit_record& rec
	) const override {

		if (!box.hit(r, ray_t))
			return false;

		bool hit_left =
			left->hit(r, ray_t, rec);

		bool hit_right =
			right->hit(
				r,
				interval(ray_t.min,
						 hit_left ? rec.t : ray_t.max),
				rec
			);

		return hit_left || hit_right;
	}

	virtual bool bounding_box(
		real time0,
		real time1,
		aabb& output_box
	) const override {
		output_box = box;
		return true;
	}

	virtual void tessellate(TriSoup& out) const override {
		// A leaf holding one object sets left and right to it, so walking both
		// unconditionally would emit that object's triangles twice.
		if (left) left->tessellate(out);
		if (right && right != left) right->tessellate(out);
	}

public:
	std::shared_ptr<hittable> left;
	std::shared_ptr<hittable> right;
	aabb box;

private:
	static int longest_axis(
		const std::vector<std::shared_ptr<hittable>>& objects,
		size_t start, size_t end, real time0, real time1
	) {
		aabb span;
		bool have = false;
		for (size_t i = start; i < end; ++i) {
			aabb b;
			if (!objects[i]->bounding_box(time0, time1, b)) continue;
			span = have ? surrounding_box(span, b) : b;
			have = true;
		}
		if (!have) return 0;

		real ex = span.axis_interval(0).max - span.axis_interval(0).min;
		real ey = span.axis_interval(1).max - span.axis_interval(1).min;
		real ez = span.axis_interval(2).max - span.axis_interval(2).min;
		if (ex >= ey && ex >= ez) return 0;
		return ey >= ez ? 1 : 2;
	}

	static bool box_compare(
		const std::shared_ptr<hittable> a,
		const std::shared_ptr<hittable> b,
		int axis
	) {
		aabb box_a;
		aabb box_b;

		if (!a->bounding_box(0, 0, box_a)
		 || !b->bounding_box(0, 0, box_b))
			std::cerr << "No bounding box in BVH compare.\n";

		return box_a.axis_interval(axis).min
			 < box_b.axis_interval(axis).min;
	}

	static bool box_x_compare(
		const std::shared_ptr<hittable> a,
		const std::shared_ptr<hittable> b) {
		return box_compare(a, b, 0);
	}

	static bool box_y_compare(
		const std::shared_ptr<hittable> a,
		const std::shared_ptr<hittable> b) {
		return box_compare(a, b, 1);
	}

	static bool box_z_compare(
		const std::shared_ptr<hittable> a,
		const std::shared_ptr<hittable> b) {
		return box_compare(a, b, 2);
	}
};

#endif