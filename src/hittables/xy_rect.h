#pragma once

#include <memory>
#include <cmath>
#include "hittable.h"
#include "acceleration/aabb.h"
#include "core/rtweekend.h"

class xy_rect : public hittable {
public:
	xy_rect() {}

	xy_rect(
		real _x0, real _x1,
		real _y0, real _y1,
		real _k,
		std::shared_ptr<material> mat
	)
		: x0(_x0), x1(_x1),
		  y0(_y0), y1(_y1),
		  k(_k),
		  mp(mat) {}

	virtual bool hit(
		const ray& r,
		const interval& ray_t,
		hit_record& rec
	) const override {

		auto t = (k - r.origin().z())
				 / r.direction().z();

		if (!ray_t.surrounds(t))
			return false;

		auto x = r.origin().x()
			   + t * r.direction().x();

		auto y = r.origin().y()
			   + t * r.direction().y();

		if (x < x0 || x > x1 ||
			y < y0 || y > y1)
			return false;

		rec.u = (x - x0) / (x1 - x0);
		rec.v = (y - y0) / (y1 - y0);

		rec.t = t;
		rec.p = r.at(t);

		vec3 outward_normal(0, 0, 1);
		rec.set_face_normal(r, outward_normal);
		rec.mat_ptr = mp;

		return true;
	}

	virtual bool bounding_box(
		real time0,
		real time1,
		aabb& output_box
	) const override {

		output_box = aabb(
			point3(x0, y0, k - 0.0001),
			point3(x1, y1, k + 0.0001)
		);

		return true;
	}

	virtual real pdf_value(
		const point3& origin,
		const vec3& direction
	) const override {

		hit_record rec;

		if (!this->hit(
				ray(origin, direction),
				interval(0.001, infinity),
				rec))
			return 0.0;

		real area = (x1 - x0) * (y1 - y0);

		real distance_squared =
			rec.t * rec.t *
			direction.length_squared();

		vec3 unit_dir = unit_vector(direction);

		real cosine =
			fabs(dot(rec.normal, -unit_dir));

		if (cosine < 1e-8)
			return 0.0;

		return distance_squared /
			(cosine * area);
	}

	virtual vec3 random(
		const point3& origin
	) const override {

		auto random_point = point3(
			random_double(x0, x1),
			random_double(y0, y1),
			k
		);

		return random_point - origin;
	}

	virtual real area() const override { return (x1-x0)*(y1-y0); }
	virtual point3 sample_area(real u1, real u2, vec3& ng) const override {
		ng = vec3(0,0,1);
		return point3(x0 + u1*(x1-x0), y0 + u2*(y1-y0), k);
	}
	virtual bool contains_point(const point3& p) const override {
		return std::fabs(p.z()-k) < 1e-3
			&& p.x() >= x0-1e-3 && p.x() <= x1+1e-3
			&& p.y() >= y0-1e-3 && p.y() <= y1+1e-3;
	}

	virtual void tessellate(TriSoup& out) const override {
		tri_add_quad(out, point3(x0,y0,k), point3(x1,y0,k),
						  point3(x1,y1,k), point3(x0,y1,k),
					 vec3(0,0,1), mp);
	}

private:
	std::shared_ptr<material> mp;
	real x0, x1, y0, y1, k;
};