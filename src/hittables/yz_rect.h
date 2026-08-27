#pragma once

#include <memory>
#include <cmath>
#include "hittable.h"
#include "acceleration/aabb.h"
#include "core/rtweekend.h"

class yz_rect : public hittable {
public:
	yz_rect() {}

	yz_rect(
		real _y0, real _y1,
		real _z0, real _z1,
		real _k,
		std::shared_ptr<material> mat
	)
		: y0(_y0), y1(_y1),
		  z0(_z0), z1(_z1),
		  k(_k),
		  mp(mat) {}

	virtual bool hit(
		const ray& r,
		const interval& ray_t,
		hit_record& rec
	) const override {

		auto t = (k - r.origin().x())
				 / r.direction().x();

		if (!ray_t.surrounds(t))
			return false;

		auto y = r.origin().y()
			   + t * r.direction().y();

		auto z = r.origin().z()
			   + t * r.direction().z();

		if (y < y0 || y > y1 ||
			z < z0 || z > z1)
			return false;

		rec.u = (y - y0) / (y1 - y0);
		rec.v = (z - z0) / (z1 - z0);

		rec.t = t;
		rec.p = r.at(t);

		vec3 outward_normal(1, 0, 0);
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
			point3(k - 0.0001, y0, z0),
			point3(k + 0.0001, y1, z1)
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

		real area = (y1 - y0) * (z1 - z0);

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
			k,
			random_double(y0, y1),
			random_double(z0, z1)
		);

		return random_point - origin;
	}

	virtual real area() const override { return (y1-y0)*(z1-z0); }
	virtual point3 sample_area(real u1, real u2, vec3& ng) const override {
		ng = vec3(1,0,0);
		return point3(k, y0 + u1*(y1-y0), z0 + u2*(z1-z0));
	}
	virtual bool contains_point(const point3& p) const override {
		return std::fabs(p.x()-k) < 1e-3
			&& p.y() >= y0-1e-3 && p.y() <= y1+1e-3
			&& p.z() >= z0-1e-3 && p.z() <= z1+1e-3;
	}

	virtual void tessellate(TriSoup& out) const override {
		tri_add_quad(out, point3(k,y0,z0), point3(k,y1,z0),
						  point3(k,y1,z1), point3(k,y0,z1),
					 vec3(1,0,0), mp);
	}

private:
	std::shared_ptr<material> mp;
	real y0, y1;
	real z0, z1;
	real k;
};