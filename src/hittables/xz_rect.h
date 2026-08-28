#pragma once

#include <memory>
#include <cmath>
#include "hittable.h"
#include "core/rtweekend.h"

class xz_rect : public hittable {
public:
	xz_rect() {}

	xz_rect(
		real _x0, real _x1,
		real _z0, real _z1,
		real _k,
		std::shared_ptr<material> mat
	)
		: x0(_x0), x1(_x1),
		  z0(_z0), z1(_z1),
		  k(_k),
		  mp(mat) {}

	virtual bool hit(
		const ray& r,
		const interval& ray_t,
		hit_record& rec
	) const override {

		auto t = (k - r.origin().y())
				 / r.direction().y();

		if (!ray_t.surrounds(t))
			return false;

		auto x = r.origin().x()
			   + t * r.direction().x();

		auto z = r.origin().z()
			   + t * r.direction().z();

		if (x < x0 || x > x1 ||
			z < z0 || z > z1)
			return false;

		rec.u = (x - x0) / (x1 - x0);
		rec.v = (z - z0) / (z1 - z0);

		rec.t = t;
		rec.p = r.at(t);

		vec3 outward_normal(0, 1, 0);
		rec.set_face_normal(r, outward_normal);
		rec.mat_ptr = mp.get();

		return true;
	}

	virtual bool bounding_box(
		real time0,
		real time1,
		aabb& output_box
	) const override {

		output_box = aabb(
			point3(x0, k - 0.0001, z0),
			point3(x1, k + 0.0001, z1)
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

		real area = (x1 - x0) * (z1 - z0);

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
			k,
			random_double(z0, z1)
		);

		return random_point - origin;
	}

	virtual real area() const override { return (x1-x0)*(z1-z0); }
	virtual point3 sample_area(real u1, real u2, vec3& ng) const override {
		ng = vec3(0,1,0);
		return point3(x0 + u1*(x1-x0), k, z0 + u2*(z1-z0));
	}
	virtual bool contains_point(const point3& p) const override {
		return std::fabs(p.y()-k) < 1e-3
			&& p.x() >= x0-1e-3 && p.x() <= x1+1e-3
			&& p.z() >= z0-1e-3 && p.z() <= z1+1e-3;
	}

	virtual void tessellate(TriSoup& out) const override {
		tri_add_quad(out, point3(x0,k,z0), point3(x0,k,z1),
						  point3(x1,k,z1), point3(x1,k,z0),
					 vec3(0,1,0), mp);
	}

private:
	std::shared_ptr<material> mp;
	real x0, x1, z0, z1, k;
};