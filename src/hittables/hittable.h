#ifndef HITTABLE_H
#define HITTABLE_H
#include <memory>
#include "core/ray.h"
#include "acceleration/aabb.h"
#include "core/interval.h"
#include "geometry/tri_soup.h"
class material;
struct hit_record {
	point3 p;
	vec3   normal;
	vec3   tangent   = vec3(1,0,0); 
	vec3   bitangent = vec3(0,1,0);  
	real t;
	bool   front_face;
	real u;
	real v;
	std::shared_ptr<material> mat_ptr;

	void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal     = front_face ? outward_normal : -outward_normal;
	}
};
class hittable {
public:
	virtual bool hit(
		const ray& r,
		const interval& ray_t,
		hit_record& rec
	) const = 0;
	virtual bool bounding_box(
		real time0,
		real time1,
		aabb& output_box
	) const = 0;
	virtual real pdf_value(const point3&, const vec3&) const { return 0.0; }
	virtual vec3   random(const point3&) const { return vec3(1,0,0); }

	// Area sampling for BDPT: a light subpath starts on the emitter with no
	// receiver, so pdf_value/random above do not apply. area() == 0 means the
	// shape cannot start one.
	virtual real area() const { return 0.0; }

	// Uniform point on the surface; ng is geometric, the caller picks the
	// emitting side (flip_face can reverse it).
	virtual point3 sample_area(real, real, vec3& ng) const {
		ng = vec3(0,1,0);
		return point3(0,0,0);
	}

	// Recovers the area density of a light vertex a camera subpath landed on.
	virtual bool contains_point(const point3&) const { return false; }

	// Display geometry for the viewport. Primitives with nothing to draw
	// leave the soup untouched.
	virtual void tessellate(TriSoup&) const {}

	virtual ~hittable() = default;
};
#endif