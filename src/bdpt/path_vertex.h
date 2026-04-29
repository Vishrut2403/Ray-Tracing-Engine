#pragma once

#include "core/ray.h"
#include "core/vec3.h"
#include "hittables/hittable.h"
#include "materials/material.h"
#include "lights/env_light.h"
#include <memory>

// PathVertex — one vertex on a camera or light subpath

enum class VertexType {
	Camera,   // camera origin
	Light,    // point on a light source
	Surface,  // surface hit point
};

struct PathVertex {
	VertexType type = VertexType::Surface;

	point3 p;           // world-space position
	vec3   n;           // shading normal
	vec3   ng;          // geometric normal (same as n for us)
	double u, v;        // surface UV

	color  beta;        // accumulated throughput to this vertex
	bool   delta = false; // was sampled from a delta distribution?

	// Directional PDFs (solid angle measure) at this vertex:
	double pdf_fwd = 0.0;  // pdf of sampling THIS vertex from the previous one
	double pdf_rev = 0.0;  // pdf of sampling the PREVIOUS vertex from this one
						   // (filled in during MIS weight computation)

	std::shared_ptr<material> mat;  // null for camera/light vertices

	// Helper methods
	bool is_on_surface() const { return type == VertexType::Surface; }
	bool is_light()      const { return type == VertexType::Light;   }
	bool is_camera()     const { return type == VertexType::Camera;  }

	// Is this vertex connectable (non-delta)?
	bool is_connectable() const { return !delta; }

	// Evaluate BSDF at this vertex: incoming wi, outgoing wo
	color f(const vec3& wi, const vec3& wo) const {
		if (!mat) return color(0,0,0);
		ray dummy_ray(p - wi * 1e-3, wi, 0.0);
		hit_record rec;
		rec.p = p; rec.normal = n; rec.u = u; rec.v = v;
		rec.front_face = dot(wi, n) < 0.0 ? false : true;
		rec.mat_ptr = mat;
		return mat->f(dummy_ray, wo, rec);
	}

	// PDF of scattering from wi to wo at this vertex
	double pdf(const vec3& wi, const vec3& wo) const {
		if (!mat) return 0.0;
		ray dummy_ray(p - wi * 1e-3, wi, 0.0);
		hit_record rec;
		rec.p = p; rec.normal = n; rec.u = u; rec.v = v;
		rec.front_face = true;
		rec.mat_ptr = mat;
		return mat->pdf(dummy_ray, wo, rec);
	}

	// Geometry term G(this <-> other): cos/dist^2
	double G(const PathVertex& other) const {
		vec3  d    = other.p - p;
		double dist2 = d.length_squared();
		if (dist2 < 1e-10) return 0.0;
		double dist = std::sqrt(dist2);
		vec3   dn   = d / dist;
		double cos_a = std::abs(dot(n,  dn));
		double cos_b = std::abs(dot(other.n, -dn));
		return cos_a * cos_b / dist2;
	}

	// Convert pdf from solid-angle measure to area measure at 'other'
	double convert_density(double pdf_sa, const PathVertex& next) const {
		vec3   d    = next.p - p;
		double dist2 = d.length_squared();
		if (dist2 < 1e-10) return 0.0;
		double cos_t = std::abs(dot(next.n, -unit_vector(d)));
		return pdf_sa * cos_t / dist2;
	}

	// Build a PathVertex from a hit_record
	static PathVertex from_hit(const hit_record& rec, const color& beta,
								double pdf_fwd, bool delta = false) {
		PathVertex v;
		v.type    = VertexType::Surface;
		v.p       = rec.p;
		v.n       = rec.normal;
		v.ng      = rec.normal;
		v.u       = rec.u;
		v.v       = rec.v;
		v.beta    = beta;
		v.pdf_fwd = pdf_fwd;
		v.delta   = delta;
		v.mat     = rec.mat_ptr;
		return v;
	}

	// Build a camera vertex
	static PathVertex make_camera(const point3& origin) {
		PathVertex v;
		v.type  = VertexType::Camera;
		v.p     = origin;
		v.n     = vec3(0,0,1);  // placeholder
		v.ng    = vec3(0,0,1);
		v.beta  = color(1,1,1);
		v.pdf_fwd = 1.0;
		return v;
	}

	// Build a light vertex
	static PathVertex make_light(const point3& pos, const vec3& normal,
								  const color& Le, double pdf) {
		PathVertex v;
		v.type    = VertexType::Light;
		v.p       = pos;
		v.n       = normal;
		v.ng      = normal;
		v.beta    = Le;
		v.pdf_fwd = pdf;
		return v;
	}
};