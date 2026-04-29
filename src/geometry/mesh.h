#pragma once

#include "hittables/hittable.h"
#include "hittables/hittable_list.h"
#include "core/rtweekend.h"
#include "acceleration/bvh.h"
#include "external/tiny_obj_loader.h"

#include <string>
#include <vector>
#include <iostream>

// Triangle with smooth per-vertex normals interpolated via barycentric coords
class triangle : public hittable {
public:
	point3 v0, v1, v2;
	vec3   n0, n1, n2;   // per-vertex normals
	std::shared_ptr<material> mat;

	triangle(const point3& a, const point3& b, const point3& c,
			 const vec3& na, const vec3& nb, const vec3& nc,
			 std::shared_ptr<material> m)
		: v0(a), v1(b), v2(c), n0(na), n1(nb), n2(nc), mat(m) {}

	// Flat-normal constructor (fallback)
	triangle(const point3& a, const point3& b, const point3& c,
			 std::shared_ptr<material> m)
		: v0(a), v1(b), v2(c), mat(m)
	{
		vec3 fn = unit_vector(cross(b - a, c - a));
		n0 = n1 = n2 = fn;
	}

	virtual bool hit(const ray& r, const interval& ray_t,
					 hit_record& rec) const override {
		const double eps = 1e-8;
		vec3 e1 = v1 - v0;
		vec3 e2 = v2 - v0;
		vec3 h  = cross(r.direction(), e2);
		double a = dot(e1, h);
		if (std::abs(a) < eps) return false;

		double f = 1.0 / a;
		vec3   s = r.origin() - v0;
		double u = f * dot(s, h);
		if (u < 0.0 || u > 1.0) return false;

		vec3   q = cross(s, e1);
		double v = f * dot(r.direction(), q);
		if (v < 0.0 || u + v > 1.0) return false;

		double t = f * dot(e2, q);
		if (!ray_t.surrounds(t)) return false;

		// Interpolate smooth normal using barycentric coords
		double w = 1.0 - u - v;
		vec3 smooth_normal = unit_vector(w*n0 + u*n1 + v*n2);

		rec.t       = t;
		rec.p       = r.at(t);
		rec.mat_ptr = mat;
		rec.u       = u;
		rec.v       = v;
		rec.set_face_normal(r, smooth_normal);
		return true;
	}

	virtual bool bounding_box(double, double, aabb& out) const override {
		point3 lo(fmin(fmin(v0.x(),v1.x()),v2.x()) - 1e-4,
				  fmin(fmin(v0.y(),v1.y()),v2.y()) - 1e-4,
				  fmin(fmin(v0.z(),v1.z()),v2.z()) - 1e-4);
		point3 hi(fmax(fmax(v0.x(),v1.x()),v2.x()) + 1e-4,
				  fmax(fmax(v0.y(),v1.y()),v2.y()) + 1e-4,
				  fmax(fmax(v0.z(),v1.z()),v2.z()) + 1e-4);
		out = aabb(lo, hi);
		return true;
	}
};

inline std::shared_ptr<hittable> load_obj(
	const std::string& path,
	std::shared_ptr<material> mat,
	double scale  = 1.0,
	vec3   offset = vec3(0,0,0)
) {
	tinyobj::attrib_t                attrib;
	std::vector<tinyobj::shape_t>    shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials,
								&warn, &err, path.c_str());
	if (!warn.empty()) std::cerr << "[OBJ] " << warn << "\n";
	if (!err.empty())  std::cerr << "[OBJ] " << err  << "\n";
	if (!ok) {
		std::cerr << "[OBJ] failed to load: " << path << "\n";
		return nullptr;
	}

	int nv = (int)attrib.vertices.size() / 3;

	// Accumulate area-weighted face normals per vertex
	std::vector<vec3> smooth_normals(nv, vec3(0,0,0));

	for (const auto& shape : shapes) {
		size_t idx_offset = 0;
		for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
			int fv = shape.mesh.num_face_vertices[f];
			if (fv != 3) { idx_offset += fv; continue; }

			int i0 = shape.mesh.indices[idx_offset + 0].vertex_index;
			int i1 = shape.mesh.indices[idx_offset + 1].vertex_index;
			int i2 = shape.mesh.indices[idx_offset + 2].vertex_index;

			auto vert = [&](int i) {
				return vec3(attrib.vertices[3*i]*scale + offset.x(),
							attrib.vertices[3*i+1]*scale + offset.y(),
							attrib.vertices[3*i+2]*scale + offset.z());
			};
			vec3 a = vert(i0), b = vert(i1), c = vert(i2);

			// Cross product magnitude = 2 * triangle area (area weighting)
			vec3 weighted_normal = cross(b - a, c - a);
			smooth_normals[i0] = smooth_normals[i0] + weighted_normal;
			smooth_normals[i1] = smooth_normals[i1] + weighted_normal;
			smooth_normals[i2] = smooth_normals[i2] + weighted_normal;

			idx_offset += fv;
		}
	}

	// Normalize accumulated normals
	for (auto& n : smooth_normals) {
		double len = n.length();
		if (len > 1e-8) n = n / len;
		else            n = vec3(0, 1, 0); // degenerate fallback
	}

	// Build triangles with smooth normals
	hittable_list tris;
	size_t tri_count = 0;

	for (const auto& shape : shapes) {
		size_t idx_offset = 0;
		for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
			int fv = shape.mesh.num_face_vertices[f];
			if (fv != 3) { idx_offset += fv; continue; }

			int i0 = shape.mesh.indices[idx_offset + 0].vertex_index;
			int i1 = shape.mesh.indices[idx_offset + 1].vertex_index;
			int i2 = shape.mesh.indices[idx_offset + 2].vertex_index;

			auto vert = [&](int i) -> point3 {
				return point3(attrib.vertices[3*i]*scale + offset.x(),
							  attrib.vertices[3*i+1]*scale + offset.y(),
							  attrib.vertices[3*i+2]*scale + offset.z());
			};

			tris.add(std::make_shared<triangle>(
				vert(i0), vert(i1), vert(i2),
				smooth_normals[i0], smooth_normals[i1], smooth_normals[i2],
				mat
			));
			++tri_count;
			idx_offset += fv;
		}
	}

	std::cerr << "[OBJ] loaded " << tri_count << " triangles (smooth normals) from "
			  << path << "\n";

	return std::make_shared<bvh_node>(
		tris.objects, 0, tris.objects.size(), 0.0, 1.0);
}