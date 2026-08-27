#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include "core/vec3.h"

class material;

// Display geometry. The integrators never see these — they exist so a viewport
// can rasterise the same scene the path tracer intersects, instead of the scene
// being described a second time by hand.
struct Tri {
	point3 v0, v1, v2;
	vec3   n0, n1, n2;
	std::shared_ptr<material> mat;
	// Participating media have no surface; their boundary stands in for them.
	bool   translucent = false;
};

using TriSoup = std::vector<Tri>;

inline void tri_add(TriSoup& out,
					const point3& a, const point3& b, const point3& c,
					const vec3& na, const vec3& nb, const vec3& nc,
					const std::shared_ptr<material>& m) {
	// Sphere pole caps collapse to a line, and a rect can be built with zero
	// extent; neither is worth a vertex.
	if (cross(b - a, c - a).length_squared() <= 0) return;
	out.push_back({a, b, c, na, nb, nc, m, false});
}

inline void tri_add_quad(TriSoup& out,
						 const point3& a, const point3& b,
						 const point3& c, const point3& d,
						 const vec3& n, const std::shared_ptr<material>& m) {
	tri_add(out, a, b, c, n, n, n, m);
	tri_add(out, a, c, d, n, n, n, m);
}

// Normals here are exact rather than faceted, so a coarse sphere still shades
// smoothly and only its silhouette gives the tessellation away.
inline void tri_add_uv_sphere(TriSoup& out, const point3& center, real radius,
							  const std::shared_ptr<material>& m,
							  int nu = 32, int nv = 16) {
	const real pi_val = (real)3.14159265358979323846;
	auto nrm = [&](int i, int j) {
		// Snapped, because sin(pi) is not zero in floating point: left alone,
		// the south pole becomes a ring of slivers instead of a single point.
		if (j == 0)  return vec3(0,  1, 0);
		if (j == nv) return vec3(0, -1, 0);
		real phi   = (real)2 * pi_val * (real)i / (real)nu;
		real theta =           pi_val * (real)j / (real)nv;
		return vec3(std::sin(theta) * std::cos(phi),
					std::cos(theta),
					std::sin(theta) * std::sin(phi));
	};
	for (int j = 0; j < nv; ++j)
		for (int i = 0; i < nu; ++i) {
			vec3 na = nrm(i,   j  ), nb = nrm(i+1, j  );
			vec3 nc = nrm(i+1, j+1), nd = nrm(i,   j+1);
			point3 a = center + radius*na, b = center + radius*nb;
			point3 c = center + radius*nc, d = center + radius*nd;
			tri_add(out, a, b, c, na, nb, nc, m);
			tri_add(out, a, c, d, na, nc, nd, m);
		}
}

// Bounding sphere of the display geometry, for framing the whole scene.
inline void tri_soup_bounds(const TriSoup& s, point3& center, real& radius) {
	if (s.empty()) { center = point3(0,0,0); radius = 1; return; }

	point3 lo = s[0].v0, hi = s[0].v0;
	for (const Tri& t : s)
		for (const point3& v : {t.v0, t.v1, t.v2})
			for (int i = 0; i < 3; ++i) {
				if (v[i] < lo[i]) lo[i] = v[i];
				if (v[i] > hi[i]) hi[i] = v[i];
			}

	center = (real)0.5 * (lo + hi);
	radius = (real)0.5 * (hi - lo).length();
	if (!(radius > 0)) radius = 1;
}
