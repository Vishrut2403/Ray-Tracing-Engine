// Tessellation checks — the display geometry must agree with the geometry the
// integrators intersect, or the viewport would show a different scene than the
// one being rendered.

#include "tests/test_util.h"

#include "hittables/sphere.h"
#include "hittables/box.h"
#include "hittables/xy_rect.h"
#include "hittables/xz_rect.h"
#include "hittables/yz_rect.h"
#include "hittables/flip_face.h"
#include "hittables/translate.h"
#include "hittables/rotate_y.h"
#include "hittables/constant_medium.h"
#include "hittables/hittable_list.h"
#include "acceleration/bvh.h"
#include "geometry/mesh.h"
#include "materials/material.h"
#include "scenes/scene_factory.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

namespace {

std::shared_ptr<material> grey() {
	return std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
}

double soup_area(const TriSoup& s) {
	double a = 0.0;
	for (const auto& t : s)
		a += 0.5 * cross(t.v1 - t.v0, t.v2 - t.v0).length();
	return a;
}

TriSoup tess(const hittable& h) {
	TriSoup s;
	h.tessellate(s);
	return s;
}

// Winding must put the geometric normal on the same side as the shading
// normal, or a rasteriser culls exactly the faces the tracer can see.
void check_winding(const std::string& name, const TriSoup& s) {
	int bad = 0;
	for (const auto& t : s) {
		vec3 ng = cross(t.v1 - t.v0, t.v2 - t.v0);
		if (dot(ng, t.n0 + t.n1 + t.n2) <= 0.0) ++bad;
	}
	check(bad == 0, name + ": winding matches shading normals",
		  (double)bad, 0.0, 0.5);
}

void test_rects() {
	xy_rect xy(0, 2, 0, 3, 1, grey());
	xz_rect xz(0, 2, 0, 3, 1, grey());
	yz_rect yz(0, 2, 0, 3, 1, grey());

	for (auto* p : {(const hittable*)&xy, (const hittable*)&xz,
					(const hittable*)&yz}) {
		TriSoup s = tess(*p);
		check(s.size() == 2, "rect: two triangles", (double)s.size(), 2.0, 0.5);
		check(std::abs(soup_area(s) - p->area()) < kNumericTol,
			  "rect: tessellated area = area()", soup_area(s), p->area(),
			  kNumericTol);
		check_winding("rect", s);
	}

	// Each rect faces its own positive axis.
	const vec3 want[3] = { vec3(0,0,1), vec3(0,1,0), vec3(1,0,0) };
	const hittable* rects[3] = { &xy, &xz, &yz };
	for (int i = 0; i < 3; ++i) {
		TriSoup s = tess(*rects[i]);
		vec3 n = unit_vector(cross(s[0].v1 - s[0].v0, s[0].v2 - s[0].v0));
		check((n - want[i]).length() < kNumericTol,
			  "rect: outward normal on the positive axis",
			  (n - want[i]).length(), 0.0, kNumericTol);
	}
}

void test_box() {
	box b(point3(0,0,0), point3(1,2,3), grey());
	TriSoup s = tess(b);
	check(s.size() == 12, "box: twelve triangles", (double)s.size(), 12.0, 0.5);
	double want = 2.0 * (1*2 + 2*3 + 1*3);
	check(std::abs(soup_area(s) - want) < kNumericTol,
		  "box: tessellated area = 2(xy+yz+xz)", soup_area(s), want,
		  kNumericTol);
	check_winding("box", s);
}

void test_sphere() {
	const real r = 2.0;
	sphere sp(point3(1, -2, 3), r, grey());
	TriSoup s = tess(sp);

	// A 32x16 inscribed polyhedron falls a little short of the true area, and
	// converges as the segment count rises; 2% is the gap at this resolution.
	double want = 4.0 * 3.14159265358979323846 * r * r;
	double got  = soup_area(s);
	check(got < want && got > 0.98 * want,
		  "sphere: tessellated area approaches 4*pi*r^2", got, want, 0.02*want);
	check_winding("sphere", s);

	int outward = 0, on_surface = 0;
	for (const auto& t : s) {
		if (dot(t.n0, t.v0 - sp.center) > 0.0) ++outward;
		if (std::abs((t.v0 - sp.center).length() - r) < kNumericTol)
			++on_surface;
	}
	check(outward == (int)s.size(), "sphere: normals point outward",
		  (double)outward, (double)s.size(), 0.5);
	check(on_surface == (int)s.size(), "sphere: vertices lie on the sphere",
		  (double)on_surface, (double)s.size(), 0.5);
}

// Mesh triangles pass straight through, keeping their per-vertex normals.
void test_triangle() {
	vec3 na(0,0,1), nb(0,0.6,0.8), nc(0.6,0,0.8);
	triangle t(point3(0,0,0), point3(1,0,0), point3(0,1,0),
			   na, nb, nc, grey());
	TriSoup s = tess(t);
	check(s.size() == 1, "triangle: one triangle out", (double)s.size(),
		  1.0, 0.5);
	check(std::abs(soup_area(s) - 0.5) < kNumericTol,
		  "triangle: area preserved", soup_area(s), 0.5, kNumericTol);
	double drift = (s[0].n0 - na).length() + (s[0].n1 - nb).length()
				 + (s[0].n2 - nc).length();
	check(drift < kNumericTol, "triangle: per-vertex normals preserved",
		  drift, 0.0, kNumericTol);
	check_winding("triangle", s);
}

void test_transforms() {
	auto b = std::make_shared<box>(point3(0,0,0), point3(1,1,1), grey());
	TriSoup base = tess(*b);

	translate tr(b, vec3(10, 20, 30));
	TriSoup moved = tess(tr);
	check(moved.size() == base.size(), "translate: triangle count preserved",
		  (double)moved.size(), (double)base.size(), 0.5);
	double drift = 0.0;
	for (size_t i = 0; i < base.size(); ++i)
		drift = std::max(drift,
			(double)((moved[i].v0 - base[i].v0) - vec3(10,20,30)).length());
	check(drift < kNumericTol, "translate: every vertex shifts by the offset",
		  drift, 0.0, kNumericTol);

	// A quarter turn about Y sends +x to -z, matching what hit() does to rec.p.
	rotate_y rot(b, 90.0);
	TriSoup spun = tess(rot);
	check(std::abs(soup_area(spun) - soup_area(base)) < kNumericTol,
		  "rotate_y: area preserved", soup_area(spun), soup_area(base),
		  kNumericTol);
	check_winding("rotate_y", spun);

	flip_face ff(b);
	TriSoup flipped = tess(ff);
	check(flipped.size() == base.size(), "flip_face: triangle count preserved",
		  (double)flipped.size(), (double)base.size(), 0.5);
	int reversed = 0;
	for (size_t i = 0; i < base.size(); ++i) {
		vec3 a = cross(base[i].v1 - base[i].v0, base[i].v2 - base[i].v0);
		vec3 c = cross(flipped[i].v1 - flipped[i].v0,
					   flipped[i].v2 - flipped[i].v0);
		if (dot(a, c) < 0.0) ++reversed;
	}
	check(reversed == (int)base.size(), "flip_face: every winding reversed",
		  (double)reversed, (double)base.size(), 0.5);
	check_winding("flip_face", flipped);
}

void test_medium_marked() {
	auto b = std::make_shared<box>(point3(0,0,0), point3(1,1,1), grey());
	constant_medium cm(b, 0.5, color(0.2, 0.4, 0.9));
	TriSoup s = tess(cm);
	int marked = 0;
	for (const auto& t : s) if (t.translucent) ++marked;
	check(marked == (int)s.size() && !s.empty(),
		  "constant_medium: boundary marked translucent",
		  (double)marked, (double)s.size(), 0.5);
}

// A single-object BVH leaf points left and right at the same child.
void test_bvh_no_duplicates() {
	std::vector<std::shared_ptr<hittable>> objs;
	for (int i = 0; i < 5; ++i)
		objs.push_back(std::make_shared<xz_rect>(
			(real)i, (real)i+1, 0, 1, 0, grey()));

	hittable_list flat;
	for (auto& o : objs) flat.add(o);

	auto tree = std::make_shared<bvh_node>(objs, 0, objs.size(), 0.0, 1.0);

	TriSoup a = tess(flat), b = tess(*tree);
	check(a.size() == b.size(), "bvh: emits each object once",
		  (double)b.size(), (double)a.size(), 0.5);
	check(std::abs(soup_area(a) - soup_area(b)) < kNumericTol,
		  "bvh: total area matches the flat list", soup_area(b), soup_area(a),
		  kNumericTol);

	std::vector<std::shared_ptr<hittable>> one{ objs[0] };
	auto leaf = std::make_shared<bvh_node>(one, 0, 1, 0.0, 1.0);
	check(tess(*leaf).size() == 2, "bvh: single-object leaf is not doubled",
		  (double)tess(*leaf).size(), 2.0, 0.5);
}

// Every shipped scene must produce drawable geometry. The mesh scenes read
// models off disk, so they are held back from the edit/build loop -- but they
// are the only cover for the glTF and OBJ triangle paths.
void test_scenes(bool quick) {
	std::vector<std::string> names = { "cornell", "furnace", "closed_furnace",
									   "ggx", "hdr", "glass", "caustics",
									   "sss", "volume", "ppm" };
	if (!quick) { names.push_back("bunny"); names.push_back("helmet"); }

	for (const std::string& name : names) {
		const char* n = name.c_str();
		Scene sc = SceneFactory::build(n);
		TriSoup s;
		sc.world->tessellate(s);
		check(!s.empty(), std::string("scene '") + n + "': tessellates",
			  (double)s.size(), 1.0, 0.5);

		int finite = 0;
		for (const auto& t : s) {
			vec3 c = t.v0 + t.v1 + t.v2;
			if (std::isfinite((double)c.x()) && std::isfinite((double)c.y())
				&& std::isfinite((double)c.z())) ++finite;
		}
		check(finite == (int)s.size(),
			  std::string("scene '") + n + "': all vertices finite",
			  (double)finite, (double)s.size(), 0.5);
	}
}

}  // namespace

void run_tessellate_tests(bool quick) {
	std::printf("\nTessellation checks\n");
	test_rects();
	test_box();
	test_sphere();
	test_triangle();
	test_transforms();
	test_medium_marked();
	test_bvh_no_duplicates();
	test_scenes(quick);
}
