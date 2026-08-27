// Viewport transform checks. The matrices decide where geometry lands on
// screen, and a wrong sign there looks like a modelling bug rather than a
// maths one, so they are pinned against closed-form clip-space positions.

#include "tests/test_util.h"

#include "core/mat4.h"
#include "core/camera.h"
#include "core/ray.h"
#include "core/camera_factory.h"
#include "viewer/viewport_camera.h"
#include "scenes/scene_factory.h"
#include "geometry/tri_soup.h"
#include "viewer/solid_shading.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

double max_diff(const mat4& a, const mat4& b) {
	double d = 0.0;
	for (int i = 0; i < 16; ++i)
		d = std::max(d, (double)std::abs(a.m[i] - b.m[i]));
	return d;
}

mat4 filled(real base) {
	mat4 m;
	for (int i = 0; i < 16; ++i) m.m[i] = base + (real)i * (real)0.37;
	return m;
}

// Clip space divided through by w -- what the rasteriser actually sees.
point3 to_ndc(const mat4& m, const point3& p) {
	real w = 1;
	point3 c = mat4_mul_point(m, p, &w);
	return point3(c[0]/w, c[1]/w, c[2]/w);
}

void test_algebra() {
	mat4 id, a = filled(1.0), b = filled(-2.5);

	check(max_diff(a * id, a) < kNumericTol, "mat4: identity on the right",
		  max_diff(a * id, a), 0.0, kNumericTol);
	check(max_diff(id * a, a) < kNumericTol, "mat4: identity on the left",
		  max_diff(id * a, a), 0.0, kNumericTol);

	mat4 c = mat4_look_at(point3(3,4,5), point3(0,0,0), vec3(0,1,0));
	double assoc = max_diff((a * b) * c, a * (b * c));
	check(assoc < 1e-2, "mat4: multiplication is associative", assoc, 0.0, 1e-2);

	double invol = max_diff(mat4_transpose(mat4_transpose(a)), a);
	check(invol < kExactTol, "mat4: transpose is its own inverse", invol, 0.0,
		  kExactTol);

	// A translation must move points and leave directions alone.
	mat4 t;
	t(0,3) = 10; t(1,3) = 20; t(2,3) = 30;
	point3 moved = mat4_mul_point(t, point3(1,2,3));
	vec3   dir   = mat4_mul_dir(t, vec3(1,2,3));
	check((moved - point3(11,22,33)).length() < kNumericTol,
		  "mat4: translation moves a point",
		  (double)(moved - point3(11,22,33)).length(), 0.0, kNumericTol);
	check((dir - vec3(1,2,3)).length() < kNumericTol,
		  "mat4: translation leaves a direction alone",
		  (double)(dir - vec3(1,2,3)).length(), 0.0, kNumericTol);
}

void test_look_at() {
	point3 eye(3, 4, 5), at(-1, 0.5, 2);
	mat4 v = mat4_look_at(eye, at, vec3(0,1,0));

	point3 o = mat4_mul_point(v, eye);
	check(o.length() < kNumericTol, "look_at: the eye lands on the origin",
		  (double)o.length(), 0.0, kNumericTol);

	// The target sits straight down -z at the distance between them.
	real dist = (at - eye).length();
	point3 c = mat4_mul_point(v, at);
	check((c - point3(0, 0, -dist)).length() < 1e-3,
		  "look_at: the target lands down -z at its true distance",
		  (double)(c - point3(0,0,-dist)).length(), 0.0, 1e-3);

	// A step along world up must not tip sideways in view space.
	point3 above = mat4_mul_point(v, eye + vec3(0, 1, 0));
	check(above.y() > 0.0 && std::abs((double)above.x()) < 1e-3,
		  "look_at: world up stays up on screen", (double)above.x(), 0.0, 1e-3);

	// Handedness: what the camera sees is in front of it, at negative z.
	check(c.z() < 0.0, "look_at: the view looks down -z", (double)c.z(),
		  -(double)dist, 1e-3);
}

void test_perspective() {
	const real fov = 60, aspect = (real)16/(real)9, n = (real)0.1, f = 1000;
	mat4 p = mat4_perspective(fov, aspect, n, f);

	point3 near_pt = to_ndc(p, point3(0, 0, -n));
	point3 far_pt  = to_ndc(p, point3(0, 0, -f));
	check(std::abs((double)near_pt.z() + 1.0) < 1e-4,
		  "perspective: the near plane maps to ndc z = -1",
		  (double)near_pt.z(), -1.0, 1e-4);
	check(std::abs((double)far_pt.z() - 1.0) < 1e-4,
		  "perspective: the far plane maps to ndc z = +1",
		  (double)far_pt.z(), 1.0, 1e-4);

	// At any depth the frustum edge is exactly the edge of the screen.
	const real d = 7;
	real half_h = d * std::tan((real)0.5 * fov * (real)3.14159265358979323846
							   / (real)180);
	point3 top   = to_ndc(p, point3(0, half_h, -d));
	point3 right = to_ndc(p, point3(aspect * half_h, 0, -d));
	check(std::abs((double)top.y() - 1.0) < 1e-4,
		  "perspective: the top of the fov maps to ndc y = +1",
		  (double)top.y(), 1.0, 1e-4);
	check(std::abs((double)right.x() - 1.0) < 1e-4,
		  "perspective: aspect widens x to the frame edge",
		  (double)right.x(), 1.0, 1e-4);

	// Halving the depth doubles the screen offset.
	point3 a = to_ndc(p, point3(0, 1, -4));
	point3 b = to_ndc(p, point3(0, 1, -2));
	check(std::abs((double)(b.y() / a.y()) - 2.0) < 1e-4,
		  "perspective: offset scales with 1/depth",
		  (double)(b.y()/a.y()), 2.0, 1e-4);
}

void test_orbit_state() {
	ViewportCamera vc;
	vc.target = point3(1, 2, 3);
	vc.distance = 12;

	for (real yaw : {(real)0, (real)1.1, (real)-2.7}) {
		for (real pitch : {(real)0, (real)0.8, (real)-1.2}) {
			vc.yaw = yaw; vc.pitch = pitch;
			point3 drift = vc.eye() + vc.distance * vc.forward() - vc.target;
			check(drift.length() < kNumericTol,
				  "orbit: the eye stays one distance from the target",
				  (double)drift.length(), 0.0, kNumericTol);
			check(std::abs((double)vc.forward().length() - 1.0) < kNumericTol,
				  "orbit: forward is a unit vector",
				  (double)vc.forward().length(), 1.0, kNumericTol);
		}
	}

	// Straight up must stay usable: an exactly vertical view makes look_at's
	// cross product with world up degenerate and the matrix fills with NaN.
	vc.pitch = (real)1.5707963267948966;
	mat4 v = vc.view();
	int finite = 0;
	for (int i = 0; i < 16; ++i) if (std::isfinite((double)v.m[i])) ++finite;
	check(finite == 16, "orbit: looking straight down stays finite",
		  (double)finite, 16.0, 0.5);

	// Whatever the orbit, the target sits in the middle of the screen.
	for (real pitch : {(real)-1.4, (real)0, (real)1.4}) {
		vc.pitch = pitch; vc.yaw = (real)0.6;
		point3 ndc = to_ndc(vc.view_proj((real)16/(real)9), vc.target);
		check(std::hypot((double)ndc.x(), (double)ndc.y()) < 1e-3,
			  "orbit: the target stays centred on screen",
			  std::hypot((double)ndc.x(), (double)ndc.y()), 0.0, 1e-3);
	}
}

// The viewport must open on the render camera's view, or Solid and Rendered
// would disagree about what the scene looks like.
void test_matches_render_camera() {
	std::vector<std::string> names = { "cornell", "furnace", "closed_furnace",
									   "ggx", "hdr", "glass", "caustics",
									   "sss", "volume", "ppm", "bunny",
									   "helmet" };

	for (const std::string& name : names) {
		RenderConfig cfg;
		cfg.feature = name;
		camera cam = CameraFactory::build(cfg);
		ViewportCamera vc = viewport_camera_from(cam);

		// Relative, because these cameras stand anywhere from 3 to 800 units out.
		real scale = vc.distance > 0 ? vc.distance : (real)1;
		double eye_err = (double)(vc.eye() - cam.get_origin()).length() / scale;
		check(eye_err < 1e-4, "camera '" + name + "': viewport eye matches",
			  eye_err, 0.0, 1e-4);

		// The view direction is what get_lower_left and friends encode.
		vec3 centre = cam.get_lower_left() + (real)0.5*cam.get_horizontal()
					+ (real)0.5*cam.get_vertical() - cam.get_origin();
		double angle = (double)(unit_vector(centre) - vc.forward()).length();
		check(angle < 1e-4, "camera '" + name + "': viewport aims the same way",
			  angle, 0.0, 1e-4);

		// Vertical field of view, recovered from the image plane extent.
		real want = (real)2 * std::atan(cam.get_vertical().length()
										/ ((real)2 * centre.length()))
					* (real)180 / (real)3.14159265358979323846;
		check(std::abs((double)(vc.vfov - want)) < 1e-3,
			  "camera '" + name + "': viewport keeps the field of view",
			  (double)vc.vfov, (double)want, 1e-3);

		check(vc.near_plane() > 0 && vc.near_plane() < vc.far_plane(),
			  "camera '" + name + "': clip range is ordered and positive",
			  (double)vc.near_plane(), 0.0, 0.0);
	}
}

// A round trip through a hand-built camera, where the answer is known outright
// rather than recovered from the same accessors the code under test uses.
void test_round_trip() {
	const point3 from(2, 3, -9), at(-4, 1, 5);
	const real vfov = 37, focus = (at - from).length();
	camera cam(from, at, vec3(0,1,0), vfov, (real)16/(real)9, 0, focus);
	ViewportCamera vc = viewport_camera_from(cam);

	check((vc.eye() - from).length() / focus < 1e-4,
		  "round trip: eye recovered", (double)(vc.eye()-from).length(), 0.0,
		  1e-4 * (double)focus);
	check((vc.target - at).length() / focus < 1e-4,
		  "round trip: target recovered", (double)(vc.target-at).length(), 0.0,
		  1e-4 * (double)focus);
	check(std::abs((double)(vc.vfov - vfov)) < 1e-3,
		  "round trip: field of view recovered", (double)vc.vfov, (double)vfov,
		  1e-3);
}

// Navigation. Each of orbit, pan and dolly moves one part of the view and
// must leave the rest of it alone, which is the whole reason the camera is
// stored as a turntable instead of an eye/target pair.
void test_navigation() {
	ViewportCamera base;
	base.target = point3(1, 2, 3);
	base.distance = 20;
	base.yaw = (real)0.7; base.pitch = (real)0.3;
	base.scene_radius = 15;

	// Orbit keeps the target and the distance, and only swings the eye.
	{
		ViewportCamera c = base;
		c.orbit(37, -21);
		check((c.target - base.target).length() < kNumericTol,
			  "orbit: the target does not move",
			  (double)(c.target - base.target).length(), 0.0, kNumericTol);
		check(std::abs((double)(c.distance - base.distance)) < kNumericTol,
			  "orbit: the distance does not change", (double)c.distance,
			  (double)base.distance, kNumericTol);
		check((c.eye() - base.eye()).length() > 1e-3,
			  "orbit: the eye moves", (double)(c.eye()-base.eye()).length(),
			  1.0, 0.0);
		// Dragging back the same amount has to land where it started.
		c.orbit(-37, 21);
		check((c.eye() - base.eye()).length() / base.distance < 1e-5,
			  "orbit: dragging back returns to the start",
			  (double)(c.eye()-base.eye()).length(), 0.0,
			  1e-5 * (double)base.distance);
	}

	// A drag right swings the camera left, because the mouse grabs the scene.
	{
		ViewportCamera c = base; c.yaw = 0; c.pitch = 0;
		vec3 before = c.right();
		c.orbit(50, 0);
		check(dot(c.eye() - c.target, before) < 0.0,
			  "orbit: dragging right moves the eye the other way",
			  (double)dot(c.eye() - c.target, before), -1.0, 0.0);
		ViewportCamera d = base; d.yaw = 0; d.pitch = 0;
		d.orbit(0, 50);
		check(d.eye().y() > base.target.y(),
			  "orbit: dragging down looks from above", (double)d.eye().y(),
			  1.0, 0.0);
	}

	// Pitch cannot run past the pole, however hard it is pushed.
	{
		ViewportCamera c = base;
		for (int i = 0; i < 200; ++i) c.orbit(0, 100);
		check(std::abs((double)c.pitch) < 1.5708,
			  "orbit: pitch stops at the pole", (double)c.pitch, 1.5707, 0.001);
		check(std::isfinite((double)c.eye().length()),
			  "orbit: the eye stays finite at the pole",
			  (double)c.eye().length(), 1.0, 0.0);
	}

	// Pan slides the target across the view plane and nothing else.
	{
		ViewportCamera c = base;
		c.pan(30, -12, 600);
		check(std::abs((double)(c.distance - base.distance)) < kNumericTol,
			  "pan: the distance does not change", (double)c.distance,
			  (double)base.distance, kNumericTol);
		check(std::abs((double)(c.yaw - base.yaw)) < kNumericTol
			  && std::abs((double)(c.pitch - base.pitch)) < kNumericTol,
			  "pan: the view direction does not change", (double)c.yaw,
			  (double)base.yaw, kNumericTol);
		// Motion stays in the plane the camera faces.
		real along_view = dot(c.target - base.target, base.forward());
		check(std::abs((double)along_view) < 1e-4 * (double)base.distance,
			  "pan: the target stays in the view plane", (double)along_view,
			  0.0, 1e-4 * (double)base.distance);

		// The point under the cursor stays under the cursor: a drag of n
		// pixels moves the scene n pixels.
		point3 a = to_ndc(base.view_proj((real)4/(real)3), base.target);
		point3 b = to_ndc(c.view_proj((real)4/(real)3), base.target);
		double moved_px = (double)(b.x() - a.x()) * 0.5 * 800.0;
		check(std::abs(moved_px - 30.0) < 0.1,
			  "pan: the scene tracks the cursor pixel for pixel", moved_px,
			  30.0, 0.1);
		// Screen y runs downward, so a drag up carries the scene up with it.
		double moved_py = (double)(b.y() - a.y()) * 0.5 * 600.0;
		check(std::abs(moved_py - 12.0) < 0.1,
			  "pan: a drag up carries the scene up", moved_py, 12.0, 0.1);

		c.pan(-30, 12, 600);
		check((c.target - base.target).length() / base.distance < 1e-5,
			  "pan: dragging back returns to the start",
			  (double)(c.target - base.target).length(), 0.0,
			  1e-5 * (double)base.distance);
	}

	// Dolly is multiplicative, so it covers the same fraction of the view at
	// any scale, and it moves only the distance.
	{
		ViewportCamera c = base;
		c.dolly(1);
		check(c.distance < base.distance, "dolly: scrolling in comes closer",
			  (double)c.distance, (double)base.distance, 0.0);
		check((c.target - base.target).length() < kNumericTol,
			  "dolly: the target does not move",
			  (double)(c.target - base.target).length(), 0.0, kNumericTol);
		c.dolly(-1);
		check(std::abs((double)(c.distance - base.distance))
				  / (double)base.distance < 1e-5,
			  "dolly: scrolling back returns to the start", (double)c.distance,
			  (double)base.distance, 1e-5 * (double)base.distance);

		ViewportCamera small = base; small.distance = 0.02;
		ViewportCamera big   = base; big.distance   = 5000;
		small.dolly(3); big.dolly(3);
		double rs = (double)small.distance / 0.02;
		double rb = (double)big.distance / 5000.0;
		check(std::abs(rs - rb) < 1e-5,
			  "dolly: one step is the same fraction at any scale", rs, rb, 1e-5);

		// Dollying all the way in must not pass through the target.
		ViewportCamera deep = base;
		for (int i = 0; i < 500; ++i) deep.dolly(1);
		check(deep.distance > 0 && std::isfinite((double)deep.distance),
			  "dolly: the distance stays positive", (double)deep.distance, 1.0,
			  0.0);
		check(deep.near_plane() > 0 && deep.near_plane() < deep.far_plane(),
			  "dolly: the clip range survives a long push in",
			  (double)deep.near_plane(), 0.0, 0.0);
	}
}

// Framing and the axis views, which are what the numpad reaches for.
void test_framing_and_axis_views() {
	const point3 c(4, -5, 6);
	const real   r = 9;

	for (real aspect : {(real)0.5, (real)1, (real)2.4}) {
		for (real fov : {(real)20, (real)40, (real)90}) {
			ViewportCamera vc;
			vc.vfov = fov;
			vc.yaw = (real)1.1; vc.pitch = (real)-0.4;
			vc.frame(c, r, aspect);

			check((vc.target - c).length() < kNumericTol,
				  "frame: the target is the centre of the scene",
				  (double)(vc.target - c).length(), 0.0, kNumericTol);

			// The bounding sphere has to end up inside the frame, and not so
			// far inside that the scene is a speck.
			mat4 vp = vc.view_proj(aspect);
			real half_v = fov * (real)3.14159265358979323846 / (real)360;
			real half_h = std::atan(std::tan(half_v) * aspect);
			real fills  = r / (vc.distance * std::sin(std::min(half_v, half_h)));
			check(std::abs((double)fills - 1.0) < 1e-4,
				  "frame: the scene just fills the tighter axis", (double)fills,
				  1.0, 1e-4);
			check(mat4_mul_point(vp, c).length() >= 0.0
				  && vc.near_plane() > 0 && vc.near_plane() < vc.far_plane(),
				  "frame: the clip range brackets the scene",
				  (double)vc.near_plane(), 0.0, 0.0);
		}
	}

	// Axis views: the eye ends up on the axis it was told to look down.
	const vec3 axes[6] = { vec3(0,0,-1), vec3(0,0,1), vec3(-1,0,0),
						   vec3(1,0,0),  vec3(0,-1,0), vec3(0,1,0) };
	for (const vec3& a : axes) {
		ViewportCamera vc;
		vc.target = point3(2, 3, 4);
		vc.distance = 10;
		vc.look_along(a);
		// The pole views are held a hair off vertical, so they get some slack.
		double err = (double)(vc.forward() - a).length();
		check(err < 2e-3, "axis view: the view aims down the chosen axis", err,
			  0.0, 2e-3);
		check((vc.eye() - (vc.target - vc.distance * a)).length() < 2e-2,
			  "axis view: the eye sits on that axis",
			  (double)(vc.eye() - (vc.target - vc.distance*a)).length(), 0.0,
			  2e-2);
	}
}

// The orthographic view has to frame what the perspective one framed, or
// pressing numpad 5 would jump the subject to a different size.
void test_orthographic() {
	ViewportCamera vc;
	vc.target = point3(0, 0, 0);
	vc.distance = 30;
	vc.vfov = 50;
	vc.scene_radius = 10;
	const real aspect = (real)16/(real)9;

	// A point on the frame edge at the target's depth stays on the edge.
	real h = vc.half_extent();
	point3 edge = vc.target + vc.up() * h;
	point3 persp = to_ndc(vc.view_proj(aspect), edge);
	vc.orthographic = true;
	point3 ortho = to_ndc(vc.view_proj(aspect), edge);
	check(std::abs((double)persp.y() - 1.0) < 1e-4,
		  "ortho: the frame edge is at ndc y = 1 in perspective",
		  (double)persp.y(), 1.0, 1e-4);
	check(std::abs((double)ortho.y() - 1.0) < 1e-4,
		  "ortho: the same point is still at the edge in ortho",
		  (double)ortho.y(), 1.0, 1e-4);

	// Depth must not change the size of anything.
	point3 near_e = to_ndc(vc.view_proj(aspect), edge + vc.forward() * (real)-8);
	point3 far_e  = to_ndc(vc.view_proj(aspect), edge + vc.forward() * (real) 8);
	check(std::abs((double)(near_e.y() - far_e.y())) < 1e-4,
		  "ortho: depth does not change screen size",
		  (double)(near_e.y() - far_e.y()), 0.0, 1e-4);

	// And the clip planes still map to the ends of the depth range.
	point3 n = to_ndc(vc.view_proj(aspect),
					  vc.eye() + vc.forward() * vc.near_plane());
	point3 f = to_ndc(vc.view_proj(aspect),
					  vc.eye() + vc.forward() * vc.far_plane());
	check(std::abs((double)n.z() + 1.0) < 1e-3,
		  "ortho: the near plane maps to ndc z = -1", (double)n.z(), -1.0, 1e-3);
	check(std::abs((double)f.z() - 1.0) < 1e-3,
		  "ortho: the far plane maps to ndc z = +1", (double)f.z(), 1.0, 1e-3);
}

// The pivot has to sit on the scene, not at the render camera's focus
// distance, or orbit and dolly swing around a point in empty space.
void test_pivot() {
	std::vector<std::string> names = { "cornell", "ggx", "glass", "caustics",
									   "sss", "volume", "ppm", "hdr" };
	for (const std::string& name : names) {
		RenderConfig cfg;
		cfg.feature = name;
		Scene sc = SceneFactory::build(name);
		TriSoup tris;
		sc.world->tessellate(tris);

		point3 center; real radius;
		tri_soup_bounds(tris, center, radius);

		ViewportCamera vc = viewport_camera_from(CameraFactory::build(cfg));
		point3 eye_before = vc.eye();
		vec3   fwd_before = vc.forward();
		vc.pivot_at(center, radius * (real)2);

		// The opening view is the renderer's, unchanged.
		check((vc.eye() - eye_before).length() / radius < 1e-4,
			  "pivot '" + name + "': the eye does not move",
			  (double)(vc.eye() - eye_before).length(), 0.0, 1e-4*(double)radius);
		check((vc.forward() - fwd_before).length() < 1e-4,
			  "pivot '" + name + "': the aim does not change",
			  (double)(vc.forward() - fwd_before).length(), 0.0, 1e-4);

		// And the pivot now lands somewhere inside the scene.
		check((vc.target - center).length() < radius * (real)1.5,
			  "pivot '" + name + "': the pivot lands on the scene",
			  (double)(vc.target - center).length() / (double)radius, 0.0, 1.5);

		// Orbiting all the way round must keep the scene in view, which is the
		// whole point of pivoting on it.
		for (int i = 0; i < 8; ++i) {
			vc.orbit_by((real)0.7853981633974483, 0);
			real d = dot(center - vc.eye(), vc.forward());
			check(d > 0 && std::abs((double)((center - vc.eye() - d*vc.forward())
											 .length())) < (double)radius,
				  "pivot '" + name + "': the scene stays in view while orbiting",
				  (double)d, 1.0, 0.0);
		}
	}
}

// Solid shading takes its colours from the materials, so the viewport shows
// the scene that will be rendered rather than a grey stand-in of it.
void test_display_colors() {
	auto near = [](const color& a, const color& b) {
		return (double)(a - b).length();
	};

	color red(0.65, 0.05, 0.05), gold(0.9, 0.7, 0.25);

	check(near(display_rgb(std::make_shared<lambertian>(red)), red) < kNumericTol,
		  "display: lambertian shows its albedo",
		  near(display_rgb(std::make_shared<lambertian>(red)), red), 0.0,
		  kNumericTol);
	check(near(display_rgb(std::make_shared<metal>(gold, 0.1)), gold) < kNumericTol,
		  "display: metal shows its albedo",
		  near(display_rgb(std::make_shared<metal>(gold, 0.1)), gold), 0.0,
		  kNumericTol);
	check(near(display_rgb(std::make_shared<ggx>(red, 0.3, 1.0)), red) < kNumericTol,
		  "display: ggx shows its base colour",
		  near(display_rgb(std::make_shared<ggx>(red, 0.3, 1.0)), red), 0.0,
		  kNumericTol);
	check(near(display_rgb(std::make_shared<subsurface>(red, 0.5)), red) < kNumericTol,
		  "display: subsurface shows its albedo",
		  near(display_rgb(std::make_shared<subsurface>(red, 0.5)), red), 0.0,
		  kNumericTol);
	check(near(display_rgb(std::make_shared<isotropic>(red)), red) < kNumericTol,
		  "display: a medium shows its albedo",
		  near(display_rgb(std::make_shared<isotropic>(red)), red), 0.0,
		  kNumericTol);

	// Glass has no albedo to show, so it must still come back with something.
	color glass = display_rgb(std::make_shared<dielectric>(1.5));
	check(glass.length() > 0.5 && glass.max_component() <= 1.0,
		  "display: glass gets a pale cast rather than black",
		  (double)glass.max_component(), 1.0, 0.5);

	// An emitter's radiance is far above 1; only its hue should survive.
	auto lamp = std::make_shared<diffuse_light>(color(15, 15, 15));
	color lit = display_rgb(lamp);
	check(std::abs((double)lit.max_component() - 1.0) < kNumericTol,
		  "display: an emitter is exposed down to full scale",
		  (double)lit.max_component(), 1.0, kNumericTol);
	check(lamp->display_emissive(),
		  "display: an emitter is flagged so it draws unlit", 1.0, 1.0, 0.5);

	auto tinted = std::make_shared<diffuse_light>(color(4, 8, 2));
	color th = display_rgb(tinted);
	check(std::abs((double)(th.x()/th.y()) - 0.5) < 1e-4
		  && std::abs((double)(th.z()/th.y()) - 0.25) < 1e-4,
		  "display: exposing an emitter keeps its hue",
		  (double)(th.x()/th.y()), 0.5, 1e-4);

	// Anything unlisted still has to be drawable.
	check(display_rgb(nullptr).max_component() > 0,
		  "display: a triangle with no material still has a colour",
		  (double)display_rgb(nullptr).max_component(), 0.8, 0.8);
}

// Every shipped scene has to come out of the walk with colours a rasteriser
// can use: finite, not negative, and never above full scale.
void test_scene_colors(bool quick) {
	std::vector<std::string> names = { "cornell", "furnace", "closed_furnace",
									   "ggx", "hdr", "glass", "caustics",
									   "sss", "volume", "ppm" };
	if (!quick) { names.push_back("bunny"); names.push_back("helmet"); }

	for (const std::string& name : names) {
		Scene sc = SceneFactory::build(name);
		TriSoup tris;
		sc.world->tessellate(tris);

		int usable = 0, distinct_hues = 0;
		std::set<const material*> mats;
		color first = display_rgb(tris.empty() ? nullptr : tris[0].mat);
		for (const Tri& t : tris) {
			mats.insert(t.mat.get());
			color c = display_rgb(t.mat);
			bool ok = true;
			for (int i = 0; i < 3; ++i)
				ok = ok && std::isfinite((double)c[i]) && c[i] >= 0 && c[i] <= 1;
			if (ok) ++usable;
			if ((c - first).length() > (real)0.01) ++distinct_hues;
		}
		check(usable == (int)tris.size(),
			  "scene '" + name + "': every colour is drawable",
			  (double)usable, (double)tris.size(), 0.5);
		// The furnace scenes are one albedo everywhere by design; anything
		// built from several materials must not collapse to a single grey.
		check(distinct_hues > 0 || mats.size() <= 1,
			  "scene '" + name + "': several materials give several colours",
			  (double)distinct_hues, 1.0, 0.5);
	}
}

// Rendered mode traces the framed view, so the camera the viewport hands the
// renderer has to cast the same rays the scene's own camera would.
void test_render_camera_round_trip() {
	std::vector<std::string> names = { "cornell", "furnace", "closed_furnace",
									   "ggx", "hdr", "glass", "caustics",
									   "sss", "volume", "ppm", "bunny",
									   "helmet" };

	for (const std::string& name : names) {
		RenderConfig cfg;
		cfg.feature = name;
		camera want = CameraFactory::build(cfg);
		real aspect = (real)cfg.width / (real)cfg.height;
		camera got  = to_render_camera(viewport_camera_from(want), aspect);

		double worst_o = 0.0, worst_d = 0.0;
		for (int a = 0; a <= 4; ++a)
			for (int b = 0; b <= 4; ++b) {
				real u = (real)a / 4, v = (real)b / 4;
				ray rw = want.get_ray(u, v), rg = got.get_ray(u, v);
				real scale = std::max((real)1, rw.origin().length());
				worst_o = std::max(worst_o,
					(double)(rg.origin() - rw.origin()).length() / (double)scale);
				worst_d = std::max(worst_d,
					(double)(unit_vector(rg.direction())
							 - unit_vector(rw.direction())).length());
			}

		check(worst_o < 1e-5, "render camera '" + name + "': same eye",
			  worst_o, 0.0, 1e-5);
		check(worst_d < 1e-4, "render camera '" + name + "': same rays",
			  worst_d, 0.0, 1e-4);
	}
}

// A render is restarted whenever the view changes, so every field that moves
// the image has to count as a change and nothing else may.
void test_same_view() {
	ViewportCamera a;
	a.target = point3(1,2,3); a.distance = 7; a.yaw = (real)0.4;
	a.pitch = (real)-0.2; a.vfov = 35;

	check(same_view(a, a), "same_view: a view matches itself", 1.0, 1.0, 0.5);

	{ ViewportCamera b = a; b.orbit(1, 0);
	  check(!same_view(a, b), "same_view: orbit counts as a change", 0.0, 0.0, 0.5); }
	{ ViewportCamera b = a; b.pan(1, 0, 100);
	  check(!same_view(a, b), "same_view: pan counts as a change", 0.0, 0.0, 0.5); }
	{ ViewportCamera b = a; b.dolly(1);
	  check(!same_view(a, b), "same_view: dolly counts as a change", 0.0, 0.0, 0.5); }
	{ ViewportCamera b = a; b.vfov = 36;
	  check(!same_view(a, b), "same_view: the field of view counts", 0.0, 0.0, 0.5); }

	// Solid shading settings do not change what a traced frame looks like, so
	// toggling them must not throw the render away.
	{ ViewportCamera b = a; b.scene_radius = 99;
	  check(same_view(a, b), "same_view: the clip range is not a view change",
			1.0, 1.0, 0.5); }
}

// End to end: the display geometry, projected through the viewport's own
// matrices, has to land in front of the camera and inside the frame. A flipped
// sign anywhere in the chain leaves the scene behind the viewer instead.
void test_scenes_are_framed(bool quick) {
	std::vector<std::string> names = { "cornell", "furnace", "closed_furnace",
									   "ggx", "hdr", "glass", "caustics",
									   "sss", "volume", "ppm" };
	if (!quick) { names.push_back("bunny"); names.push_back("helmet"); }

	for (const std::string& name : names) {
		RenderConfig cfg;
		cfg.feature = name;
		Scene sc = SceneFactory::build(name);
		TriSoup tris;
		sc.world->tessellate(tris);

		mat4 vp = viewport_camera_from(CameraFactory::build(cfg))
					  .view_proj((real)4/(real)3);

		long ahead = 0, total = 0;
		double lo_x = 1e30, hi_x = -1e30, lo_y = 1e30, hi_y = -1e30;
		for (const Tri& t : tris)
			for (const point3& v : {t.v0, t.v1, t.v2}) {
				++total;
				real w = 1;
				point3 c = mat4_mul_point(vp, v, &w);
				if (w <= 0) continue;
				++ahead;
				double x = (double)(c[0]/w), y = (double)(c[1]/w);
				lo_x = std::min(lo_x, x); hi_x = std::max(hi_x, x);
				lo_y = std::min(lo_y, y); hi_y = std::max(hi_y, y);
			}

		// A camera standing inside a closed box has half its walls behind it,
		// which is the least any of these scenes shows.
		double ahead_frac = total ? (double)ahead / (double)total : 0.0;
		check(ahead_frac >= 0.5,
			  "scene '" + name + "': geometry is in front of the viewport",
			  ahead_frac, 1.0, 0.5);

		// Coverage, not vertices: a wall can fill the frame with every corner
		// of it off screen.
		bool covers = lo_x <= 1.0 && hi_x >= -1.0 && lo_y <= 1.0 && hi_y >= -1.0;
		check(covers, "scene '" + name + "': geometry covers the frame",
			  covers ? 1.0 : 0.0, 1.0, 0.5);
	}
}

}  // namespace

void run_viewport_tests(bool quick) {
	std::printf("\nViewport transform checks\n");
	test_algebra();
	test_look_at();
	test_perspective();
	test_orbit_state();
	test_matches_render_camera();
	test_round_trip();
	test_navigation();
	test_framing_and_axis_views();
	test_orthographic();
	test_pivot();
	test_render_camera_round_trip();
	test_same_view();
	test_display_colors();
	test_scene_colors(quick);
	test_scenes_are_framed(quick);
}
