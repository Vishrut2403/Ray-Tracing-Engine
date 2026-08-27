#pragma once

#include <algorithm>
#include <cmath>
#include "core/mat4.h"
#include "core/camera.h"

// Blender-style turntable: a point being looked at, plus a direction and a
// distance to it. Orbit, pan and dolly each move exactly one of those, which
// is why the view is stored this way instead of as an eye/target pair.
struct ViewportCamera {
	point3 target       = point3(0, 0, 0);
	real   distance     = 5;
	real   yaw          = 0;    // radians, about +y
	real   pitch        = 0;    // radians, off the xz plane
	real   vfov         = 40;   // degrees, vertical
	bool   orthographic = false;
	// The extent being looked at, which sets the clip planes. Zero means it is
	// not known and the framing distance stands in for it.
	real   scene_radius = 0;

	// Straight up is a singularity for a turntable, so the poles stay open by
	// a hair rather than letting the up vector become parallel to the view.
	static real clamp_pitch(real p) {
		const real limit = (real)1.5707963267948966 - (real)1e-3;
		return std::max(-limit, std::min(limit, p));
	}

	vec3 forward() const {
		real p  = clamp_pitch(pitch);
		real cp = std::cos(p);
		return vec3(-cp * std::sin(yaw), -std::sin(p), -cp * std::cos(yaw));
	}

	vec3 right() const { return unit_vector(cross(forward(), vec3(0,1,0))); }
	vec3 up()    const { return cross(right(), forward()); }

	point3 eye() const { return target - distance * forward(); }

	// Half the world-space height of the frame at the target's depth. The
	// orthographic view reuses it so toggling does not resize the subject.
	real half_extent() const {
		return distance * std::tan(vfov * (real)0.008726646259971648);
	}

	// Derived rather than stored: dollying in past a fixed near plane clips
	// away whatever you dollied in to look at.
	real far_plane() const {
		real r = scene_radius > 0 ? scene_radius : distance;
		return distance + (real)4 * r;
	}
	real near_plane() const {
		// The lower bound keeps the depth buffer's range finite when the view
		// is pushed right up against a surface.
		return std::max(distance * (real)0.001, far_plane() * (real)1e-6);
	}

	mat4 view() const { return mat4_look_at(eye(), target, vec3(0, 1, 0)); }

	mat4 proj(real aspect) const {
		return orthographic
			? mat4_orthographic(half_extent(), aspect, near_plane(), far_plane())
			: mat4_perspective(vfov, aspect, near_plane(), far_plane());
	}

	mat4 view_proj(real aspect) const { return proj(aspect) * view(); }

	// Navigation. The mouse grabs the scene and moves it, which is why a drag
	// to the right swings the camera to the left.
	void orbit_by(real dyaw, real dpitch) {
		yaw   += dyaw;
		pitch  = clamp_pitch(pitch + dpitch);
	}

	void orbit(real dx_px, real dy_px) {
		const real per_pixel = (real)0.008;   // radians, about Blender's rate
		orbit_by(-dx_px * per_pixel, dy_px * per_pixel);
	}

	// Drag distances are in pixels, so the frame height is what converts them
	// to world units and the scene stays glued to the cursor.
	void pan(real dx_px, real dy_px, int viewport_height_px) {
		if (viewport_height_px <= 0) return;
		real per_pixel = (real)2 * half_extent() / (real)viewport_height_px;
		target = target - right() * (dx_px * per_pixel)
						+ up()    * (dy_px * per_pixel);
	}

	// Multiplicative, so a step covers the same fraction of the view whether
	// the scene is a unit sphere or a 555-unit box.
	void dolly(real steps) {
		distance *= std::pow((real)0.8, steps);
		real floor_dist = scene_radius > 0 ? scene_radius * (real)1e-4
										  : (real)1e-4;
		distance = std::max(distance, floor_dist);
	}

	void frame(const point3& center, real radius, real aspect) {
		target       = center;
		scene_radius = radius;
		// Fit the smaller of the two half-angles, so a tall window frames the
		// scene by its width rather than cutting the sides off.
		real half_v = vfov * (real)0.008726646259971648;
		real half_h = std::atan(std::tan(half_v) * aspect);
		distance    = radius / std::sin(std::min(half_v, half_h));
	}

	// Slides the pivot along the view ray to the depth of what is being looked
	// at, holding the eye and the aim where they are. A render camera's focus
	// distance is a depth-of-field setting and says nothing about what the view
	// is centred on -- Cornell's is 10 units, 790 short of the box -- so it
	// makes a useless thing to orbit around.
	void pivot_at(const point3& scene_center, real fallback) {
		point3 e = eye();
		vec3   f = forward();
		real   d = dot(scene_center - e, f);
		distance = d > 0 ? d : fallback;
		target   = e + distance * f;
	}

	// Axis views, as the numpad gives in Blender, adapted to this y-up world.
	void look_along(const vec3& axis) {
		yaw   = std::atan2(-axis.x(), -axis.z());
		pitch = clamp_pitch(std::asin(
					std::max((real)-1, std::min((real)1, -axis.y()))));
	}
};

// The viewport opens on the view the renderer would use, so that Solid shading
// and a path-traced frame frame the same thing rather than two hand-set views.
inline ViewportCamera viewport_camera_from(const camera& cam) {
	point3 origin = cam.get_origin();
	// The ray through the centre of the image; its length is the focus
	// distance, which is what get_horizontal and get_vertical are scaled by.
	vec3 centre = cam.get_lower_left() + (real)0.5 * cam.get_horizontal()
									   + (real)0.5 * cam.get_vertical() - origin;
	real focus = centre.length();
	vec3 fwd   = centre / focus;

	ViewportCamera vc;
	vc.target   = origin + focus * fwd;
	vc.distance = focus;
	vc.look_along(fwd);
	vc.vfov     = (real)2 * std::atan(cam.get_vertical().length() / ((real)2 * focus))
				  * (real)180 / (real)3.14159265358979323846;
	return vc;
}
