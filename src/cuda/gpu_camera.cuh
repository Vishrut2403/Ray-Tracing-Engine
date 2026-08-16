#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "cuda/cuda_rand.cuh"

struct GpuCamera {
	vec3   origin, lower_left, horizontal, vertical, u, v, w;
	real lens_radius;
};

__device__ inline ray gpu_get_ray(const GpuCamera& cam, real s, real t,
								   curandState* rng) {
	vec3 rd     = cam.lens_radius * rand_in_unit_sphere(rng);
	rd[2] = 0.0;
	vec3 offset = cam.u * rd.x() + cam.v * rd.y();
	return ray(cam.origin + offset,
			   cam.lower_left + s*cam.horizontal + t*cam.vertical
			   - cam.origin - offset, 0.0);
}

// Derived pinhole quantities for BDPT, which evaluates the camera as a light
// source. img_plane_dist is in *pixel* units, so the image plane has unit area
// per pixel and splats land in the accumulator's units. Aperture must be 0.
struct GpuCamAux {
	vec3   forward;         // unit vector along the view axis
	real img_plane_dist;  // aperture → image plane, in pixel units
	real mis_scale;       // 1/(W*H); see gpu_cam_pdf_dir_mis
	int    W, H;
};

__device__ inline GpuCamAux gpu_make_cam_aux(const GpuCamera& cam, int W, int H) {
	GpuCamAux a;
	vec3   center = cam.lower_left + 0.5*cam.horizontal + 0.5*cam.vertical;
	vec3   d      = center - cam.origin;
	real focal  = d.length();
	a.forward        = d / focal;
	a.img_plane_dist = focal * (real)W / cam.horizontal.length();
	a.mis_scale      = 1.0 / ((real)W * (real)H);
	a.W = W;
	a.H = H;
	return a;
}

// Solid-angle density of the primary ray, one ray drawn per pixel.
__device__ inline real gpu_cam_pdf_dir(const GpuCamAux& aux, const vec3& dir_unit) {
	real cos_t = dot(aux.forward, dir_unit);
	if (cos_t <= 1e-9) return 0.0;
	return (aux.img_plane_dist * aux.img_plane_dist) / (cos_t * cos_t * cos_t);
}

// Same density per *image* rather than per pixel: one camera ray per pixel but
// W*H light subpaths that may land anywhere, so the camera technique carries an
// extra 1/(W*H). MIS weights use this; estimators use the unscaled form above.
__device__ inline real gpu_cam_pdf_dir_mis(const GpuCamAux& aux,
											  const vec3& dir_unit) {
	return gpu_cam_pdf_dir(aux, dir_unit) * aux.mis_scale;
}

// World point to film. False if behind the camera or off-frame. Raster coords
// match gpu_get_ray: s,t in [0,1] span pixels 0..W-1 and 0..H-1.
__device__ inline bool gpu_project_to_pixel(const GpuCamera& cam,
											 const GpuCamAux& aux,
											 const vec3& p,
											 real& raster_x, real& raster_y) {
	vec3 d = p - cam.origin;
	if (dot(aux.forward, d) <= 1e-9) return false;

	vec3   nrm   = cross(cam.horizontal, cam.vertical);
	real denom = dot(d, nrm);
	if (fabs(denom) < 1e-12) return false;

	real lambda = dot(cam.lower_left - cam.origin, nrm) / denom;
	if (lambda <= 0.0) return false;

	vec3   q = cam.origin + lambda*d - cam.lower_left;
	real s = dot(q, cam.horizontal) / cam.horizontal.length_squared();
	real t = dot(q, cam.vertical)   / cam.vertical.length_squared();
	if (s < 0.0 || s >= 1.0 || t < 0.0 || t >= 1.0) return false;

	raster_x = s * (real)(aux.W - 1);
	raster_y = t * (real)(aux.H - 1);
	return true;
}
