#pragma once

// Bidirectional path tracing (Veach, ch. 10).
//
// Camera and light subpaths are traced independently and every (t,s) prefix pair
// is connected, combined with the balance heuristic. The t == 1 strategy splats
// a light vertex straight to the camera — caustics cannot be sampled any other
// way, since a connection at a specular vertex is always zero — hence the atomic
// framebuffer writes.
//
// Scope: closed scenes, area lights, pinhole camera (aperture 0). No env/
// background. Dielectrics ignore the radiance/importance asymmetry, which
// cancels for a closed refractive object.

#include "core/vec3.h"
#include "core/ray.h"
#include "core/onb.h"
#include "cuda/gpu_scene.cuh"
#include "cuda/gpu_hit.cuh"
#include "cuda/gpu_material.cuh"
#include "cuda/gpu_camera.cuh"
#include "cuda/cuda_rand.cuh"

// A depth-d path splits as (t,s) for any t in 1..d+2, so both subpaths must hold
// d+2 vertices or the MIS weight divides by splits that are never evaluated and
// the path loses that energy. Hence MAX_BDPT_DEPTH = MAX_BDPT_VERTS - 2.
#define MAX_BDPT_VERTS 12
#define MAX_BDPT_DEPTH (MAX_BDPT_VERTS - 2)
#define BDPT_RAY_EPS   0.02

struct GpuBDPTScene {
	const GpuHittable*   hittables;
	int                  n_hittables;
	const GpuMaterial*   materials;
	const int*           light_ids;
	int                  n_lights;
	const GpuTriangle*   tris;
	const GpuTriBVHNode* tri_bvh;
	int                  tri_root;
	int                  n_tris;
};

struct GpuPathVertex {
	vec3   p;
	vec3   n;
	vec3   beta;        // throughput of the subpath up to and including this vertex
	vec3   Le;          // emitted radiance, for vertices on emissive surfaces
	int    mat_id;
	bool   delta;       // scattering here is a delta lobe → not connectible
	bool   is_emissive; // sits on an emissive surface
	bool   is_light_ep; // is a light-subpath origin (emission, not a BSDF)
	bool   is_camera;
	real pdf_fwd;     // area density, sampled from the previous subpath vertex
	real pdf_rev;     // area density, sampled from the next vertex, reversed
};

__device__ inline real bdpt_remap0(real f) { return f != 0.0 ? f : 1.0; }

__device__ inline real gpu_convert_density(real pdf, const vec3& from_p,
											  const vec3& to_p, const vec3& to_n,
											  bool to_on_surface) {
	vec3   w  = to_p - from_p;
	real d2 = w.length_squared();
	if (d2 <= 1e-12) return 0.0;
	real inv = 1.0 / d2;
	if (to_on_surface) pdf *= fabs(dot(to_n, w * sqrt(inv)));
	return pdf * inv;
}

__device__ inline bool gpu_bdpt_visible(const GpuBDPTScene& sc,
										 const vec3& a, const vec3& b) {
	vec3   d    = b - a;
	real dist = d.length();
	if (dist < 1e-6) return true;
	GpuHitRecord tmp;
	return !hit_scene(sc.hittables, sc.n_hittables,
					  ray(a, d/dist, 0.0),
					  interval(BDPT_RAY_EPS, dist - BDPT_RAY_EPS), tmp,
					  sc.tris, sc.tri_bvh, sc.tri_root, sc.n_tris);
}

// Lights are axis-aligned xz rects, matching the rest of the GPU backend.

__device__ inline bool gpu_sample_light_point(const GpuBDPTScene& sc,
											   GpuSampler* rng,
											   vec3& pos, vec3& nrm, vec3& Le,
											   real& pdf_pos) {
	if (sc.n_lights <= 0) return false;
	int li = min((int)(rand_double(rng) * sc.n_lights), sc.n_lights - 1);
	const GpuHittable& lh = sc.hittables[sc.light_ids[li]];

	real rx = (real)lh.a0 + rand_double(rng) * (real)(lh.a1 - lh.a0);
	real rz = (real)lh.b0 + rand_double(rng) * (real)(lh.b1 - lh.b0);

	pos = vec3(rx, (real)lh.k, rz) + lh.translate_offset;
	nrm = lh.flip_face ? vec3(0,-1,0) : vec3(0,1,0);
	Le  = sc.materials[lh.mat_id].albedo;

	real area = (real)(lh.a1 - lh.a0) * (real)(lh.b1 - lh.b0);
	if (area <= 0.0) return false;
	pdf_pos = 1.0 / (area * (real)sc.n_lights);
	return true;
}

__device__ inline real gpu_pdf_light_origin(const GpuBDPTScene& sc,
											   const vec3& p) {
	for (int i = 0; i < sc.n_lights; ++i) {
		const GpuHittable& lh = sc.hittables[sc.light_ids[i]];
		vec3 lp = p - lh.translate_offset;
		if (fabs(lp.y() - (real)lh.k) < 1e-3 &&
			lp.x() >= (real)lh.a0 - 1e-3 && lp.x() <= (real)lh.a1 + 1e-3 &&
			lp.z() >= (real)lh.b0 - 1e-3 && lp.z() <= (real)lh.b1 + 1e-3) {
			real area = (real)(lh.a1 - lh.a0) * (real)(lh.b1 - lh.b0);
			if (area <= 0.0) return 0.0;
			return 1.0 / (area * (real)sc.n_lights);
		}
	}
	return 0.0;
}

// Area density at `next` of the emission leaving light vertex lv.
__device__ inline real gpu_pdf_light_dir(const GpuPathVertex& lv,
											const GpuPathVertex& next) {
	vec3   w  = next.p - lv.p;
	real d2 = w.length_squared();
	if (d2 <= 1e-12) return 0.0;
	real inv = 1.0 / d2;
	w = w * sqrt(inv);

	real cos_l = dot(lv.n, w);
	if (cos_l <= 0.0) return 0.0;

	real pdf = (cos_l / GPU_PI) * inv;
	if (!next.is_camera) pdf *= fabs(dot(next.n, w));
	return pdf;
}

// Area density at `next` of continuing from `cur`, entered from `prev`.
__device__ inline real gpu_vertex_pdf(const GpuBDPTScene& sc,
										 const GpuCamAux& aux,
										 const GpuPathVertex* prev,
										 const GpuPathVertex& cur,
										 const GpuPathVertex& next) {
	if (cur.is_light_ep) return gpu_pdf_light_dir(cur, next);

	vec3   wn = next.p - cur.p;
	real d2 = wn.length_squared();
	if (d2 <= 1e-12) return 0.0;
	wn = wn / sqrt(d2);

	real pdf_dir = 0.0;
	if (cur.is_camera) {
		pdf_dir = gpu_cam_pdf_dir_mis(aux, wn);
	} else {
		if (!prev) return 0.0;
		vec3 wp = prev->p - cur.p;
		if (wp.length_squared() <= 1e-12) return 0.0;
		wp = unit_vector(wp);
		pdf_dir = gpu_pdf_dir(sc.materials[cur.mat_id], wp, wn, cur.n);
	}
	return gpu_convert_density(pdf_dir, cur.p, next.p, next.n, !next.is_camera);
}

// ── Subpath construction ─────────────────────────────────────────────────────

__device__ inline int gpu_random_walk(const GpuBDPTScene& sc,
									   ray r, vec3 beta, real pdf_dir,
									   int max_verts, GpuPathVertex* path,
									   int idx, GpuSampler* rng) {
	real pdf_fwd = pdf_dir, pdf_rev = 0.0;

	while (idx < max_verts) {
		GpuHitRecord rec;
		if (!hit_scene(sc.hittables, sc.n_hittables, r,
					   interval(BDPT_RAY_EPS, 1e30), rec,
					   sc.tris, sc.tri_bvh, sc.tri_root, sc.n_tris)) break;

		const GpuMaterial& m    = sc.materials[rec.mat_id];
		GpuPathVertex&     v    = path[idx];
		GpuPathVertex&     prev = path[idx-1];

		v.p           = rec.p;
		v.n           = rec.normal;
		v.beta        = beta;
		v.Le          = gpu_emitted(m, rec.front_face);
		v.mat_id      = rec.mat_id;
		v.delta       = false;
		v.is_emissive = (m.type == MatType::DIFFUSE_LIGHT);
		v.is_light_ep = false;
		v.is_camera   = false;
		v.pdf_fwd     = gpu_convert_density(pdf_fwd, prev.p, v.p, v.n, true);
		v.pdf_rev     = 0.0;

		++idx;
		if (idx >= max_verts) break;

		vec3          wo = -unit_vector(r.direction());
		GpuBSDFSample bs = gpu_sample_dir(m, wo, rec, rng);
		if (bs.pdf <= 0.0) break;
		if (!(bs.f.length_squared() > 0.0)) break;

		pdf_fwd = bs.pdf;
		pdf_rev = bs.is_delta ? 0.0
							  : gpu_pdf_dir(m, bs.wi, wo, rec.normal);
		if (bs.is_delta) { v.delta = true; pdf_fwd = 0.0; }

		beta = beta * bs.f * (fabs(dot(rec.normal, bs.wi)) / bs.pdf);
		if (!(beta.length_squared() > 0.0)) break;

		prev.pdf_rev = gpu_convert_density(pdf_rev, v.p, prev.p, prev.n,
										   !prev.is_camera);
		r = ray(rec.p, bs.wi, 0.0);
	}
	return idx;
}

__device__ inline int gpu_generate_camera_subpath(const GpuBDPTScene& sc,
												   const GpuCamAux& aux,
												   const GpuCamera& cam,
												   const ray& r, int max_verts,
												   GpuPathVertex* path,
												   GpuSampler* rng) {
	if (max_verts <= 0) return 0;

	GpuPathVertex& v0 = path[0];
	v0.p           = cam.origin;
	v0.n           = aux.forward;
	v0.beta        = vec3(1,1,1);
	v0.Le          = vec3(0,0,0);
	v0.mat_id      = -1;
	v0.delta       = false;
	v0.is_emissive = false;
	v0.is_light_ep = false;
	v0.is_camera   = true;
	v0.pdf_fwd     = 1.0;   // pinhole
	v0.pdf_rev     = 0.0;
	if (max_verts == 1) return 1;

	real pdf_dir = gpu_cam_pdf_dir_mis(aux, unit_vector(r.direction()));
	if (pdf_dir <= 0.0) return 1;

	return gpu_random_walk(sc, r, vec3(1,1,1), pdf_dir, max_verts, path, 1, rng);
}

__device__ inline int gpu_generate_light_subpath(const GpuBDPTScene& sc,
												  int max_verts,
												  GpuPathVertex* path,
												  GpuSampler* rng) {
	if (max_verts <= 0) return 0;

	vec3   lp, ln, Le;
	real pdf_pos;
	if (!gpu_sample_light_point(sc, rng, lp, ln, Le, pdf_pos)) return 0;

	GpuPathVertex& v0 = path[0];
	v0.p           = lp;
	v0.n           = ln;
	v0.beta        = Le;
	v0.Le          = Le;
	v0.mat_id      = -1;
	v0.delta       = false;
	v0.is_emissive = true;
	v0.is_light_ep = true;
	v0.is_camera   = false;
	v0.pdf_fwd     = pdf_pos;
	v0.pdf_rev     = 0.0;
	if (max_verts == 1) return 1;

	// Cosine-weighted about the side that radiates (ceiling lights face down).
	onb uvw; uvw.build_from_w(ln);
	vec3   dir   = unit_vector(uvw.local(rand_cosine_direction(rng)));
	real cos_e = dot(ln, dir);
	if (cos_e <= 1e-9) return 1;

	real pdf_dir = cos_e / GPU_PI;
	vec3   beta    = Le * (cos_e / (pdf_pos * pdf_dir));

	return gpu_random_walk(sc, ray(lp, dir, 0.0), beta, pdf_dir,
						   max_verts, path, 1, rng);
}

// Balance heuristic over every (s,t) split. The connection temporarily rewrites
// densities at the joined vertices and their neighbours, saved/restored here.

__device__ inline real gpu_mis_weight(const GpuBDPTScene& sc,
										 const GpuCamAux& aux,
										 GpuPathVertex* camv, int t,
										 GpuPathVertex* lightv, int s,
										 const GpuPathVertex& sampled,
										 bool has_sampled) {
	if (s + t == 2) return 1.0;

	GpuPathVertex* qs      = (s > 0) ? &lightv[s-1] : nullptr;
	GpuPathVertex* pt      = (t > 0) ? &camv[t-1]   : nullptr;
	GpuPathVertex* qsMinus = (s > 1) ? &lightv[s-2] : nullptr;
	GpuPathVertex* ptMinus = (t > 1) ? &camv[t-2]   : nullptr;

	// Substitute the endpoint that the connection sampled fresh.
	GpuPathVertex  save_ep;
	GpuPathVertex* ep_slot = nullptr;
	if (has_sampled) {
		if      (s == 1) ep_slot = &lightv[0];
		else if (t == 1) ep_slot = &camv[0];
		if (ep_slot) {
			save_ep  = *ep_slot;
			*ep_slot = sampled;
			if (s == 1) qs = &lightv[0];
			if (t == 1) pt = &camv[0];
		}
	}

	bool   save_pt_delta = false, save_qs_delta = false;
	real save_pt_rev = 0.0, save_ptM_rev = 0.0;
	real save_qs_rev = 0.0, save_qsM_rev = 0.0;

	if (pt) { save_pt_delta = pt->delta; pt->delta = false; }
	if (qs) { save_qs_delta = qs->delta; qs->delta = false; }

	if (pt) {
		save_pt_rev = pt->pdf_rev;
		pt->pdf_rev = (s > 0) ? gpu_vertex_pdf(sc, aux, qsMinus, *qs, *pt)
							  : gpu_pdf_light_origin(sc, pt->p);
	}
	if (ptMinus) {
		save_ptM_rev = ptMinus->pdf_rev;
		ptMinus->pdf_rev = (s > 0) ? gpu_vertex_pdf(sc, aux, qs, *pt, *ptMinus)
								   : gpu_pdf_light_dir(*pt, *ptMinus);
	}
	if (qs) {
		save_qs_rev = qs->pdf_rev;
		qs->pdf_rev = gpu_vertex_pdf(sc, aux, ptMinus, *pt, *qs);
	}
	if (qsMinus) {
		save_qsM_rev = qsMinus->pdf_rev;
		qsMinus->pdf_rev = gpu_vertex_pdf(sc, aux, pt, *qs, *qsMinus);
	}

	real sumRi = 0.0, ri = 1.0;
	for (int i = t - 1; i > 0; --i) {
		ri *= bdpt_remap0(camv[i].pdf_rev) / bdpt_remap0(camv[i].pdf_fwd);
		if (!camv[i].delta && !camv[i-1].delta) sumRi += ri;
	}
	ri = 1.0;
	for (int i = s - 1; i >= 0; --i) {
		ri *= bdpt_remap0(lightv[i].pdf_rev) / bdpt_remap0(lightv[i].pdf_fwd);
		bool prev_delta = (i > 0) ? lightv[i-1].delta : false;
		if (!lightv[i].delta && !prev_delta) sumRi += ri;
	}

	if (qsMinus) qsMinus->pdf_rev = save_qsM_rev;
	if (qs)      { qs->pdf_rev = save_qs_rev; qs->delta = save_qs_delta; }
	if (ptMinus) ptMinus->pdf_rev = save_ptM_rev;
	if (pt)      { pt->pdf_rev = save_pt_rev; pt->delta = save_pt_delta; }
	if (ep_slot) *ep_slot = save_ep;

	return 1.0 / (1.0 + sumRi);
}

// MIS-weighted contribution of strategy (s,t). For t == 1 the result belongs to
// the pixel written to raster_x/raster_y, not the calling thread's pixel.

__device__ inline vec3 gpu_connect_bdpt(const GpuBDPTScene& sc,
										 const GpuCamAux& aux,
										 const GpuCamera& cam,
										 GpuPathVertex* camv, int t,
										 GpuPathVertex* lightv, int s,
										 GpuSampler* rng,
										 real& raster_x, real& raster_y) {
	vec3          L(0,0,0);
	GpuPathVertex sampled{};
	bool          has_sampled = false;

	// Would real-count the s == 0 strategy.
	if (t > 1 && s != 0 && camv[t-1].is_light_ep) return vec3(0,0,0);

	if (s == 0) {
		const GpuPathVertex& pt = camv[t-1];
		if (!pt.is_emissive) return vec3(0,0,0);
		L = pt.beta * pt.Le;

	} else if (t == 1) {
		const GpuPathVertex& qs = lightv[s-1];
		if (qs.delta || qs.is_light_ep) return vec3(0,0,0);

		vec3   d     = cam.origin - qs.p;
		real dist2 = d.length_squared();
		if (dist2 < 1e-10) return vec3(0,0,0);
		real dist = sqrt(dist2);
		vec3   wi   = d / dist;

		if (!gpu_project_to_pixel(cam, aux, qs.p, raster_x, raster_y))
			return vec3(0,0,0);

		// Camera's area density here, also the importance-to-area splat Jacobian.
		real pdf_cam_A = gpu_cam_pdf_dir(aux, -wi)
						 * fabs(dot(qs.n, wi)) / dist2;
		if (pdf_cam_A <= 0.0) return vec3(0,0,0);

		vec3 wo = unit_vector(lightv[s-2].p - qs.p);
		vec3 f  = gpu_f_dir(sc.materials[qs.mat_id], wo, wi, qs.n);
		if (!(f.length_squared() > 0.0)) return vec3(0,0,0);

		if (!gpu_bdpt_visible(sc, qs.p, cam.origin)) return vec3(0,0,0);

		L = qs.beta * f * pdf_cam_A;

		sampled.p           = cam.origin;
		sampled.n           = aux.forward;
		sampled.beta        = vec3(1,1,1);
		sampled.Le          = vec3(0,0,0);
		sampled.mat_id      = -1;
		sampled.delta       = false;
		sampled.is_emissive = false;
		sampled.is_light_ep = false;
		sampled.is_camera   = true;
		sampled.pdf_fwd     = 1.0;
		sampled.pdf_rev     = 0.0;
		has_sampled         = true;

	} else if (s == 1) {
		// NEE: sample a fresh point on a light.
		const GpuPathVertex& pt = camv[t-1];
		if (pt.delta) return vec3(0,0,0);

		vec3   lp, ln, Le;
		real pdf_pos;
		if (!gpu_sample_light_point(sc, rng, lp, ln, Le, pdf_pos))
			return vec3(0,0,0);

		vec3   d     = lp - pt.p;
		real dist2 = d.length_squared();
		if (dist2 < 1e-10) return vec3(0,0,0);
		real dist = sqrt(dist2);
		vec3   wi   = d / dist;

		real cos_l = dot(ln, -wi);
		if (cos_l <= 1e-9) return vec3(0,0,0);

		real pdf_solid = pdf_pos * dist2 / cos_l;
		if (pdf_solid <= 0.0) return vec3(0,0,0);

		vec3 wo = unit_vector(camv[t-2].p - pt.p);
		vec3 f  = gpu_f_dir(sc.materials[pt.mat_id], wo, wi, pt.n);
		if (!(f.length_squared() > 0.0)) return vec3(0,0,0);

		if (!gpu_bdpt_visible(sc, pt.p, lp)) return vec3(0,0,0);

		L = pt.beta * f * Le * (fabs(dot(pt.n, wi)) / pdf_solid);

		sampled.p           = lp;
		sampled.n           = ln;
		sampled.beta        = Le / pdf_solid;
		sampled.Le          = Le;
		sampled.mat_id      = -1;
		sampled.delta       = false;
		sampled.is_emissive = true;
		sampled.is_light_ep = true;
		sampled.is_camera   = false;
		sampled.pdf_fwd     = pdf_pos;
		sampled.pdf_rev     = 0.0;
		has_sampled         = true;

	} else {
		const GpuPathVertex& qs = lightv[s-1];
		const GpuPathVertex& pt = camv[t-1];
		if (qs.delta || pt.delta) return vec3(0,0,0);

		vec3   d     = qs.p - pt.p;
		real dist2 = d.length_squared();
		if (dist2 < 1e-10) return vec3(0,0,0);
		real dist = sqrt(dist2);
		vec3   wi   = d / dist;

		vec3 wo_pt = unit_vector(camv[t-2].p   - pt.p);
		vec3 wo_qs = unit_vector(lightv[s-2].p - qs.p);

		vec3 f_pt = gpu_f_dir(sc.materials[pt.mat_id], wo_pt,  wi, pt.n);
		if (!(f_pt.length_squared() > 0.0)) return vec3(0,0,0);
		vec3 f_qs = gpu_f_dir(sc.materials[qs.mat_id], wo_qs, -wi, qs.n);
		if (!(f_qs.length_squared() > 0.0)) return vec3(0,0,0);

		real G = fabs(dot(pt.n, wi)) * fabs(dot(qs.n, -wi)) / dist2;
		if (G <= 0.0) return vec3(0,0,0);

		if (!gpu_bdpt_visible(sc, pt.p, qs.p)) return vec3(0,0,0);

		L = pt.beta * f_pt * f_qs * qs.beta * G;
	}

	if (!(L.length_squared() > 0.0)) return vec3(0,0,0);

	real w = gpu_mis_weight(sc, aux, camv, t, lightv, s, sampled, has_sampled);
	if (!(w > 0.0)) return vec3(0,0,0);

	return L * w;
}
