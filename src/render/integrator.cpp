#include "render/integrator.h"
#include "materials/material.h"
#include "lights/env_light.h"
#include <algorithm>
#include <cmath>
#include <vector>

static inline bool is_valid(const color& c) {
	return std::isfinite(c.x()) && std::isfinite(c.y()) && std::isfinite(c.z())
		&& c.x() >= 0.0 && c.y() >= 0.0 && c.z() >= 0.0;
}

static inline color safe(const color& c) {
	if (!is_valid(c)) return color(0,0,0);
	real lum = 0.2126*c.x() + 0.7152*c.y() + 0.0722*c.z();
	if (lum > 50.0) return c * (50.0 / lum);
	return c;
}

static inline real power_heuristic(real pdf_a, real pdf_b) {
	real a2 = pdf_a*pdf_a, b2 = pdf_b*pdf_b;
	return a2 / (a2 + b2 + 1e-12);
}

// Bidirectional path tracing (Veach, ch. 10). Port of cuda/gpu_bdpt.cuh — keep
// the two in step. Scope: closed scenes, area lights, pinhole camera, no env.

static const int    MAX_BDPT_VERTS = 12;
static const int    MAX_BDPT_DEPTH = MAX_BDPT_VERTS - 2;
static const real BDPT_RAY_EPS   = 1e-4;

struct BDPTVertex {
	point3 p;
	vec3   n;
	color  beta;
	color  Le;
	std::shared_ptr<material> mat;
	bool   delta       = false;
	bool   is_emissive = false;
	bool   is_light_ep = false;
	bool   is_camera   = false;
	real pdf_fwd     = 0.0;
	real pdf_rev     = 0.0;
};

// Matches GpuCamAux.
struct BDPTCamera {
	point3 origin, lower_left;
	vec3   horizontal, vertical, forward;
	real img_plane_dist;  // aperture to image plane, in pixel units
	real mis_scale;       // 1/(W*H)
	int    W, H;
};

static BDPTCamera make_bdpt_camera(const camera& cam, int W, int H) {
	BDPTCamera a;
	a.origin     = cam.get_origin();
	a.lower_left = cam.get_lower_left();
	a.horizontal = cam.get_horizontal();
	a.vertical   = cam.get_vertical();
	point3 center = a.lower_left + 0.5*a.horizontal + 0.5*a.vertical;
	vec3   d      = center - a.origin;
	real focal  = d.length();
	a.forward        = d / focal;
	a.img_plane_dist = focal * (real)W / a.horizontal.length();
	a.mis_scale      = 1.0 / ((real)W * (real)H);
	a.W = W; a.H = H;
	return a;
}

// Solid-angle density of a primary ray, one per pixel.
static real cam_pdf_dir(const BDPTCamera& c, const vec3& dir_unit) {
	real ct = dot(c.forward, dir_unit);
	if (ct <= 1e-9) return 0.0;
	return (c.img_plane_dist * c.img_plane_dist) / (ct*ct*ct);
}

// Per image rather than per pixel: W*H light subpaths may land anywhere, so the
// camera technique carries an extra 1/(W*H). Estimators use the form above.
static real cam_pdf_dir_mis(const BDPTCamera& c, const vec3& dir_unit) {
	return cam_pdf_dir(c, dir_unit) * c.mis_scale;
}

static bool project_to_pixel(const BDPTCamera& c, const point3& p,
							  real& rx, real& ry) {
	vec3 d = p - c.origin;
	if (dot(c.forward, d) <= 1e-9) return false;
	vec3   nrm   = cross(c.horizontal, c.vertical);
	real denom = dot(d, nrm);
	if (std::abs(denom) < 1e-12) return false;
	real lambda = dot(c.lower_left - c.origin, nrm) / denom;
	if (lambda <= 0.0) return false;
	vec3   q = c.origin + lambda*d - c.lower_left;
	real s = dot(q, c.horizontal) / c.horizontal.length_squared();
	real t = dot(q, c.vertical)   / c.vertical.length_squared();
	if (s < 0.0 || s >= 1.0 || t < 0.0 || t >= 1.0) return false;
	rx = s * (real)(c.W - 1);
	ry = t * (real)(c.H - 1);
	return true;
}

static real remap0(real f) { return f != 0.0 ? f : 1.0; }

static real convert_density(real pdf, const point3& from,
							   const point3& to, const vec3& to_n,
							   bool to_on_surface) {
	vec3   w  = to - from;
	real d2 = w.length_squared();
	if (d2 <= 1e-12) return 0.0;
	real inv = 1.0 / d2;
	if (to_on_surface) pdf *= std::abs(dot(to_n, w * std::sqrt(inv)));
	return pdf * inv;
}

static bool visible(const point3& p, const point3& q,
					const std::shared_ptr<hittable>& world) {
	vec3   d    = q - p;
	real dist = d.length();
	if (dist < 1e-6) return true;
	hit_record shadow;
	return !world->hit(ray(p, d/dist, 0.0),
					   interval(BDPT_RAY_EPS, dist - BDPT_RAY_EPS), shadow);
}

// Which side emits: flip_face can reverse the geometric normal, so probe both.
static bool probe_emission(const std::shared_ptr<hittable>& world,
							const point3& p, const vec3& ng,
							vec3& n_emit, color& Le) {
	for (int side = 0; side < 2; ++side) {
		vec3   nn = (side == 0) ? ng : -ng;
		point3 o  = p + nn * 1e-3;
		ray    probe(o, -nn, 0.0);
		hit_record rec;
		if (world->hit(probe, interval(1e-6, 1e-2), rec) && rec.mat_ptr) {
			color e = rec.mat_ptr->emitted(probe, rec, rec.u, rec.v, rec.p);
			if (e.length_squared() > 0.0) { n_emit = nn; Le = e; return true; }
		}
	}
	return false;
}

static int pick_light(const std::shared_ptr<hittable_list>& lights,
					   real u, int& n_usable) {
	n_usable = 0;
	for (const auto& o : lights->objects) if (o->area() > 0.0) ++n_usable;
	if (n_usable == 0) return -1;
	int want = std::min<real>((int)(u * n_usable), n_usable - 1);
	int seen = 0;
	for (int i = 0; i < (int)lights->objects.size(); ++i) {
		if (lights->objects[i]->area() <= 0.0) continue;
		if (seen == want) return i;
		++seen;
	}
	return -1;
}

static bool sample_light_point(const std::shared_ptr<hittable_list>& lights,
								const std::shared_ptr<hittable>& world,
								point3& pos, vec3& nrm, color& Le,
								real& pdf_pos) {
	if (!lights || lights->objects.empty()) return false;
	int n_usable = 0;
	int li = pick_light(lights, random_double(), n_usable);
	if (li < 0) return false;

	const auto& L = lights->objects[li];
	vec3 ng;
	// Separate statements: the order of a call's arguments is unspecified, and
	// these two draws are consecutive dimensions of the sample sequence.
	real su = random_double(), sv = random_double();
	pos = L->sample_area(su, sv, ng);
	if (!probe_emission(world, pos, ng, nrm, Le)) return false;

	real a = L->area();
	if (a <= 0.0) return false;
	pdf_pos = 1.0 / (a * (real)n_usable);
	return true;
}

static real pdf_light_origin(const std::shared_ptr<hittable_list>& lights,
								const point3& p) {
	if (!lights) return 0.0;
	int n_usable = 0;
	for (const auto& o : lights->objects) if (o->area() > 0.0) ++n_usable;
	if (n_usable == 0) return 0.0;
	for (const auto& o : lights->objects) {
		if (o->area() > 0.0 && o->contains_point(p))
			return 1.0 / (o->area() * (real)n_usable);
	}
	return 0.0;
}

// Area density at `next` of the emission leaving light vertex lv.
static real pdf_light_dir(const BDPTVertex& lv, const BDPTVertex& next) {
	vec3   w  = next.p - lv.p;
	real d2 = w.length_squared();
	if (d2 <= 1e-12) return 0.0;
	real inv = 1.0 / d2;
	w = w * std::sqrt(inv);
	real cos_l = dot(lv.n, w);
	if (cos_l <= 0.0) return 0.0;
	real pdf = (cos_l / pi) * inv;
	if (!next.is_camera) pdf *= std::abs(dot(next.n, w));
	return pdf;
}

static hit_record fake_rec(const BDPTVertex& v) {
	hit_record rec;
	rec.p = v.p; rec.normal = v.n; rec.u = 0.0; rec.v = 0.0;
	rec.front_face = true; rec.mat_ptr = v.mat;
	return rec;
}

static real vertex_pdf(const BDPTCamera& cam,
						  const BDPTVertex* prev,
						  const BDPTVertex& cur,
						  const BDPTVertex& next) {
	if (cur.is_light_ep) return pdf_light_dir(cur, next);

	vec3   wn = next.p - cur.p;
	real d2 = wn.length_squared();
	if (d2 <= 1e-12) return 0.0;
	wn = wn / std::sqrt(d2);

	real pdf_dir = 0.0;
	if (cur.is_camera) {
		pdf_dir = cam_pdf_dir_mis(cam, wn);
	} else {
		if (!prev || !cur.mat) return 0.0;
		vec3 wp = prev->p - cur.p;
		if (wp.length_squared() <= 1e-12) return 0.0;
		wp = unit_vector(wp);
		pdf_dir = cur.mat->pdf_dir(wp, wn, fake_rec(cur));
	}
	return convert_density(pdf_dir, cur.p, next.p, next.n, !next.is_camera);
}

static int random_walk(const std::shared_ptr<hittable>& world,
						ray r, color beta, real pdf_dir,
						int max_verts, BDPTVertex* path, int idx) {
	real pdf_fwd = pdf_dir, pdf_rev = 0.0;

	while (idx < max_verts) {
		hit_record rec;
		if (!world->hit(r, interval(BDPT_RAY_EPS, infinity), rec)) break;
		if (!rec.mat_ptr) break;

		BDPTVertex& v    = path[idx];
		BDPTVertex& prev = path[idx-1];

		v.p           = rec.p;
		v.n           = rec.normal;
		v.beta        = beta;
		v.Le          = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
		v.mat         = rec.mat_ptr;
		v.delta       = false;
		v.is_emissive = v.Le.length_squared() > 0.0;
		v.is_light_ep = false;
		v.is_camera   = false;
		v.pdf_fwd     = convert_density(pdf_fwd, prev.p, v.p, v.n, true);
		v.pdf_rev     = 0.0;

		++idx;
		if (idx >= max_verts) break;

		vec3       wo = -unit_vector(r.direction());
		BSDFSample bs = rec.mat_ptr->sample_dir(wo, rec);
		if (bs.pdf <= 0.0 || !is_valid(bs.f) || bs.f.length_squared() <= 0.0) break;

		pdf_fwd = bs.pdf;
		pdf_rev = bs.is_delta ? 0.0 : rec.mat_ptr->pdf_dir(bs.wi, wo, rec);
		if (bs.is_delta) { v.delta = true; pdf_fwd = 0.0; }

		color nb = bs.is_phase ? beta * (bs.f / bs.pdf)
							   : beta * bs.f * (std::abs(dot(rec.normal, bs.wi)) / bs.pdf);
		if (!is_valid(nb) || nb.length_squared() <= 0.0) break;
		beta = nb;

		prev.pdf_rev = convert_density(pdf_rev, v.p, prev.p, prev.n,
									   !prev.is_camera);
		r = ray(rec.p, bs.wi, r.time());
	}
	return idx;
}

static int generate_camera_subpath(const std::shared_ptr<hittable>& world,
									const BDPTCamera& cam, const ray& r,
									int max_verts, BDPTVertex* path) {
	if (max_verts <= 0) return 0;
	BDPTVertex& v0 = path[0];
	v0 = BDPTVertex();
	v0.p         = cam.origin;
	v0.n         = cam.forward;
	v0.beta      = color(1,1,1);
	v0.is_camera = true;
	v0.pdf_fwd   = 1.0;   // pinhole
	if (max_verts == 1) return 1;

	real pdf_dir = cam_pdf_dir_mis(cam, unit_vector(r.direction()));
	if (pdf_dir <= 0.0) return 1;
	return random_walk(world, r, color(1,1,1), pdf_dir, max_verts, path, 1);
}

static int generate_light_subpath(const std::shared_ptr<hittable>& world,
								   const std::shared_ptr<hittable_list>& lights,
								   int max_verts, BDPTVertex* path) {
	if (max_verts <= 0) return 0;
	point3 lp; vec3 ln; color Le; real pdf_pos;
	if (!sample_light_point(lights, world, lp, ln, Le, pdf_pos)) return 0;

	BDPTVertex& v0 = path[0];
	v0 = BDPTVertex();
	v0.p           = lp;
	v0.n           = ln;
	v0.beta        = Le;
	v0.Le          = Le;
	v0.is_emissive = true;
	v0.is_light_ep = true;
	v0.pdf_fwd     = pdf_pos;
	if (max_verts == 1) return 1;

	onb uvw; uvw.build_from_w(ln);
	vec3   dir   = unit_vector(uvw.local(random_cosine_direction()));
	real cos_e = dot(ln, dir);
	if (cos_e <= 1e-9) return 1;

	real pdf_dir = cos_e / pi;
	color  beta    = Le * (cos_e / (pdf_pos * pdf_dir));
	if (!is_valid(beta)) return 1;
	return random_walk(world, ray(lp, dir, 0.0), beta, pdf_dir, max_verts, path, 1);
}

static real bdpt_mis_weight(const BDPTCamera& cam,
							   const std::shared_ptr<hittable_list>& lights,
							   BDPTVertex* camv, int t,
							   BDPTVertex* lightv, int s,
							   const BDPTVertex& sampled, bool has_sampled) {
	if (s + t == 2) return 1.0;

	BDPTVertex* qs      = (s > 0) ? &lightv[s-1] : nullptr;
	BDPTVertex* pt      = (t > 0) ? &camv[t-1]   : nullptr;
	BDPTVertex* qsMinus = (s > 1) ? &lightv[s-2] : nullptr;
	BDPTVertex* ptMinus = (t > 1) ? &camv[t-2]   : nullptr;

	BDPTVertex  save_ep;
	BDPTVertex* ep_slot = nullptr;
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
		pt->pdf_rev = (s > 0) ? vertex_pdf(cam, qsMinus, *qs, *pt)
							  : pdf_light_origin(lights, pt->p);
	}
	if (ptMinus) {
		save_ptM_rev = ptMinus->pdf_rev;
		ptMinus->pdf_rev = (s > 0) ? vertex_pdf(cam, qs, *pt, *ptMinus)
								   : pdf_light_dir(*pt, *ptMinus);
	}
	if (qs) {
		save_qs_rev = qs->pdf_rev;
		qs->pdf_rev = vertex_pdf(cam, ptMinus, *pt, *qs);
	}
	if (qsMinus) {
		save_qsM_rev = qsMinus->pdf_rev;
		qsMinus->pdf_rev = vertex_pdf(cam, pt, *qs, *qsMinus);
	}

	real sumRi = 0.0, ri = 1.0;
	for (int i = t - 1; i > 0; --i) {
		ri *= remap0(camv[i].pdf_rev) / remap0(camv[i].pdf_fwd);
		if (!camv[i].delta && !camv[i-1].delta) sumRi += ri;
	}
	ri = 1.0;
	for (int i = s - 1; i >= 0; --i) {
		ri *= remap0(lightv[i].pdf_rev) / remap0(lightv[i].pdf_fwd);
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

// MIS-weighted contribution of (s,t); for t == 1 the result belongs to rx/ry.
static color connect_bdpt(const std::shared_ptr<hittable>& world,
						   const std::shared_ptr<hittable_list>& lights,
						   const BDPTCamera& cam,
						   BDPTVertex* camv, int t,
						   BDPTVertex* lightv, int s,
						   real& rx, real& ry) {
	color      L(0,0,0);
	BDPTVertex sampled;
	bool       has_sampled = false;

	if (t > 1 && s != 0 && camv[t-1].is_light_ep) return color(0,0,0);

	if (s == 0) {
		const BDPTVertex& pt = camv[t-1];
		if (!pt.is_emissive) return color(0,0,0);
		L = pt.beta * pt.Le;

	} else if (t == 1) {
		const BDPTVertex& qs = lightv[s-1];
		if (qs.delta || qs.is_light_ep || !qs.mat) return color(0,0,0);

		vec3   d     = cam.origin - qs.p;
		real dist2 = d.length_squared();
		if (dist2 < 1e-10) return color(0,0,0);
		real dist = std::sqrt(dist2);
		vec3   wi   = d / dist;

		if (!project_to_pixel(cam, qs.p, rx, ry)) return color(0,0,0);

		// Camera's area density here, also the splat Jacobian.
		real pdf_cam_A = cam_pdf_dir(cam, -wi) * std::abs(dot(qs.n, wi)) / dist2;
		if (pdf_cam_A <= 0.0) return color(0,0,0);

		vec3  wo = unit_vector(lightv[s-2].p - qs.p);
		color f  = qs.mat->f_dir(wo, wi, fake_rec(qs));
		if (f.length_squared() <= 0.0) return color(0,0,0);
		if (!visible(qs.p, cam.origin, world)) return color(0,0,0);

		L = qs.beta * f * pdf_cam_A;

		sampled = BDPTVertex();
		sampled.p         = cam.origin;
		sampled.n         = cam.forward;
		sampled.beta      = color(1,1,1);
		sampled.is_camera = true;
		sampled.pdf_fwd   = 1.0;
		has_sampled       = true;

	} else if (s == 1) {
		const BDPTVertex& pt = camv[t-1];
		if (pt.delta || !pt.mat) return color(0,0,0);

		point3 lp; vec3 ln; color Le; real pdf_pos;
		if (!sample_light_point(lights, world, lp, ln, Le, pdf_pos))
			return color(0,0,0);

		vec3   d     = lp - pt.p;
		real dist2 = d.length_squared();
		if (dist2 < 1e-10) return color(0,0,0);
		real dist = std::sqrt(dist2);
		vec3   wi   = d / dist;

		real cos_l = dot(ln, -wi);
		if (cos_l <= 1e-9) return color(0,0,0);
		real pdf_solid = pdf_pos * dist2 / cos_l;
		if (pdf_solid <= 0.0) return color(0,0,0);

		vec3  wo = unit_vector(camv[t-2].p - pt.p);
		color f  = pt.mat->f_dir(wo, wi, fake_rec(pt));
		if (f.length_squared() <= 0.0) return color(0,0,0);
		if (!visible(pt.p, lp, world)) return color(0,0,0);

		L = pt.beta * f * Le * (std::abs(dot(pt.n, wi)) / pdf_solid);

		sampled = BDPTVertex();
		sampled.p           = lp;
		sampled.n           = ln;
		sampled.beta        = Le / pdf_solid;
		sampled.Le          = Le;
		sampled.is_emissive = true;
		sampled.is_light_ep = true;
		sampled.pdf_fwd     = pdf_pos;
		has_sampled         = true;

	} else {
		const BDPTVertex& qs = lightv[s-1];
		const BDPTVertex& pt = camv[t-1];
		if (qs.delta || pt.delta || !qs.mat || !pt.mat) return color(0,0,0);

		vec3   d     = qs.p - pt.p;
		real dist2 = d.length_squared();
		if (dist2 < 1e-10) return color(0,0,0);
		real dist = std::sqrt(dist2);
		vec3   wi   = d / dist;

		vec3  wo_pt = unit_vector(camv[t-2].p   - pt.p);
		vec3  wo_qs = unit_vector(lightv[s-2].p - qs.p);
		color f_pt  = pt.mat->f_dir(wo_pt,  wi, fake_rec(pt));
		if (f_pt.length_squared() <= 0.0) return color(0,0,0);
		color f_qs  = qs.mat->f_dir(wo_qs, -wi, fake_rec(qs));
		if (f_qs.length_squared() <= 0.0) return color(0,0,0);

		real G = std::abs(dot(pt.n, wi)) * std::abs(dot(qs.n, -wi)) / dist2;
		if (G <= 0.0) return color(0,0,0);
		if (!visible(pt.p, qs.p, world)) return color(0,0,0);

		L = pt.beta * f_pt * f_qs * qs.beta * G;
	}

	if (!is_valid(L) || L.length_squared() <= 0.0) return color(0,0,0);

	real w = bdpt_mis_weight(cam, lights, camv, t, lightv, s,
							   sampled, has_sampled);
	if (!(w > 0.0)) return color(0,0,0);
	return L * w;
}

color bdpt_Li(
	const ray& camera_ray,
	const camera& cam,
	const std::shared_ptr<hittable>& world,
	const std::shared_ptr<hittable_list>& lights,
	int max_depth,
	BDPTSplatBuffer& splat
) {
	if (max_depth > MAX_BDPT_DEPTH) max_depth = MAX_BDPT_DEPTH;

	BDPTCamera bc = make_bdpt_camera(cam, splat.W, splat.H);

	BDPTVertex camv  [MAX_BDPT_VERTS];
	BDPTVertex lightv[MAX_BDPT_VERTS];

	int cam_max = std::min<real>(max_depth + 2, MAX_BDPT_VERTS);
	int lit_max = std::min<real>(max_depth + 2, MAX_BDPT_VERTS);

	int nt = generate_camera_subpath(world, bc, camera_ray, cam_max, camv);
	int ns = generate_light_subpath(world, lights, lit_max, lightv);

	// One light subpath per pixel; the splat spreads over the film, hence 1/(W*H).
	real inv_light_paths = bc.mis_scale;

	color L(0,0,0);

	for (int t = 1; t <= nt; ++t) {
		for (int s = 0; s <= ns; ++s) {
			int depth = t + s - 2;
			if ((s == 1 && t == 1) || depth < 0 || depth > max_depth) continue;

			real rx = 0.0, ry = 0.0;
			color c = connect_bdpt(world, lights, bc, camv, t, lightv, s, rx, ry);
			if (c.length_squared() <= 0.0) continue;

			if (t == 1) {
				// Clamp after the 1/(W*H): the contribution is carried undivided.
				splat.add((int)(rx + 0.5), (int)(ry + 0.5),
						  safe(c * inv_light_paths));
			} else {
				L += safe(c);
			}
		}
	}
	return safe(L);
}

color Li(
	const ray& initial_ray,
	const color& background,
	const std::shared_ptr<hittable>& world,
	const std::shared_ptr<hittable_list>& lights,
	int max_depth,
	const std::shared_ptr<env_light>& env
) {
	ray   r    = initial_ray;
	color L    (0,0,0);
	color beta (1,1,1);
	bool  specular_bounce = false;
	real last_bsdf_pdf  = 0.0;
	point3 prev_p         = initial_ray.origin();

	for (int depth = 0; depth < max_depth; ++depth) {
		hit_record rec;
		if (!world->hit(r, interval(0.001, infinity), rec)) {
			color Le = env ? env->Le(r.direction()) : background;
			if (specular_bounce || depth == 0
				|| !(lights && !lights->objects.empty()))
				L += beta * Le;
			else {
				// rec is unwritten on a miss; use the previous vertex.
				real lp = lights->pdf_value(prev_p, r.direction());
				L += beta * Le * power_heuristic(last_bsdf_pdf, lp);
			}
			break;
		}

		color emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
		if (emitted.length_squared() > 0.0) {
			if (specular_bounce || depth == 0) L += beta * emitted;
			else {
				// MIS partner is NEE's density at the previous vertex. From rec.p
				// (already on the light) it returns 0, giving this hit full
				// weight on top of NEE.
				real lp = (lights && !lights->objects.empty())
							? lights->pdf_value(prev_p, r.direction()) : 0.0;
				L += beta * emitted * power_heuristic(last_bsdf_pdf, lp);
			}
		}

		if (!specular_bounce && lights && !lights->objects.empty()) {
			vec3   to_light  = lights->random(rec.p);
			real distance  = to_light.length();
			vec3   wi        = to_light / distance;
			real light_pdf = lights->pdf_value(rec.p, wi);
			if (light_pdf > 0.0) {
				hit_record lr;
				bool blocked = world->hit(ray(rec.p, wi, r.time()),
										  interval(0.001, infinity), lr);
				color Le2(0,0,0);
				if (blocked)
					Le2 = lr.mat_ptr->emitted(ray(rec.p,wi,r.time()),
											  lr, lr.u, lr.v, lr.p);
				else if (env)
					Le2 = env->Le(wi);

				if (Le2.length_squared() > 0.0) {
					color  f   = rec.mat_ptr->f(r, wi, rec);
					real bp  = rec.mat_ptr->pdf(r, wi, rec);
					real wt  = power_heuristic(light_pdf, bp);
					real ct  = rec.mat_ptr->is_phase_function()
								 ? 1.0 : std::abs(dot(rec.normal, wi));
					L += beta * f * Le2 * ct * wt / light_pdf;
				}
			}
		}

		BSDFSample bs = rec.mat_ptr->sample(r, rec);
		if (bs.pdf <= 0.0) break;
		last_bsdf_pdf = bs.pdf;
		if (bs.is_delta)      { beta *= bs.f; specular_bounce = true; }
		else if (bs.is_phase) { beta *= bs.f / bs.pdf; specular_bounce = false; }
		else { beta *= bs.f * std::abs(dot(rec.normal,bs.wi)) / bs.pdf; specular_bounce = false; }
		if (depth >= 3) {
			real s = std::clamp<real>(beta.max_component(), 0.05, 0.95);
			if (random_double() > s) break;
			beta /= s;
		}
		prev_p = rec.p;
		r = ray(rec.p, bs.wi, r.time());
	}
	return L;
}