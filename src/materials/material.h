#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>
#include <cmath>

#include "core/rtweekend.h"
#include "core/ray.h"
#include "core/onb.h"
#include "hittables/hittable.h"
#include "textures/texture.h"
#include "materials/bsdf_sample.h"
#include "materials/ggx_energy.h"

class material {
public:
	virtual ~material() = default;

	virtual color emitted(
		const ray&, const hit_record&,
		real, real, const point3&
	) const { return color(0, 0, 0); }

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const = 0;
	virtual color      f  (const ray& wo, const vec3& wi, const hit_record& rec) const = 0;
	virtual real     pdf(const ray& wo, const vec3& wi, const hit_record& rec) const = 0;

	// Bidirectional interface for BDPT: wo and wi both point away from the
	// surface, f excludes the cosine, and delta lobes use f = weight/|cos| with
	// pdf = 1. Default is a non-scattering material.
	virtual color f_dir(const vec3&, const vec3&, const hit_record&) const {
		return color(0,0,0);
	}
	virtual real pdf_dir(const vec3&, const vec3&, const hit_record&) const {
		return 0.0;
	}
	virtual BSDFSample sample_dir(const vec3&, const hit_record&) const {
		return { vec3(0,0,0), color(0,0,0), 0.0, false };
	}

	// True for participating-media phase functions, which have no surface
	// normal and so take no cosine factor.
	virtual bool is_phase_function() const { return false; }

	// Viewport shading only. The integrators never call these; they exist so
	// the solid view can colour a surface without re-describing the scene.
	virtual color display_color()    const { return color(0.8, 0.8, 0.8); }
	virtual bool  display_emissive() const { return false; }
};

class lambertian : public material {
public:
	std::shared_ptr<texture> albedo;

	lambertian(const color& a) : albedo(std::make_shared<solid_color>(a)) {}
	lambertian(std::shared_ptr<texture> a) : albedo(a) {}

	virtual color display_color() const override {
		return albedo->value(0.5, 0.5, point3(0,0,0));
	}

	virtual BSDFSample sample(const ray&, const hit_record& rec) const override {
		onb uvw; uvw.build_from_w(rec.normal);
		vec3   wi      = uvw.local(random_cosine_direction());
		real cosine  = dot(rec.normal, wi);
		real pdf_val = cosine > 0.0 ? cosine / pi : 0.0;
		return { wi, albedo->value(rec.u, rec.v, rec.p) / pi, pdf_val, false };
	}

	virtual color f(const ray&, const vec3& wi, const hit_record& rec) const override {
		if (dot(rec.normal, wi) <= 0.0) return color(0, 0, 0);
		return albedo->value(rec.u, rec.v, rec.p) / pi;
	}

	virtual real pdf(const ray&, const vec3& wi, const hit_record& rec) const override {
		real c = dot(rec.normal, wi);
		return c > 0.0 ? c / pi : 0.0;
	}

	virtual color f_dir(const vec3& wo, const vec3& wi,
						const hit_record& rec) const override {
		if (dot(rec.normal, wi) <= 0.0 || dot(rec.normal, wo) <= 0.0)
			return color(0,0,0);
		return albedo->value(rec.u, rec.v, rec.p) / pi;
	}
	virtual real pdf_dir(const vec3& wo, const vec3& wi,
						   const hit_record& rec) const override {
		real c = dot(rec.normal, wi);
		if (c <= 0.0 || dot(rec.normal, wo) <= 0.0) return 0.0;
		return c / pi;
	}
	virtual BSDFSample sample_dir(const vec3& wo,
								  const hit_record& rec) const override {
		if (dot(rec.normal, wo) <= 0.0) return { vec3(0,0,0), color(0,0,0), 0.0, false };
		onb uvw; uvw.build_from_w(rec.normal);
		vec3   wi = unit_vector(uvw.local(random_cosine_direction()));
		real c  = dot(rec.normal, wi);
		if (c <= 0.0) return { wi, color(0,0,0), 0.0, false };
		return { wi, albedo->value(rec.u, rec.v, rec.p) / pi, c / pi, false };
	}
};

class metal : public material {
public:
	color  albedo;
	real fuzz;

	metal(const color& a, real f) : albedo(a), fuzz(f < 1.0 ? f : 1.0) {}

	virtual color display_color() const override { return albedo; }

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		vec3 r = unit_vector(reflect(unit_vector(wo.direction()), rec.normal)
							 + fuzz * random_in_unit_sphere());
		return { r, albedo, 1.0, true };
	}

	virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
	virtual real pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }

	virtual BSDFSample sample_dir(const vec3& wo,
								  const hit_record& rec) const override {
		vec3   wi = unit_vector(reflect(-wo, rec.normal)
								+ fuzz * random_in_unit_sphere());
		real c  = dot(wi, rec.normal);
		if (c <= 0.0) return { wi, color(0,0,0), 0.0, false };
		return { wi, albedo / std::max<real>(std::abs(c), 1e-9), 1.0, true };
	}
};

class dielectric : public material {
public:
	real ir;

	dielectric(real index_of_refraction) : ir(index_of_refraction) {}

	// Clear glass has no albedo to show, so it gets the pale cast a solid view
	// conventionally draws it with.
	virtual color display_color() const override {
		return color(0.82, 0.88, 0.95);
	}

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		vec3   ud    = unit_vector(wo.direction());
		real ratio = rec.front_face ? (1.0 / ir) : ir;
		real cos_t = fmin(dot(-ud, rec.normal), 1.0);
		real sin_t = std::sqrt(1.0 - cos_t * cos_t);
		vec3 dir = (ratio * sin_t > 1.0 || schlick(cos_t, ratio) > random_double())
				   ? reflect(ud, rec.normal)
				   : refract(ud, rec.normal, ratio);
		return { dir, color(1, 1, 1), 1.0, true };
	}

	virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
	virtual real pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }

	virtual BSDFSample sample_dir(const vec3& wo,
								  const hit_record& rec) const override {
		real ratio = rec.front_face ? (1.0 / ir) : ir;
		vec3   ud    = -wo;
		real cos_t = fmin(dot(wo, rec.normal), 1.0);
		real sin_t = std::sqrt(std::max<real>(0.0, 1.0 - cos_t * cos_t));
		vec3   dir   = (ratio * sin_t > 1.0 || schlick(cos_t, ratio) > random_double())
					   ? reflect(ud, rec.normal)
					   : refract(ud, rec.normal, ratio);
		dir = unit_vector(dir);
		return { dir, color(1,1,1) / std::max<real>(std::abs(dot(dir, rec.normal)), 1e-9),
				 1.0, true };
	}

private:
	static real schlick(real cos, real ri) {
		real r0 = (1.0 - ri) / (1.0 + ri); r0 *= r0;
		return r0 + (1.0 - r0) * std::pow(1.0 - cos, 5.0);
	}
};

class ggx : public material {
public:
	color  base_color;
	real roughness;
	real metallic;

	ggx(const color& base, real r, real m = 0.0)
		: base_color(base),
		  roughness(clamp(r, 0.001, 1.0)),
		  metallic (clamp(m, 0.0,   1.0)) {}

	virtual color display_color() const override { return base_color; }

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		return sample_dir(-unit_vector(wo.direction()), rec);
	}

	// Without the cosine: the integrators apply cos(theta) themselves.
	virtual color f(const ray& wo, const vec3& wi, const hit_record& rec) const override {
		if (dot(rec.normal, wi) <= 0.0) return color(0,0,0);
		return brdf(-unit_vector(wo.direction()), wi, rec.normal);
	}

	virtual real pdf(const ray& wo, const vec3& wi, const hit_record& rec) const override {
		return combined_pdf(-unit_vector(wo.direction()), wi, rec.normal);
	}

	virtual color f_dir(const vec3& wo, const vec3& wi,
						const hit_record& rec) const override {
		return brdf(wo, wi, rec.normal);
	}
	virtual real pdf_dir(const vec3& wo, const vec3& wi,
						   const hit_record& rec) const override {
		return combined_pdf(wo, wi, rec.normal);
	}
	virtual BSDFSample sample_dir(const vec3& wo,
								  const hit_record& rec) const override {
		const vec3& n = rec.normal;
		real alpha = roughness * roughness;
		onb uvw; uvw.build_from_w(n);
		vec3 wo_l(dot(wo, uvw.u()), dot(wo, uvw.v()), dot(wo, uvw.w()));
		if (wo_l.z() <= 0.0) return { vec3(0,0,0), color(0,0,0), 0.0, false };

		vec3 wi;
		if (random_double() < spec_prob()) {
			vec3 h_l = sample_vndf(wo_l, alpha);
			vec3 h   = unit_vector(uvw.local(h_l));
			wi = unit_vector(reflect(-wo, h));
		} else {
			wi = unit_vector(uvw.local(random_cosine_direction()));
		}
		if (dot(wi, n) <= 0.0) return { wi, color(0,0,0), 0.0, false };

		real p = combined_pdf(wo, wi, n);
		if (p <= 0.0) return { wi, color(0,0,0), 0.0, false };
		return { wi, brdf(wo, wi, n), p, false };
	}

private:
	static real D(real ndoth, real a) {
		// a2*c2 + (1-c2), not c2*(a2-1)+1: in single precision the latter
		// cancels to exactly zero near normal incidence for small alpha.
		real a2 = a*a, c2 = ndoth*ndoth;
		real d  = a2*c2 + std::max<real>(1.0 - c2, 0.0);
		return a2 / (pi * d * d);
	}

	static real G2(real ndotv, real ndotl, real a) {
		real a2 = a*a;
		real gv = ndotl * std::sqrt(a2 + (1.0-a2)*ndotv*ndotv);
		real gl = ndotv * std::sqrt(a2 + (1.0-a2)*ndotl*ndotl);
		return 2.0*ndotv*ndotl / (gv + gl + 1e-7);
	}

	static real G1(real ndotv, real a) {
		real a2 = a*a;
		return 2.0*ndotv / (ndotv + std::sqrt(a2 + (1.0-a2)*ndotv*ndotv));
	}

	color F(real vdoth) const {
		color f0 = color(0.04,0.04,0.04)*(1.0-metallic) + base_color*metallic;
		return f0 + (color(1,1,1)-f0) * std::pow(1.0-vdoth, 5.0);
	}

	// Two lobes: GGX specular + Lambertian. Sampler and pdf must agree, so both
	// pick a lobe and report the combined density.
	real spec_prob() const {
		color  f0 = color(0.04,0.04,0.04)*(1.0-metallic) + base_color*metallic;
		real ls = 0.2126*f0.x() + 0.7152*f0.y() + 0.0722*f0.z();
		real ld = (1.0-metallic) * (0.2126*base_color.x()
					+ 0.7152*base_color.y() + 0.0722*base_color.z());
		real t  = ls + ld;
		return clamp(t > 1e-9 ? ls / t : 1.0, 0.1, 0.9);
	}

	// ndoth must be formed exactly as brdf() forms it: at low roughness D is
	// delta-like, so a last-bit difference in ndoth moves f/pdf by orders of
	// magnitude instead of cancelling.
	real combined_pdf(const vec3& v, const vec3& l, const vec3& n) const {
		real ndotv = dot(n, v), ndotl = dot(n, l);
		if (ndotv <= 0.0 || ndotl <= 0.0) return 0.0;
		real alpha = roughness * roughness;
		vec3   h     = unit_vector(v + l);
		real ndoth = std::max<real>(dot(n, h), 1e-7);
		real ps    = spec_prob();
		return ps * vndf_pdf(ndotv, ndoth, alpha) + (1.0 - ps) * ndotl / pi;
	}

	// True BSDF, without the trailing cosine.
	color brdf(const vec3& v, const vec3& l, const vec3& n) const {
		real ndotv = dot(n,v), ndotl = dot(n,l);
		if (ndotv <= 0.0 || ndotl <= 0.0) return color(0,0,0);

		real a     = roughness * roughness;
		vec3   h     = unit_vector(v + l);
		real ndoth = std::max<real>(dot(n,h), 1e-7);
		real vdoth = std::max<real>(dot(v,h), 1e-7);

		color  Fval = F(vdoth);
		real Dval = D(ndoth, a);
		real Gval = G2(ndotv, ndotl, a);

		color  f0 = color(0.04,0.04,0.04)*(1.0-metallic) + base_color*metallic;
		color specular = Fval * (Dval * Gval / (4.0 * ndotv * ndotl));
		color ms       = ggx_ms_brdf(f0, roughness, ndotv, ndotl);
		color diffuse  = base_color / pi * (1.0 - metallic) * (color(1,1,1) - Fval);
		return specular + ms + diffuse;
	}

	static vec3 sample_vndf(const vec3& wo, real a) {
		vec3 vh = unit_vector(vec3(a*wo.x(), a*wo.y(), wo.z()));

		real lensq = vh.x()*vh.x() + vh.y()*vh.y();
		vec3 t1 = lensq > 0.0
				  ? vec3(-vh.y(), vh.x(), 0.0) / std::sqrt(lensq)
				  : vec3(1,0,0);
		vec3 t2 = cross(vh, t1);

		real r  = std::sqrt(random_double());
		real phi = 2.0 * pi * random_double();
		real t  = r * std::cos(phi);
		real s  = r * std::sin(phi);
		real bz = std::sqrt(std::max<real>(0.0, 1.0 - t*t - s*s));

		vec3 nh = t*t1 + s*t2 + bz*vh;
		return unit_vector(vec3(a*nh.x(), a*nh.y(), std::max<real>(0.0, nh.z())));
	}

	// The h.wo factor of the VNDF cancels, so it takes the two cosines only.
	static real vndf_pdf(real ndotwo, real ndoth, real a) {
		return D(ndoth, a) * G1(std::max<real>(ndotwo, 1e-7), a)
			 / (4.0 * std::max<real>(ndotwo, 1e-7));
	}
};

class diffuse_light : public material {
public:
	std::shared_ptr<texture> emit;

	diffuse_light(std::shared_ptr<texture> a) : emit(a) {}
	diffuse_light(color c) : emit(std::make_shared<solid_color>(c)) {}

	virtual color display_color() const override {
		return emit->value(0.5, 0.5, point3(0,0,0));
	}
	virtual bool display_emissive() const override { return true; }

	virtual color emitted(const ray&, const hit_record& rec,
						  real u, real v, const point3& p) const override {
		if (!rec.front_face) return color(0,0,0);
		return emit->value(u, v, p);
	}

	virtual BSDFSample sample(const ray&, const hit_record&) const override {
		return { vec3(0,0,0), color(0,0,0), 0.0, false };
	}
	virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
	virtual real pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }
};

// Henyey-Greenstein phase function; g = 0 is isotropic. Mirrors hg_phase /
// hg_sample on the GPU so both backends model the same medium.
class isotropic : public material {
public:
	virtual bool is_phase_function() const override { return true; }

	std::shared_ptr<texture> albedo;
	real g;

	isotropic(const color& c, real g_ = 0.0)
		: albedo(std::make_shared<solid_color>(c)), g(clamp(g_, -0.99, 0.99)) {}
	isotropic(std::shared_ptr<texture> a, real g_ = 0.0)
		: albedo(a), g(clamp(g_, -0.99, 0.99)) {}

	virtual color display_color() const override {
		return albedo->value(0.5, 0.5, point3(0,0,0));
	}

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		return sample_dir(-unit_vector(wo.direction()), rec);
	}

	virtual color f(const ray& wo, const vec3& wi, const hit_record& rec) const override {
		return f_dir(-unit_vector(wo.direction()), wi, rec);
	}

	virtual real pdf(const ray& wo, const vec3& wi, const hit_record& rec) const override {
		return pdf_dir(-unit_vector(wo.direction()), wi, rec);
	}

	// A phase function is spherical: wo points back along the incoming ray, so
	// the scattering angle is measured against -wo.
	virtual color f_dir(const vec3& wo, const vec3& wi,
						const hit_record& rec) const override {
		return albedo->value(rec.u, rec.v, rec.p) * hg(dot(-wo, wi));
	}
	virtual real pdf_dir(const vec3& wo, const vec3& wi,
						   const hit_record&) const override {
		return hg(dot(-wo, wi));
	}
	virtual BSDFSample sample_dir(const vec3& wo,
								  const hit_record& rec) const override {
		real cos_t;
		if (std::abs(g) < 1e-4) {
			cos_t = 1.0 - 2.0 * random_double();
		} else {
			real xi  = random_double();
			real sqr = (1.0 - g*g) / (1.0 - g + 2.0*g*xi);
			cos_t = (1.0 + g*g - sqr*sqr) / (2.0*g);
		}
		real sin_t = std::sqrt(std::max<real>(0.0, 1.0 - cos_t*cos_t));
		real phi   = 2.0 * pi * random_double();

		onb uvw; uvw.build_from_w(-wo);
		vec3 wi = unit_vector(uvw.local(vec3(sin_t*std::cos(phi),
											 sin_t*std::sin(phi), cos_t)));
		return { wi, albedo->value(rec.u, rec.v, rec.p) * hg(dot(-wo, wi)),
				 hg(dot(-wo, wi)), false, true };
	}

private:
	real hg(real cos_t) const {
		real g2  = g*g;
		real den = 1.0 + g2 - 2.0*g*cos_t;
		return (1.0 - g2) / (4.0*pi * den * std::sqrt(den) + 1e-12);
	}
};

// Subsurface scattering — Jensen dipole as a diffusion BRDF.
//
// The exit point is never displaced, so this is a BRDF: the analytic total
// diffuse reflectance of the dipole (Jensen et al. 2001) wrapped in Fresnel
// transmittance. f(), pdf() and sample() all describe that one BRDF so every
// estimator agrees. mean_free_path therefore has no effect on shading — Rd is
// scale-invariant in the exit radius. albedo_color is the scattering albedo a'.
class subsurface : public material {
public:
	color  albedo_color;
	real mean_free_path;
	real ior;

	subsurface(const color& alb, real mfp, real ior_ = 1.4)
		: albedo_color(alb),
		  mean_free_path(std::max<real>(mfp, 1e-4)),
		  ior(ior_),
		  Rd(total_diffuse_reflectance(alb, ior_)),
		  T_entry(1.0 - fdr(ior_)) {}

	virtual color display_color() const override { return albedo_color; }

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		return sample_dir(-unit_vector(wo.direction()), rec);
	}

	virtual color f(const ray&, const vec3& wi, const hit_record& rec) const override {
		return brdf(wi, rec.normal);
	}

	virtual real pdf(const ray&, const vec3& wi, const hit_record& rec) const override {
		real c = dot(rec.normal, wi);
		return c > 0.0 ? c / pi : 0.0;
	}

	virtual color f_dir(const vec3&, const vec3& wi,
						const hit_record& rec) const override {
		return brdf(wi, rec.normal);
	}
	virtual real pdf_dir(const vec3&, const vec3& wi,
						   const hit_record& rec) const override {
		real c = dot(rec.normal, wi);
		return c > 0.0 ? c / pi : 0.0;
	}
	virtual BSDFSample sample_dir(const vec3&,
								  const hit_record& rec) const override {
		onb uvw; uvw.build_from_w(rec.normal);
		vec3   wi = unit_vector(uvw.local(random_cosine_direction()));
		real c  = dot(rec.normal, wi);
		if (c <= 0.0) return { wi, color(0,0,0), 0.0, false };
		return { wi, brdf(wi, rec.normal), c / pi, false };
	}

private:
	color  Rd;       // total diffuse reflectance of the dipole, per channel
	real T_entry;  // hemispherical Fresnel transmittance on the way in

	// Entry uses the hemispherical average so f() needs only one direction.
	color brdf(const vec3& wi, const vec3& n) const {
		real c = dot(n, wi);
		if (c <= 0.0) return color(0,0,0);
		real r0 = (1.0 - ior) / (1.0 + ior); r0 *= r0;
		real T_exit = 1.0 - (r0 + (1.0 - r0) * std::pow(1.0 - c, 5.0));
		return Rd * (T_entry * T_exit / pi);
	}

	// Fresnel diffuse reflectance (Egan & Hilgeman approximation)
	static real fdr(real eta) {
		if (eta >= 1.0)
			return -0.4399 + 0.7099/eta - 0.3319/(eta*eta) + 0.0636/(eta*eta*eta);
		else
			return -1.4399/(eta*eta) + 0.7099/eta + 0.6681 + 0.0636*eta;
	}

	static color total_diffuse_reflectance(const color& alpha, real eta) {
		real fd = fdr(eta);
		real A  = (1.0 + fd) / std::max<real>(1.0 - fd, 1e-6);
		auto ch = [&](real ap) {
			ap = clamp(ap, 0.0, 0.999);
			real s = std::sqrt(3.0 * (1.0 - ap));
			return 0.5 * ap * (1.0 + std::exp(-(4.0/3.0) * A * s)) * std::exp(-s);
		};
		return color(ch(alpha.x()), ch(alpha.y()), ch(alpha.z()));
	}
};

class rough_dielectric : public material {
public:
	color  tint;
	real roughness;
	real ior;

	rough_dielectric(const color& t, real r, real ior = 1.5)
		: tint(t),
		  roughness(clamp(r, 0.001, 1.0)),
		  ior(ior) {}

	virtual color display_color() const override {
		return tint * color(0.82, 0.88, 0.95);
	}

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		return sample_dir(-unit_vector(wo.direction()), rec);
	}

	virtual color f(const ray& wo, const vec3& wi,
					const hit_record& rec) const override {
		return f_dir(-unit_vector(wo.direction()), wi, rec);
	}

	virtual real pdf(const ray& wo, const vec3& wi,
					   const hit_record& rec) const override {
		return pdf_dir(-unit_vector(wo.direction()), wi, rec);
	}

	virtual color f_dir(const vec3& wo, const vec3& wi,
						const hit_record& rec) const override {
		color f; real p;
		return eval(wo, wi, rec, f, p) ? f : color(0,0,0);
	}

	virtual real pdf_dir(const vec3& wo, const vec3& wi,
						   const hit_record& rec) const override {
		color f; real p;
		return eval(wo, wi, rec, f, p) ? p : 0.0;
	}

	virtual BSDFSample sample_dir(const vec3& v,
								  const hit_record& rec) const override {
		real alpha = roughness * roughness;
		real eta   = rec.front_face ? (1.0 / ior) : ior;
		const vec3& n = rec.normal;
		if (dot(n, v) <= 0.0) return { vec3(0,0,0), color(0,0,0), 0.0, false };

		onb uvw; uvw.build_from_w(n);
		vec3 v_l(dot(v, uvw.u()), dot(v, uvw.v()), dot(v, uvw.w()));
		vec3 h_l = sample_vndf(v_l, alpha);
		vec3 h   = unit_vector(uvw.local(h_l));
		if (dot(h, v) < 0.0) h = -h;

		real vdoth = clamp(dot(v, h), 0.0, 1.0);
		real F_val = schlick(vdoth, eta);

		vec3 wi;
		bool want_reflect = random_double() < F_val;
		if (!want_reflect) {
			vec3 wi_try;
			if (refract_microfacet(-v, h, eta, wi_try)) wi = unit_vector(wi_try);
			else { want_reflect = true; }                      // total internal
		}
		if (want_reflect) wi = unit_vector(reflect(-v, h));

		// Otherwise eval() classifies the sample into the other lobe and returns
		// a density for a direction that lobe never generates.
		if (want_reflect ? dot(wi, n) <= 0.0 : dot(wi, n) >= 0.0)
			return { wi, color(0,0,0), 0.0, false };

		// Plain BSDF + density, as every other material here: a pre-weighted
		// value alongside the pdf makes the integrator apply f*cos/pdf twice.
		color  f; real p;
		if (!eval(v, wi, rec, f, p) || p <= 0.0)
			return { wi, color(0,0,0), 0.0, false };
		return { wi, f, p, false };
	}

private:
	// Both lobes of the microfacet dielectric (Walter et al. 2007).
	bool eval(const vec3& v, const vec3& wi, const hit_record& rec,
			  color& f_out, real& pdf_out) const {
		real alpha = roughness * roughness;
		real eta   = rec.front_face ? (1.0 / ior) : ior;
		const vec3& n = rec.normal;

		real ndotv = dot(n, v);
		real ndotl = dot(n, wi);
		if (ndotv <= 1e-7) return false;

		// Both densities are written out against the same D and the same
		// cosines the BSDF used. Recomputing them from a local-frame half
		// vector loses the cancellation: at low roughness D swings by orders
		// of magnitude over a last-bit change in ndoth.
		if (ndotl > 0.0) {                                  // reflection
			vec3 h = unit_vector(v + wi);
			if (dot(h, n) < 0.0) h = -h;
			real vdoth = clamp(dot(v, h), 0.0, 1.0);
			real ndoth = std::max<real>(dot(n, h), 1e-7);
			real F_val = schlick(vdoth, eta);
			real Dval  = D(ndoth, alpha);
			real Gval  = G2(ndotv, std::max<real>(ndotl, 1e-7), alpha);

			f_out = color(F_val,F_val,F_val)
				  * (Dval * Gval / (4.0 * ndotv * std::max<real>(ndotl, 1e-7)));
			pdf_out = F_val * Dval * G1(ndotv, alpha) / (4.0 * ndotv);
			return true;
		}

		// transmission: inverting refract_microfacet gives h ~ eta*v - wi.
		vec3 h = eta * v - wi;
		if (h.length_squared() < 1e-14) return false;
		h = unit_vector(h);
		if (dot(h, n) < 0.0) h = -h;

		real vdoth = dot(v, h);
		real idoth = -dot(wi, h);
		if (vdoth <= 1e-7 || idoth <= 1e-7) return false;

		real ndoth = std::max<real>(dot(n, h), 1e-7);
		real al    = std::max<real>(-ndotl, 1e-7);
		real F_val = schlick(clamp(vdoth, 0.0, 1.0), eta);
		real Dval  = D(ndoth, alpha);
		real Gval  = G2(ndotv, al, alpha);
		real denom = eta * vdoth + idoth;
		if (std::abs(denom) < 1e-9) return false;

		f_out = tint * ((1.0 - F_val) * Dval * Gval * vdoth * idoth
						/ (ndotv * al * denom * denom));

		pdf_out = (1.0 - F_val) * Dval * G1(ndotv, alpha) * vdoth * idoth
				/ (ndotv * denom * denom);
		return true;
	}

	static real schlick(real cos_theta, real eta) {
		real r0 = (1.0 - eta) / (1.0 + eta); r0 *= r0;
		return r0 + (1.0 - r0) * std::pow(1.0 - cos_theta, 5.0);
	}

	static real D(real ndoth, real a) {
		real a2 = a*a, c2 = ndoth*ndoth;
		real d  = a2*c2 + std::max<real>(1.0 - c2, 0.0);
		return a2 / (pi * d * d);
	}

	static real G2(real ndotv, real ndotl, real a) {
		real a2 = a*a;
		real gv = ndotl * std::sqrt(a2 + (1.0-a2)*ndotv*ndotv);
		real gl = ndotv * std::sqrt(a2 + (1.0-a2)*ndotl*ndotl);
		return 2.0*ndotv*ndotl / (gv + gl + 1e-7);
	}

	static real G1(real ndotv, real a) {
		real a2 = a*a;
		return 2.0*ndotv / (ndotv + std::sqrt(a2 + (1.0-a2)*ndotv*ndotv));
	}

	static bool refract_microfacet(const vec3& v, const vec3& h,
									real eta, vec3& refracted) {
		real cos_i = dot(v, h);
		real sin2_t = eta*eta * (1.0 - cos_i*cos_i);
		if (sin2_t >= 1.0) return false;
		refracted = (eta * cos_i - std::sqrt(1.0 - sin2_t)) * h - eta * v;
		return true;
	}

	static vec3 sample_vndf(const vec3& wo, real a) {
		vec3 vh = unit_vector(vec3(a*wo.x(), a*wo.y(), wo.z()));
		real lensq = vh.x()*vh.x() + vh.y()*vh.y();
		vec3 t1 = lensq > 0.0
				  ? vec3(-vh.y(), vh.x(), 0.0) / std::sqrt(lensq)
				  : vec3(1,0,0);
		vec3 t2 = cross(vh, t1);
		real r  = std::sqrt(random_double());
		real phi = 2.0*pi*random_double();
		real t  = r*std::cos(phi), s = r*std::sin(phi);
		real bz = std::sqrt(std::max<real>(0.0, 1.0-t*t-s*s));
		vec3 nh = t*t1 + s*t2 + bz*vh;
		return unit_vector(vec3(a*nh.x(), a*nh.y(), std::max<real>(0.0, nh.z())));
	}

};

#endif