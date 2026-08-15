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

class material {
public:
	virtual ~material() = default;

	virtual color emitted(
		const ray&, const hit_record&,
		double, double, const point3&
	) const { return color(0, 0, 0); }

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const = 0;
	virtual color      f  (const ray& wo, const vec3& wi, const hit_record& rec) const = 0;
	virtual double     pdf(const ray& wo, const vec3& wi, const hit_record& rec) const = 0;

	// Bidirectional interface for BDPT: wo and wi both point away from the
	// surface, f excludes the cosine, and delta lobes use f = weight/|cos| with
	// pdf = 1. Default is a non-scattering material.
	virtual color f_dir(const vec3&, const vec3&, const hit_record&) const {
		return color(0,0,0);
	}
	virtual double pdf_dir(const vec3&, const vec3&, const hit_record&) const {
		return 0.0;
	}
	virtual BSDFSample sample_dir(const vec3&, const hit_record&) const {
		return { vec3(0,0,0), color(0,0,0), 0.0, false };
	}
};

class lambertian : public material {
public:
	std::shared_ptr<texture> albedo;

	lambertian(const color& a) : albedo(std::make_shared<solid_color>(a)) {}
	lambertian(std::shared_ptr<texture> a) : albedo(a) {}

	virtual BSDFSample sample(const ray&, const hit_record& rec) const override {
		onb uvw; uvw.build_from_w(rec.normal);
		vec3   wi      = uvw.local(random_cosine_direction());
		double cosine  = dot(rec.normal, wi);
		double pdf_val = cosine > 0.0 ? cosine / pi : 0.0;
		return { wi, albedo->value(rec.u, rec.v, rec.p) / pi, pdf_val, false };
	}

	virtual color f(const ray&, const vec3& wi, const hit_record& rec) const override {
		if (dot(rec.normal, wi) <= 0.0) return color(0, 0, 0);
		return albedo->value(rec.u, rec.v, rec.p) / pi;
	}

	virtual double pdf(const ray&, const vec3& wi, const hit_record& rec) const override {
		double c = dot(rec.normal, wi);
		return c > 0.0 ? c / pi : 0.0;
	}

	virtual color f_dir(const vec3& wo, const vec3& wi,
						const hit_record& rec) const override {
		if (dot(rec.normal, wi) <= 0.0 || dot(rec.normal, wo) <= 0.0)
			return color(0,0,0);
		return albedo->value(rec.u, rec.v, rec.p) / pi;
	}
	virtual double pdf_dir(const vec3& wo, const vec3& wi,
						   const hit_record& rec) const override {
		double c = dot(rec.normal, wi);
		if (c <= 0.0 || dot(rec.normal, wo) <= 0.0) return 0.0;
		return c / pi;
	}
	virtual BSDFSample sample_dir(const vec3& wo,
								  const hit_record& rec) const override {
		if (dot(rec.normal, wo) <= 0.0) return { vec3(0,0,0), color(0,0,0), 0.0, false };
		onb uvw; uvw.build_from_w(rec.normal);
		vec3   wi = unit_vector(uvw.local(random_cosine_direction()));
		double c  = dot(rec.normal, wi);
		if (c <= 0.0) return { wi, color(0,0,0), 0.0, false };
		return { wi, albedo->value(rec.u, rec.v, rec.p) / pi, c / pi, false };
	}
};

class metal : public material {
public:
	color  albedo;
	double fuzz;

	metal(const color& a, double f) : albedo(a), fuzz(f < 1.0 ? f : 1.0) {}

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		vec3 r = unit_vector(reflect(unit_vector(wo.direction()), rec.normal)
							 + fuzz * random_in_unit_sphere());
		return { r, albedo, 1.0, true };
	}

	virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
	virtual double pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }

	virtual BSDFSample sample_dir(const vec3& wo,
								  const hit_record& rec) const override {
		vec3   wi = unit_vector(reflect(-wo, rec.normal)
								+ fuzz * random_in_unit_sphere());
		double c  = dot(wi, rec.normal);
		if (c <= 0.0) return { wi, color(0,0,0), 0.0, false };
		return { wi, albedo / std::max(std::abs(c), 1e-9), 1.0, true };
	}
};

class dielectric : public material {
public:
	double ir;

	dielectric(double index_of_refraction) : ir(index_of_refraction) {}

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		vec3   ud    = unit_vector(wo.direction());
		double ratio = rec.front_face ? (1.0 / ir) : ir;
		double cos_t = fmin(dot(-ud, rec.normal), 1.0);
		double sin_t = std::sqrt(1.0 - cos_t * cos_t);
		vec3 dir = (ratio * sin_t > 1.0 || schlick(cos_t, ratio) > random_double())
				   ? reflect(ud, rec.normal)
				   : refract(ud, rec.normal, ratio);
		return { dir, color(1, 1, 1), 1.0, true };
	}

	virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
	virtual double pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }

	virtual BSDFSample sample_dir(const vec3& wo,
								  const hit_record& rec) const override {
		double ratio = rec.front_face ? (1.0 / ir) : ir;
		vec3   ud    = -wo;
		double cos_t = fmin(dot(wo, rec.normal), 1.0);
		double sin_t = std::sqrt(std::max(0.0, 1.0 - cos_t * cos_t));
		vec3   dir   = (ratio * sin_t > 1.0 || schlick(cos_t, ratio) > random_double())
					   ? reflect(ud, rec.normal)
					   : refract(ud, rec.normal, ratio);
		dir = unit_vector(dir);
		return { dir, color(1,1,1) / std::max(std::abs(dot(dir, rec.normal)), 1e-9),
				 1.0, true };
	}

private:
	static double schlick(double cos, double ri) {
		double r0 = (1.0 - ri) / (1.0 + ri); r0 *= r0;
		return r0 + (1.0 - r0) * std::pow(1.0 - cos, 5.0);
	}
};

class ggx : public material {
public:
	color  base_color;
	double roughness;
	double metallic;

	ggx(const color& base, double r, double m = 0.0)
		: base_color(base),
		  roughness(clamp(r, 0.001, 1.0)),
		  metallic (clamp(m, 0.0,   1.0)) {}

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		return sample_dir(-unit_vector(wo.direction()), rec);
	}

	// Without the cosine: the integrators apply cos(theta) themselves.
	virtual color f(const ray& wo, const vec3& wi, const hit_record& rec) const override {
		if (dot(rec.normal, wi) <= 0.0) return color(0,0,0);
		return brdf(-unit_vector(wo.direction()), wi, rec.normal);
	}

	virtual double pdf(const ray& wo, const vec3& wi, const hit_record& rec) const override {
		return combined_pdf(-unit_vector(wo.direction()), wi, rec.normal);
	}

	virtual color f_dir(const vec3& wo, const vec3& wi,
						const hit_record& rec) const override {
		return brdf(wo, wi, rec.normal);
	}
	virtual double pdf_dir(const vec3& wo, const vec3& wi,
						   const hit_record& rec) const override {
		return combined_pdf(wo, wi, rec.normal);
	}
	virtual BSDFSample sample_dir(const vec3& wo,
								  const hit_record& rec) const override {
		const vec3& n = rec.normal;
		double alpha = roughness * roughness;
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

		double p = combined_pdf(wo, wi, n);
		if (p <= 0.0) return { wi, color(0,0,0), 0.0, false };
		return { wi, brdf(wo, wi, n), p, false };
	}

private:
	static double D(double ndoth, double a) {
		double a2 = a*a, d = ndoth*ndoth*(a2-1.0)+1.0;
		return a2 / (pi * d * d);
	}

	static double G2(double ndotv, double ndotl, double a) {
		double a2 = a*a;
		double gv = ndotl * std::sqrt(a2 + (1.0-a2)*ndotv*ndotv);
		double gl = ndotv * std::sqrt(a2 + (1.0-a2)*ndotl*ndotl);
		return 2.0*ndotv*ndotl / (gv + gl + 1e-7);
	}

	static double G1(double ndotv, double a) {
		double a2 = a*a;
		return 2.0*ndotv / (ndotv + std::sqrt(a2 + (1.0-a2)*ndotv*ndotv));
	}

	color F(double vdoth) const {
		color f0 = color(0.04,0.04,0.04)*(1.0-metallic) + base_color*metallic;
		return f0 + (color(1,1,1)-f0) * std::pow(1.0-vdoth, 5.0);
	}

	// Two lobes: GGX specular + Lambertian. Sampler and pdf must agree, so both
	// pick a lobe and report the combined density.
	double spec_prob() const {
		color  f0 = color(0.04,0.04,0.04)*(1.0-metallic) + base_color*metallic;
		double ls = 0.2126*f0.x() + 0.7152*f0.y() + 0.0722*f0.z();
		double ld = (1.0-metallic) * (0.2126*base_color.x()
					+ 0.7152*base_color.y() + 0.0722*base_color.z());
		double t  = ls + ld;
		return clamp(t > 1e-9 ? ls / t : 1.0, 0.1, 0.9);
	}

	double combined_pdf(const vec3& v, const vec3& l, const vec3& n) const {
		if (dot(n, v) <= 0.0 || dot(n, l) <= 0.0) return 0.0;
		double alpha = roughness * roughness;
		onb uvw; uvw.build_from_w(n);
		vec3 v_l(dot(v, uvw.u()), dot(v, uvw.v()), dot(v, uvw.w()));
		vec3 h = unit_vector(v + l);
		vec3 h_l(dot(h, uvw.u()), dot(h, uvw.v()), dot(h, uvw.w()));
		double ps = spec_prob();
		return ps * vndf_pdf(v_l, h_l, alpha) + (1.0 - ps) * dot(n, l) / pi;
	}

	// True BSDF, without the trailing cosine.
	color brdf(const vec3& v, const vec3& l, const vec3& n) const {
		double ndotv = dot(n,v), ndotl = dot(n,l);
		if (ndotv <= 0.0 || ndotl <= 0.0) return color(0,0,0);

		double a     = roughness * roughness;
		vec3   h     = unit_vector(v + l);
		double ndoth = std::max(dot(n,h), 1e-7);
		double vdoth = std::max(dot(v,h), 1e-7);

		color  Fval = F(vdoth);
		double Dval = D(ndoth, a);
		double Gval = G2(ndotv, ndotl, a);

		color specular = Fval * (Dval * Gval / (4.0 * ndotv * ndotl));
		color diffuse  = base_color / pi * (1.0 - metallic) * (color(1,1,1) - Fval);
		return specular + diffuse;
	}

	static vec3 sample_vndf(const vec3& wo, double a) {
		vec3 vh = unit_vector(vec3(a*wo.x(), a*wo.y(), wo.z()));

		double lensq = vh.x()*vh.x() + vh.y()*vh.y();
		vec3 t1 = lensq > 0.0
				  ? vec3(-vh.y(), vh.x(), 0.0) / std::sqrt(lensq)
				  : vec3(1,0,0);
		vec3 t2 = cross(vh, t1);

		double r  = std::sqrt(random_double());
		double phi = 2.0 * pi * random_double();
		double t  = r * std::cos(phi);
		double s  = r * std::sin(phi);
		double bz = std::sqrt(std::max(0.0, 1.0 - t*t - s*s));

		vec3 nh = t*t1 + s*t2 + bz*vh;
		return unit_vector(vec3(a*nh.x(), a*nh.y(), std::max(0.0, nh.z())));
	}

	static double vndf_pdf(const vec3& wo, const vec3& h, double a) {
		double ndotwo = std::max(wo.z(), 1e-7);
		double hdotwo = std::max(dot(h, wo), 1e-7);
		double ndoth  = std::max(h.z(), 1e-7);
		return D(ndoth, a) * G1(ndotwo, a) * hdotwo / (4.0*ndotwo*hdotwo);
	}
};

class diffuse_light : public material {
public:
	std::shared_ptr<texture> emit;

	diffuse_light(std::shared_ptr<texture> a) : emit(a) {}
	diffuse_light(color c) : emit(std::make_shared<solid_color>(c)) {}

	virtual color emitted(const ray&, const hit_record& rec,
						  double u, double v, const point3& p) const override {
		if (!rec.front_face) return color(0,0,0);
		return emit->value(u, v, p);
	}

	virtual BSDFSample sample(const ray&, const hit_record&) const override {
		return { vec3(0,0,0), color(0,0,0), 0.0, false };
	}
	virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
	virtual double pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }
};

class isotropic : public material {
public:
	std::shared_ptr<texture> albedo;

	isotropic(const color& c) : albedo(std::make_shared<solid_color>(c)) {}
	isotropic(std::shared_ptr<texture> a) : albedo(a) {}

	virtual BSDFSample sample(const ray&, const hit_record& rec) const override {
		vec3 wi  = random_unit_vector();
		color rho = albedo->value(rec.u, rec.v, rec.p);
		return { wi, rho / (4.0*pi), 1.0/(4.0*pi), false };
	}

	virtual color f(const ray&, const vec3&, const hit_record& rec) const override {
		return albedo->value(rec.u, rec.v, rec.p) / (4.0 * pi);
	}

	virtual double pdf(const ray&, const vec3&, const hit_record&) const override {
		return 1.0 / (4.0 * pi);
	}

	// Phase function: spherical, independent of both directions.
	virtual color f_dir(const vec3&, const vec3&,
						const hit_record& rec) const override {
		return albedo->value(rec.u, rec.v, rec.p) / (4.0 * pi);
	}
	virtual double pdf_dir(const vec3&, const vec3&,
						   const hit_record&) const override {
		return 1.0 / (4.0 * pi);
	}
	virtual BSDFSample sample_dir(const vec3&,
								  const hit_record& rec) const override {
		vec3 wi = random_unit_vector();
		return { wi, albedo->value(rec.u, rec.v, rec.p) / (4.0*pi),
				 1.0/(4.0*pi), false };
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
	double mean_free_path;
	double ior;

	subsurface(const color& alb, double mfp, double ior_ = 1.4)
		: albedo_color(alb),
		  mean_free_path(std::max(mfp, 1e-4)),
		  ior(ior_),
		  Rd(total_diffuse_reflectance(alb, ior_)),
		  T_entry(1.0 - fdr(ior_)) {}

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		return sample_dir(-unit_vector(wo.direction()), rec);
	}

	virtual color f(const ray&, const vec3& wi, const hit_record& rec) const override {
		return brdf(wi, rec.normal);
	}

	virtual double pdf(const ray&, const vec3& wi, const hit_record& rec) const override {
		double c = dot(rec.normal, wi);
		return c > 0.0 ? c / pi : 0.0;
	}

	virtual color f_dir(const vec3&, const vec3& wi,
						const hit_record& rec) const override {
		return brdf(wi, rec.normal);
	}
	virtual double pdf_dir(const vec3&, const vec3& wi,
						   const hit_record& rec) const override {
		double c = dot(rec.normal, wi);
		return c > 0.0 ? c / pi : 0.0;
	}
	virtual BSDFSample sample_dir(const vec3&,
								  const hit_record& rec) const override {
		onb uvw; uvw.build_from_w(rec.normal);
		vec3   wi = unit_vector(uvw.local(random_cosine_direction()));
		double c  = dot(rec.normal, wi);
		if (c <= 0.0) return { wi, color(0,0,0), 0.0, false };
		return { wi, brdf(wi, rec.normal), c / pi, false };
	}

private:
	color  Rd;       // total diffuse reflectance of the dipole, per channel
	double T_entry;  // hemispherical Fresnel transmittance on the way in

	// Entry uses the hemispherical average so f() needs only one direction.
	color brdf(const vec3& wi, const vec3& n) const {
		double c = dot(n, wi);
		if (c <= 0.0) return color(0,0,0);
		double r0 = (1.0 - ior) / (1.0 + ior); r0 *= r0;
		double T_exit = 1.0 - (r0 + (1.0 - r0) * std::pow(1.0 - c, 5.0));
		return Rd * (T_entry * T_exit / pi);
	}

	// Fresnel diffuse reflectance (Egan & Hilgeman approximation)
	static double fdr(double eta) {
		if (eta >= 1.0)
			return -0.4399 + 0.7099/eta - 0.3319/(eta*eta) + 0.0636/(eta*eta*eta);
		else
			return -1.4399/(eta*eta) + 0.7099/eta + 0.6681 + 0.0636*eta;
	}

	static color total_diffuse_reflectance(const color& alpha, double eta) {
		double fd = fdr(eta);
		double A  = (1.0 + fd) / std::max(1.0 - fd, 1e-6);
		auto ch = [&](double ap) {
			ap = clamp(ap, 0.0, 0.999);
			double s = std::sqrt(3.0 * (1.0 - ap));
			return 0.5 * ap * (1.0 + std::exp(-(4.0/3.0) * A * s)) * std::exp(-s);
		};
		return color(ch(alpha.x()), ch(alpha.y()), ch(alpha.z()));
	}
};

class rough_dielectric : public material {
public:
	color  tint;
	double roughness;
	double ior;

	rough_dielectric(const color& t, double r, double ior = 1.5)
		: tint(t),
		  roughness(clamp(r, 0.001, 1.0)),
		  ior(ior) {}

	virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
		return sample_dir(-unit_vector(wo.direction()), rec);
	}

	virtual color f(const ray& wo, const vec3& wi,
					const hit_record& rec) const override {
		return f_dir(-unit_vector(wo.direction()), wi, rec);
	}

	virtual double pdf(const ray& wo, const vec3& wi,
					   const hit_record& rec) const override {
		return pdf_dir(-unit_vector(wo.direction()), wi, rec);
	}

	virtual color f_dir(const vec3& wo, const vec3& wi,
						const hit_record& rec) const override {
		color f; double p;
		return eval(wo, wi, rec, f, p) ? f : color(0,0,0);
	}

	virtual double pdf_dir(const vec3& wo, const vec3& wi,
						   const hit_record& rec) const override {
		color f; double p;
		return eval(wo, wi, rec, f, p) ? p : 0.0;
	}

	virtual BSDFSample sample_dir(const vec3& v,
								  const hit_record& rec) const override {
		double alpha = roughness * roughness;
		double eta   = rec.front_face ? (1.0 / ior) : ior;
		const vec3& n = rec.normal;
		if (dot(n, v) <= 0.0) return { vec3(0,0,0), color(0,0,0), 0.0, false };

		onb uvw; uvw.build_from_w(n);
		vec3 v_l(dot(v, uvw.u()), dot(v, uvw.v()), dot(v, uvw.w()));
		vec3 h_l = sample_vndf(v_l, alpha);
		vec3 h   = unit_vector(uvw.local(h_l));
		if (dot(h, v) < 0.0) h = -h;

		double vdoth = clamp(dot(v, h), 0.0, 1.0);
		double F_val = schlick(vdoth, eta);

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
		color  f; double p;
		if (!eval(v, wi, rec, f, p) || p <= 0.0)
			return { wi, color(0,0,0), 0.0, false };
		return { wi, f, p, false };
	}

private:
	// Both lobes of the microfacet dielectric (Walter et al. 2007).
	bool eval(const vec3& v, const vec3& wi, const hit_record& rec,
			  color& f_out, double& pdf_out) const {
		double alpha = roughness * roughness;
		double eta   = rec.front_face ? (1.0 / ior) : ior;
		const vec3& n = rec.normal;

		double ndotv = dot(n, v);
		double ndotl = dot(n, wi);
		if (ndotv <= 1e-7) return false;

		onb uvw; uvw.build_from_w(n);
		vec3 v_l(dot(v, uvw.u()), dot(v, uvw.v()), dot(v, uvw.w()));

		if (ndotl > 0.0) {                                  // reflection
			vec3 h = unit_vector(v + wi);
			if (dot(h, n) < 0.0) h = -h;
			double vdoth = clamp(dot(v, h), 0.0, 1.0);
			double ndoth = std::max(dot(n, h), 1e-7);
			double F_val = schlick(vdoth, eta);
			double Dval  = D(ndoth, alpha);
			double Gval  = G2(ndotv, std::max(ndotl, 1e-7), alpha);

			f_out = color(F_val,F_val,F_val)
				  * (Dval * Gval / (4.0 * ndotv * std::max(ndotl, 1e-7)));
			vec3 h_l(dot(h, uvw.u()), dot(h, uvw.v()), dot(h, uvw.w()));
			pdf_out = F_val * vndf_pdf(v_l, h_l, alpha);
			return true;
		}

		// transmission: inverting refract_microfacet gives h ~ eta*v - wi.
		vec3 h = eta * v - wi;
		if (h.length_squared() < 1e-14) return false;
		h = unit_vector(h);
		if (dot(h, n) < 0.0) h = -h;

		double vdoth = dot(v, h);
		double idoth = -dot(wi, h);
		if (vdoth <= 1e-7 || idoth <= 1e-7) return false;

		double ndoth = std::max(dot(n, h), 1e-7);
		double al    = std::max(-ndotl, 1e-7);
		double F_val = schlick(clamp(vdoth, 0.0, 1.0), eta);
		double Dval  = D(ndoth, alpha);
		double Gval  = G2(ndotv, al, alpha);
		double denom = eta * vdoth + idoth;
		if (std::abs(denom) < 1e-9) return false;

		f_out = tint * ((1.0 - F_val) * Dval * Gval * vdoth * idoth
						/ (ndotv * al * denom * denom));

		vec3 h_l(dot(h, uvw.u()), dot(h, uvw.v()), dot(h, uvw.w()));
		pdf_out = (1.0 - F_val)
				* vndf_pdf_transmission(v_l, h_l, wi, h, eta, alpha);
		return true;
	}

	static double schlick(double cos_theta, double eta) {
		double r0 = (1.0 - eta) / (1.0 + eta); r0 *= r0;
		return r0 + (1.0 - r0) * std::pow(1.0 - cos_theta, 5.0);
	}

	static double D(double ndoth, double a) {
		double a2 = a*a, d = ndoth*ndoth*(a2-1.0)+1.0;
		return a2 / (pi * d * d);
	}

	static double G2(double ndotv, double ndotl, double a) {
		double a2 = a*a;
		double gv = ndotl * std::sqrt(a2 + (1.0-a2)*ndotv*ndotv);
		double gl = ndotv * std::sqrt(a2 + (1.0-a2)*ndotl*ndotl);
		return 2.0*ndotv*ndotl / (gv + gl + 1e-7);
	}

	static double G1(double ndotv, double a) {
		double a2 = a*a;
		return 2.0*ndotv / (ndotv + std::sqrt(a2 + (1.0-a2)*ndotv*ndotv));
	}

	static bool refract_microfacet(const vec3& v, const vec3& h,
									double eta, vec3& refracted) {
		double cos_i = dot(v, h);
		double sin2_t = eta*eta * (1.0 - cos_i*cos_i);
		if (sin2_t >= 1.0) return false;
		refracted = (eta * cos_i - std::sqrt(1.0 - sin2_t)) * h - eta * v;
		return true;
	}

	static vec3 sample_vndf(const vec3& wo, double a) {
		vec3 vh = unit_vector(vec3(a*wo.x(), a*wo.y(), wo.z()));
		double lensq = vh.x()*vh.x() + vh.y()*vh.y();
		vec3 t1 = lensq > 0.0
				  ? vec3(-vh.y(), vh.x(), 0.0) / std::sqrt(lensq)
				  : vec3(1,0,0);
		vec3 t2 = cross(vh, t1);
		double r  = std::sqrt(random_double());
		double phi = 2.0*pi*random_double();
		double t  = r*std::cos(phi), s = r*std::sin(phi);
		double bz = std::sqrt(std::max(0.0, 1.0-t*t-s*s));
		vec3 nh = t*t1 + s*t2 + bz*vh;
		return unit_vector(vec3(a*nh.x(), a*nh.y(), std::max(0.0, nh.z())));
	}

	static double vndf_pdf(const vec3& wo, const vec3& h, double a) {
		double ndotwo = std::max(wo.z(), 1e-7);
		double hdotwo = std::max(dot(h, wo), 1e-7);
		double ndoth  = std::max(h.z(), 1e-7);
		return D(ndoth, a) * G1(ndotwo, a) * hdotwo / (4.0*ndotwo*hdotwo);
	}

	static double vndf_pdf_transmission(const vec3& v_l, const vec3& h_l,
										 const vec3& wi, const vec3& h,
										 double eta, double a) {
		double vdoth = std::max(dot(vec3(v_l.x(),v_l.y(),v_l.z()), h_l), 1e-7);
		double idoth = std::max(-dot(wi, h), 1e-7);
		double denom = eta * vdoth + idoth;
		double jacobian = idoth / (denom*denom);
		double ndoth = std::max(h_l.z(), 1e-7);
		double ndov  = std::max(v_l.z(), 1e-7);
		return D(ndoth, a) * G1(ndov, a) * vdoth / ndov * jacobian;
	}
};

#endif