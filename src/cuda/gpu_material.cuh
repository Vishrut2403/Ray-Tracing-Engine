#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "core/onb.h"
#include "cuda/gpu_scene.cuh"
#include "cuda/gpu_hit.cuh"
#include "cuda/cuda_rand.cuh"

struct GpuBSDFSample {
	vec3   wi;
	vec3   f;
	double pdf;
	bool   is_delta;
};

constexpr double GPU_PI = 3.1415926535897932385;

// Emitted radiance
__device__ inline vec3 gpu_emitted(const GpuMaterial& mat, bool front_face) {
	if (mat.type == MatType::DIFFUSE_LIGHT && front_face) return mat.albedo;
	return vec3(0,0,0);
}

// GGX helpers
__device__ inline double ggx_D(double ndoth, double a) {
	double a2 = a*a, d = ndoth*ndoth*(a2-1.0)+1.0;
	return a2 / (GPU_PI * d * d);
}

__device__ inline double ggx_G1(double ndotv, double a) {
	double a2 = a*a;
	return 2.0*ndotv / (ndotv + sqrt(a2 + (1.0-a2)*ndotv*ndotv));
}

__device__ inline double ggx_G2(double ndotv, double ndotl, double a) {
	double a2 = a*a;
	double gv = ndotl * sqrt(a2 + (1.0-a2)*ndotv*ndotv);
	double gl = ndotv * sqrt(a2 + (1.0-a2)*ndotl*ndotl);
	return 2.0*ndotv*ndotl / (gv + gl + 1e-7);
}

__device__ inline vec3 ggx_F(const vec3& f0, double vdoth) {
	return f0 + (vec3(1,1,1)-f0) * pow(1.0-vdoth, 5.0);
}

// GGX BRDF, without the trailing cosine.
__device__ inline vec3 ggx_brdf(const GpuMaterial& mat,
								 const vec3& v, const vec3& l, const vec3& n) {
	double ndotv = dot(n, v), ndotl = dot(n, l);
	if (ndotv <= 0.0 || ndotl <= 0.0) return vec3(0,0,0);

	double a     = (double)mat.roughness * (double)mat.roughness;
	vec3   h     = unit_vector(v + l);
	double ndoth = fmax(dot(n, h), 1e-7);
	double vdoth = fmax(dot(v, h), 1e-7);

	vec3 f0 = vec3(0.04,0.04,0.04)*(1.0-(double)mat.metallic)
			+ mat.albedo*(double)mat.metallic;

	vec3   F = ggx_F(f0, vdoth);
	double D = ggx_D(ndoth, a);
	double G = ggx_G2(ndotv, ndotl, a);

	vec3 specular = F * (D * G / (4.0 * ndotv * ndotl));
	vec3 diffuse  = mat.albedo / GPU_PI * (1.0-(double)mat.metallic)
				  * (vec3(1,1,1) - F);
	return specular + diffuse;
}

__device__ inline vec3 ggx_sample_vndf(const vec3& wo, double a, curandState* rng) {
	vec3 vh = unit_vector(vec3(a*wo.x(), a*wo.y(), wo.z()));

	double lensq = vh.x()*vh.x() + vh.y()*vh.y();
	vec3 t1 = lensq > 0.0
			  ? vec3(-vh.y(), vh.x(), 0.0) / sqrt(lensq)
			  : vec3(1,0,0);
	vec3 t2 = cross(vh, t1);

	double r   = sqrt(rand_double(rng));
	double phi = 2.0 * GPU_PI * rand_double(rng);
	double t   = r * cos(phi);
	double s   = r * sin(phi);
	double bz  = sqrt(fmax(0.0, 1.0 - t*t - s*s));

	vec3 nh = t*t1 + s*t2 + bz*vh;
	return unit_vector(vec3(a*nh.x(), a*nh.y(), fmax(0.0, nh.z())));
}

__device__ inline double ggx_vndf_pdf(const vec3& wo, const vec3& h, double a) {
	double ndotwo = fmax(wo.z(), 1e-7);
	double hdotwo = fmax(dot(h, wo), 1e-7);
	double ndoth  = fmax(h.z(),  1e-7);
	return ggx_D(ndoth, a) * ggx_G1(ndotwo, a) * hdotwo / (4.0 * ndotwo * hdotwo);
}

// Two lobes: GGX specular + Lambertian. Sampler and pdf must agree, so both
// pick a lobe and report the combined density.
__device__ inline double ggx_spec_prob(const GpuMaterial& mat) {
	vec3 f0 = vec3(0.04,0.04,0.04)*(1.0-(double)mat.metallic)
			+ mat.albedo*(double)mat.metallic;
	double ls = 0.2126*f0.x() + 0.7152*f0.y() + 0.0722*f0.z();
	double ld = (1.0-(double)mat.metallic) * (0.2126*mat.albedo.x()
				+ 0.7152*mat.albedo.y() + 0.0722*mat.albedo.z());
	double t  = ls + ld;
	double p  = (t > 1e-9) ? ls / t : 1.0;
	return fmin(fmax(p, 0.1), 0.9);
}

__device__ inline double ggx_combined_pdf(const GpuMaterial& mat, const vec3& v,
										   const vec3& l, const vec3& n) {
	if (dot(n, v) <= 0.0 || dot(n, l) <= 0.0) return 0.0;
	double a = (double)mat.roughness * (double)mat.roughness;
	onb uvw; uvw.build_from_w(n);
	vec3 v_l(dot(v, uvw.u()), dot(v, uvw.v()), dot(v, uvw.w()));
	vec3 h = unit_vector(v + l);
	vec3 h_l(dot(h, uvw.u()), dot(h, uvw.v()), dot(h, uvw.w()));
	double ps = ggx_spec_prob(mat);
	return ps * ggx_vndf_pdf(v_l, h_l, a) + (1.0 - ps) * dot(n, l) / GPU_PI;
}

// wo points away from the surface.
__device__ inline GpuBSDFSample ggx_sample_lobes(const GpuMaterial& mat,
												  const vec3& wo, const vec3& n,
												  curandState* rng) {
	GpuBSDFSample bs{}; bs.pdf = 0.0;
	double a = (double)mat.roughness * (double)mat.roughness;
	onb uvw; uvw.build_from_w(n);
	vec3 wo_l(dot(wo, uvw.u()), dot(wo, uvw.v()), dot(wo, uvw.w()));
	if (wo_l.z() <= 0.0) return bs;

	vec3 wi;
	if (rand_double(rng) < ggx_spec_prob(mat)) {
		vec3 h_l = ggx_sample_vndf(wo_l, a, rng);
		vec3 h   = unit_vector(uvw.local(h_l));
		wi = unit_vector(reflect(-wo, h));
	} else {
		wi = unit_vector(uvw.local(rand_cosine_direction(rng)));
	}
	if (dot(wi, n) <= 0.0) return bs;

	double p = ggx_combined_pdf(mat, wo, wi, n);
	if (p <= 0.0) return bs;

	bs.wi = wi; bs.f = ggx_brdf(mat, wo, wi, n);
	bs.pdf = p; bs.is_delta = false;
	return bs;
}

// SSS — Jensen dipole as a diffusion BRDF; mirrors the CPU `subsurface` class.
// mat.mfp does not affect shading: Rd is scale-invariant in the exit radius and
// the exit point stays pinned to the entry point.

// Fresnel diffuse reflectance (Egan & Hilgeman approximation)
__device__ inline double sss_Fdr(double eta) {
	if (eta >= 1.0)
		return -0.4399 + 0.7099/eta - 0.3319/(eta*eta) + 0.0636/(eta*eta*eta);
	else
		return -1.4399/(eta*eta) + 0.7099/eta + 0.6681 + 0.0636*eta;
}

// Total diffuse reflectance of the dipole (Jensen et al. 2001).
__device__ inline vec3 sss_Rd_total(const vec3& alpha, double eta) {
	double fd = sss_Fdr(eta);
	double A  = (1.0 + fd) / fmax(1.0 - fd, 1e-6);
	double out[3];
	for (int c = 0; c < 3; ++c) {
		double ap = (c == 0) ? alpha.x() : (c == 1) ? alpha.y() : alpha.z();
		ap = fmin(fmax(ap, 0.0), 0.999);
		double sq = sqrt(3.0 * (1.0 - ap));
		out[c] = 0.5 * ap * (1.0 + exp(-(4.0/3.0) * A * sq)) * exp(-sq);
	}
	return vec3(out[0], out[1], out[2]);
}

// Depends only on the exit direction, so gpu_f (which has no outgoing
// direction) returns exactly what gpu_sample and gpu_f_dir do.
__device__ inline vec3 sss_brdf(const GpuMaterial& mat,
								 const vec3& wi, const vec3& n) {
	double c = dot(n, wi);
	if (c <= 0.0) return vec3(0,0,0);
	double eta = (double)mat.ir;
	double r0  = (1.0 - eta) / (1.0 + eta); r0 *= r0;
	double T_exit  = 1.0 - (r0 + (1.0 - r0) * pow(1.0 - c, 5.0));
	double T_entry = 1.0 - sss_Fdr(eta);
	return sss_Rd_total(mat.albedo, eta) * (T_entry * T_exit / GPU_PI);
}

// Public interface

__device__ inline vec3 gpu_f(const GpuMaterial& mat,
							   const vec3& wi, const vec3& normal) {
	if (mat.type == MatType::LAMBERTIAN) {
		if (dot(normal, wi) <= 0.0) return vec3(0,0,0);
		return mat.albedo / GPU_PI;
	}
	if (mat.type == MatType::GGX) {
		// Cosine-free like LAMBERTIAN. No outgoing direction here, so the view
		// vector is approximated by the normal; gpu_f_dir does it properly.
		if (dot(normal, wi) <= 0.0) return vec3(0,0,0);
		return ggx_brdf(mat, normal, wi, normal);
	}
	if (mat.type == MatType::SSS) {
		return sss_brdf(mat, wi, normal);
	}
	return vec3(0,0,0);
}

__device__ inline double gpu_pdf(const GpuMaterial& mat,
								  const vec3& wi, const vec3& normal) {
	if (mat.type == MatType::LAMBERTIAN) {
		double c = dot(normal, wi);
		return c > 0.0 ? c / GPU_PI : 0.0;
	}
	if (mat.type == MatType::GGX) {
		double c = dot(normal, wi);
		return c > 0.0 ? c / GPU_PI : 0.0;
	}
	if (mat.type == MatType::SSS) {
		double c = dot(normal, wi);
		return c > 0.0 ? c / GPU_PI : 0.0;
	}
	return 0.0;
}

__device__ inline GpuBSDFSample gpu_sample(const GpuMaterial& mat,
											const ray& wo,
											const GpuHitRecord& rec,
											curandState* rng) {
	GpuBSDFSample bs{};

	if (mat.type == MatType::LAMBERTIAN) {
		onb uvw; uvw.build_from_w(rec.normal);
		vec3   wi      = uvw.local(rand_cosine_direction(rng));
		double cosine  = dot(rec.normal, wi);
		double pdf_val = cosine > 0.0 ? cosine / GPU_PI : 0.0;
		return { wi, mat.albedo / GPU_PI, pdf_val, false };
	}

	if (mat.type == MatType::GGX) {
		return ggx_sample_lobes(mat, -unit_vector(wo.direction()),
								rec.normal, rng);
	}

	if (mat.type == MatType::METAL) {
		vec3 r = reflect(unit_vector(wo.direction()), rec.normal);
		r = unit_vector(r + (double)mat.fuzz * rand_in_unit_sphere(rng));
		return { r, mat.albedo, 1.0, true };
	}

	if (mat.type == MatType::DIELECTRIC) {
		double ratio = rec.front_face ? (1.0/(double)mat.ir) : (double)mat.ir;
		vec3   ud    = unit_vector(wo.direction());
		double cos_t = fmin(dot(-ud, rec.normal), 1.0);
		double sin_t = sqrt(1.0 - cos_t*cos_t);
		vec3 dir = (ratio*sin_t > 1.0 || reflectance(cos_t, ratio) > rand_double(rng))
				   ? reflect(ud, rec.normal)
				   : refract(ud, rec.normal, ratio);
		return { dir, vec3(1,1,1), 1.0, true };
	}

	if (mat.type == MatType::SSS) {
		onb uvw; uvw.build_from_w(rec.normal);
		vec3   wi = unit_vector(uvw.local(rand_cosine_direction(rng)));
		double c  = dot(rec.normal, wi);
		if (c <= 0.0) return bs;
		return { wi, sss_brdf(mat, wi, rec.normal), c / GPU_PI, false };
	}

	return bs;
}

// Bidirectional BSDF interface: gpu_f/gpu_pdf above take no outgoing direction,
// which BDPT needs. wo and wi both point away from the surface, f excludes the
// cosine, and specular lobes use f = weight/|cos| with pdf = 1 so the walk's
// f*cos/pdf collapses to the plain reflectance.

__device__ inline vec3 gpu_f_dir(const GpuMaterial& mat, const vec3& wo,
								  const vec3& wi, const vec3& n) {
	switch (mat.type) {
		case MatType::LAMBERTIAN:
			if (dot(n, wi) <= 0.0 || dot(n, wo) <= 0.0) return vec3(0,0,0);
			return mat.albedo / GPU_PI;
		case MatType::SSS:
			if (dot(n, wo) <= 0.0) return vec3(0,0,0);
			return sss_brdf(mat, wi, n);
		case MatType::GGX:
			return ggx_brdf(mat, wo, wi, n);
		default:
			return vec3(0,0,0);
	}
}

__device__ inline double gpu_pdf_dir(const GpuMaterial& mat, const vec3& wo,
									  const vec3& wi, const vec3& n) {
	switch (mat.type) {
		case MatType::LAMBERTIAN:
		case MatType::SSS: {
			double c = dot(n, wi);
			if (c <= 0.0 || dot(n, wo) <= 0.0) return 0.0;
			return c / GPU_PI;
		}
		case MatType::GGX:
			return ggx_combined_pdf(mat, wo, wi, n);
		default:
			return 0.0;
	}
}

__device__ inline GpuBSDFSample gpu_sample_dir(const GpuMaterial& mat,
												const vec3& wo,
												const GpuHitRecord& rec,
												curandState* rng) {
	GpuBSDFSample bs{};
	bs.pdf = 0.0;
	const vec3& n = rec.normal;

	switch (mat.type) {
		case MatType::LAMBERTIAN:
		case MatType::SSS: {
			if (dot(n, wo) <= 0.0) return bs;
			onb uvw; uvw.build_from_w(n);
			vec3   wi = unit_vector(uvw.local(rand_cosine_direction(rng)));
			double c  = dot(n, wi);
			if (c <= 0.0) return bs;
			bs.wi = wi;
			bs.f  = (mat.type == MatType::SSS) ? sss_brdf(mat, wi, n)
											   : mat.albedo / GPU_PI;
			bs.pdf = c / GPU_PI; bs.is_delta = false;
			return bs;
		}
		case MatType::GGX:
			return ggx_sample_lobes(mat, wo, n, rng);
		case MatType::METAL: {
			vec3 wi = unit_vector(reflect(-wo, n)
								  + (double)mat.fuzz * rand_in_unit_sphere(rng));
			double c = dot(wi, n);
			if (c <= 0.0) return bs;
			bs.wi = wi; bs.f = mat.albedo / fmax(fabs(c), 1e-9);
			bs.pdf = 1.0; bs.is_delta = true;
			return bs;
		}
		case MatType::DIELECTRIC: {
			double ratio = rec.front_face ? (1.0/(double)mat.ir) : (double)mat.ir;
			vec3   ud    = -wo;
			double cos_t = fmin(dot(wo, n), 1.0);
			double sin_t = sqrt(fmax(0.0, 1.0 - cos_t*cos_t));
			vec3   dir   = (ratio*sin_t > 1.0
							|| reflectance(cos_t, ratio) > rand_double(rng))
						   ? reflect(ud, n)
						   : refract(ud, n, ratio);
			dir = unit_vector(dir);
			bs.wi = dir;
			bs.f  = vec3(1,1,1) / fmax(fabs(dot(dir, n)), 1e-9);
			bs.pdf = 1.0; bs.is_delta = true;
			return bs;
		}
		default:
			return bs;   // DIFFUSE_LIGHT and anything unhandled
	}
}