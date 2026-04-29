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

__device__ inline vec3 ggx_eval(const GpuMaterial& mat,
								 const vec3& v, const vec3& l, const vec3& n) {
	if (dot(n, v) <= 0.0 || dot(n, l) <= 0.0) return vec3(0,0,0);

	double a     = (double)mat.roughness * (double)mat.roughness;
	vec3   h     = unit_vector(v + l);
	double ndotv = fmax(dot(n,v), 1e-7);
	double ndotl = fmax(dot(n,l), 1e-7);
	double ndoth = fmax(dot(n,h), 1e-7);
	double vdoth = fmax(dot(v,h), 1e-7);

	vec3 f0    = vec3(0.04,0.04,0.04)*(1.0-(double)mat.metallic)
			   + mat.albedo*(double)mat.metallic;
	vec3   Fval   = ggx_F(f0, vdoth);
	double Dval   = ggx_D(ndoth, a);
	double Gval   = ggx_G2(ndotv, ndotl, a);

	vec3 specular = Fval * (Dval * Gval / (4.0 * ndotv * ndotl));
	vec3 diffuse  = mat.albedo / GPU_PI * (1.0-(double)mat.metallic)
				  * (vec3(1,1,1)-Fval);

	return (specular + diffuse) * ndotl;
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

// SSS — Jensen Dipole
//
// Model: light enters at a surface point, scatters inside the volume,
// and exits at a nearby point sampled from the dipole diffusion profile.
//
// Parameters stored in GpuMaterial:
//   albedo  — single-scattering albedo (per channel)
//   mfp     — mean free path in scene units
//   ir      — IOR at the air/medium boundary (used for Fresnel at entry/exit)
//
// The dipole diffusion profile for a reduced scattering coefficient sigma_r:
//   R(r) = alpha' / (4*pi) * [ exp(-sigma_r*d_r) / d_r^2 * (sigma_r + 1/d_r)
//                             + exp(-sigma_r*d_v) / d_v^2 * (sigma_r + 1/d_v) ]
// where d_r = sqrt(r^2 + z_r^2), d_v = sqrt(r^2 + z_v^2)
// z_r = 1/sigma_t, z_v = z_r * (1 + 4/3 * A)
// A   = (1 + F_dr) / (1 - F_dr),  F_dr = -1.44/eta^2 + 0.71/eta + 0.668 + 0.0636*eta
//
// Sampling: sample r from a 1D exponential in r, then uniform phi.
// The exit point is x_o = x_i + r*(tangent direction).
// Exit direction is cosine-weighted on the outward hemisphere at x_o.

// Fresnel diffuse reflectance approximation (Egan & Hilgeman)
__device__ inline double sss_Fdr(double eta) {
	if (eta >= 1.0)
		return -0.4399 + 0.7099/eta - 0.3319/(eta*eta) + 0.0636/(eta*eta*eta);
	else
		return -1.4399/(eta*eta) + 0.7099/eta + 0.6681 + 0.0636*eta;
}

// Dipole profile R(r) — returns per-channel exitant radiance for unit incident flux
__device__ inline vec3 sss_dipole_R(float r, const vec3& sigma_t_r,
									 const vec3& z_r, const vec3& z_v) {
	// d_r = sqrt(r^2 + z_r^2),  d_v = sqrt(r^2 + z_v^2)
	double r2 = (double)r * (double)r;

	vec3 result(0,0,0);
	for (int c = 0; c < 3; ++c) {
		double sr  = (c==0) ? sigma_t_r.x() : (c==1) ? sigma_t_r.y() : sigma_t_r.z();
		double zr  = (c==0) ? z_r.x()       : (c==1) ? z_r.y()       : z_r.z();
		double zv  = (c==0) ? z_v.x()       : (c==1) ? z_v.y()       : z_v.z();

		double dr  = sqrt(r2 + zr*zr);
		double dv  = sqrt(r2 + zv*zv);

		double Rr  = exp(-sr*dr) / (dr*dr) * (sr + 1.0/dr);
		double Rv  = exp(-sr*dv) / (dv*dv) * (sr + 1.0/dv);

		double val = (Rr - Rv) / (4.0 * GPU_PI);
		val = fmax(0.0, val);

		if (c==0)      result = vec3(val, result.y(), result.z());
		else if (c==1) result = vec3(result.x(), val, result.z());
		else           result = vec3(result.x(), result.y(), val);
	}
	return result;
}

// Sample exit radius from exponential profile p(r) ∝ exp(-r / mfp) * r
// using the inversion method on the CDF of r*exp(-r/mfp)
__device__ inline float sss_sample_radius(float mfp, curandState* rng) {
	// CDF of p(r) = (1/mfp^2) * r * exp(-r/mfp) is analytically invertible
	// but we use the simpler exponential importance sample: r = -mfp * log(xi)
	// then accept/reject weighting handles the profile shape.
	// For simplicity and to avoid complex inversion, sample r = -mfp*log(u1)
	// and weight by r (the actual profile weighting is absorbed into the MIS).
	// This matches d'Eon & Irving 2011's approach for a single-lobe profile.
	double u1 = fmax(rand_double(rng), 1e-7);
	double u2 = fmax(rand_double(rng), 1e-7);
	return (float)(-(double)mfp * (log(u1) + log(u2)));
}

__device__ inline GpuBSDFSample gpu_sample_sss(
	const GpuMaterial& mat,
	const ray& wo,
	const GpuHitRecord& rec,
	const GpuHittable* hittables, int n_hittables,
	const GpuTriangle* tris, const GpuTriBVHNode* tri_bvh,
	int tri_root, int n_tris,
	curandState* rng)
{
	GpuBSDFSample bs{};

	float  mfp = mat.mfp;
	double eta = (double)mat.ir;

	// Dipole parameters
	// Reduced scattering: sigma_s' = sigma_s * (1-g), g=0 for isotropic
	// We parameterise directly from mfp: sigma_t = 1/mfp per channel
	// For coloured materials, scale per channel by albedo luminance
	double inv_mfp = 1.0 / (double)mfp;
	vec3   sigma_t = vec3(inv_mfp, inv_mfp, inv_mfp);

	// Scale sigma_t per channel by inverse albedo so brighter channels scatter further
	// Clamp to avoid division by zero on zero-albedo channels
	vec3 alb = mat.albedo;
	sigma_t = vec3(
		sigma_t.x() / fmax((double)alb.x(), 0.01),
		sigma_t.y() / fmax((double)alb.y(), 0.01),
		sigma_t.z() / fmax((double)alb.z(), 0.01)
	);

	// Fresnel diffuse reflectance
	double Fdr = sss_Fdr(eta);
	double A   = (1.0 + Fdr) / fmax(1.0 - Fdr, 1e-6);

	// Dipole source depths
	vec3 z_r = vec3(1.0/sigma_t.x(), 1.0/sigma_t.y(), 1.0/sigma_t.z());
	vec3 z_v = z_r * (1.0 + 4.0/3.0 * A);

	// 
	// Build tangent frame at the entry point
	onb uvw; uvw.build_from_w(rec.normal);

	// Sample exit radius from exponential (importance sampling the diffusion profile)
	float r = sss_sample_radius(mfp, rng);

	// Sample uniformly on a disk of radius r in the tangent plane
	double phi = 2.0 * GPU_PI * rand_double(rng);
	vec3 offset = uvw.u() * (r * cos(phi)) + uvw.v() * (r * sin(phi));

	// Exit point: project along the normal to stay near the surface
	// We shoot a ray from above (entry point + offset + normal*epsilon) downward
	vec3 x_exit = rec.p + offset;

	// Evaluate the dipole profile at radius r
	vec3 Rd = sss_dipole_R(r, sigma_t, z_r, z_v);

	// Scale by albedo — the profile gives the reduced scattering contribution,
	// albedo modulates it per channel
	Rd = Rd * alb;

	// Exit direction: cosine-weighted on outward hemisphere
	vec3 wi = uvw.local(rand_cosine_direction(rng));
	double cos_out = fmax(dot(rec.normal, wi), 0.0);
	if (cos_out <= 0.0) return bs;

	// 
	// Entry Fresnel (air -> medium): T_i = 1 - F(cos_i)
	vec3  ud      = unit_vector(wo.direction());
	double cos_in = fmax(dot(-ud, rec.normal), 0.0);
	// Schlick approximation
	double r0     = (1.0 - eta) / (1.0 + eta); r0 *= r0;
	double F_in   = r0 + (1.0-r0) * pow(1.0-cos_in, 5.0);
	double T_in   = 1.0 - F_in;

	// Exit Fresnel (medium -> air): T_o = 1 - F(cos_out, 1/eta)
	double eta_inv = 1.0 / eta;
	double r0_out  = (1.0 - eta_inv) / (1.0 + eta_inv); r0_out *= r0_out;
	double F_out   = r0_out + (1.0-r0_out) * pow(1.0-cos_out, 5.0);
	double T_out   = 1.0 - F_out;

	// 
	// pdf(r) = exp(-r/mfp) / mfp  (exponential sampling)
	// pdf(phi) = 1/(2*pi)
	// pdf(wi) = cos_out / pi  (cosine-weighted)
	double pdf_r   = (double)r * exp(-(double)r / (double)mfp) / ((double)mfp * (double)mfp);
	double pdf_phi = 1.0 / (2.0 * GPU_PI);
	double pdf_wi  = cos_out / GPU_PI;
	double pdf     = pdf_r * pdf_phi * pdf_wi;
	if (pdf <= 1e-10) return bs;

	// 
	// f = T_in * Rd * T_out * cos_out / pi
	// The pi in the denominator and the cos_out cancel with the pdf,
	// but we keep the full form here and let the integrator divide by pdf.
	vec3 f_val = Rd * (T_in * T_out * cos_out / GPU_PI);

	bs.wi       = wi;
	bs.f        = f_val;
	bs.pdf      = pdf;
	bs.is_delta = false;
	return bs;
}

// Public interface

__device__ inline vec3 gpu_f(const GpuMaterial& mat,
							   const vec3& wi, const vec3& normal) {
	if (mat.type == MatType::LAMBERTIAN) {
		if (dot(normal, wi) <= 0.0) return vec3(0,0,0);
		return mat.albedo / GPU_PI;
	}
	if (mat.type == MatType::GGX) {
		if (dot(normal, wi) <= 0.0) return vec3(0,0,0);
		return ggx_eval(mat, normal, wi, normal);
	}
	if (mat.type == MatType::SSS) {
		// SSS is not evaluated via gpu_f (it needs the full hit context).
		// Return a Lambertian approximation for ReSTIR GI p_hat estimation.
		if (dot(normal, wi) <= 0.0) return vec3(0,0,0);
		return mat.albedo / GPU_PI;
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
		double alpha = (double)mat.roughness * (double)mat.roughness;
		onb uvw; uvw.build_from_w(rec.normal);

		vec3 wo_l = -unit_vector(vec3(
			dot(wo.direction(), uvw.u()),
			dot(wo.direction(), uvw.v()),
			dot(wo.direction(), uvw.w())
		));

		if (wo_l.z() <= 0.0) return bs;

		vec3 h_l = ggx_sample_vndf(wo_l, alpha, rng);
		vec3 h   = unit_vector(uvw.local(h_l));
		vec3 wi  = unit_vector(reflect(-unit_vector(wo.direction()), h));

		if (dot(wi, rec.normal) <= 0.0) return bs;

		double p = ggx_vndf_pdf(wo_l, h_l, alpha);
		if (p <= 0.0) return bs;

		vec3 v = -unit_vector(wo.direction());
		vec3 f = ggx_eval(mat, v, wi, rec.normal);
		return { wi, f, p, false };
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
		// SSS sampling — dipole model, no scene intersection needed for the
		// basic single-layer case (exit point stays on the same local surface).
		// For a full multi-layer implementation, scene intersection would find
		// the actual exit point; here we sample the exit direction analytically.
		return gpu_sample_sss(mat, wo, rec,
							  nullptr, 0,   // no scene intersection in basic mode
							  nullptr, nullptr, 0, 0,
							  rng);
	}

	return bs;
}