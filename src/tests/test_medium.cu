// Participating-medium checks. Guards the CPU/GPU medium model: both backends
// must sample the same phase function in the same frame, and the GPU must treat
// ray parameters as parameters, not as world distances.

#include "tests/test_util.h"
#include "materials/material.h"
#include "cuda/gpu_volume.cuh"
#include "cuda/cuda_rand.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <memory>
#include <vector>

namespace {

constexpr double SIGMA_T = 0.003;
constexpr double ALBEDO  = 0.9;
constexpr double BOX_HI  = 555.0;

struct PhaseAcc {
	double norm_sum;     // integral of p over the sphere
	double mean_cos_sum; // integral of p*cos over the sphere
	double sampled_cos;  // mean cos of sampled directions
	unsigned long long n_sampled;
};

__global__ void phase_kernel(float g, int iters, PhaseAcc* acc) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curandState rng;
	curand_init(4242ULL, tid, 0, &rng);

	const vec3 fwd(0, 0, 1);
	double norm = 0.0, mcos = 0.0, scos = 0.0;

	for (int i = 0; i < iters; ++i) {
		vec3   wi = rand_unit_vector(&rng);
		double p  = hg_phase((float)dot(fwd, wi), g);
		norm += p;
		mcos += p * dot(fwd, wi);

		vec3 s = hg_sample(fwd, g, &rng);
		scos  += dot(fwd, s);
	}

	const double sphere = 4.0 * GPU_PI;
	atomicAdd(&acc->norm_sum,     norm * sphere);
	atomicAdd(&acc->mean_cos_sum, mcos * sphere);
	atomicAdd(&acc->sampled_cos,  scos);
	atomicAdd(&acc->n_sampled,    (unsigned long long)iters);
}

__global__ void phase_eval_kernel(float g, const vec3* wis, int n,
								   double* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	out[i] = (double)hg_phase((float)dot(vec3(0,0,1), wis[i]), g);
}

// Transmittance for a set of rays whose directions differ only in length.
__global__ void tr_kernel(GpuMedium med, const vec3* origins,
						   const vec3* dirs, int n, double* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	out[i] = transmittance_seg(med, ray(origins[i], dirs[i], 0.0), 1e30).x();
}

struct MediumAcc {
	unsigned long long scattered, total;
	double weight_scatter, weight_through;
};

__global__ void medium_kernel(GpuMedium med, vec3 origin, vec3 dir,
							   int iters, MediumAcc* acc) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curandState rng;
	curand_init(1357ULL, tid, 0, &rng);

	unsigned long long sc = 0;
	double ws = 0.0, wt = 0.0;

	for (int i = 0; i < iters; ++i) {
		MediumSample ms = sample_medium(med, ray(origin, dir, 0.0), 1e30, &rng);
		if (ms.scattered) { ++sc; ws += ms.weight.x(); }
		else              {        wt += ms.weight.x(); }
	}

	atomicAdd(&acc->scattered,      sc);
	atomicAdd(&acc->total,          (unsigned long long)iters);
	atomicAdd(&acc->weight_scatter, ws);
	atomicAdd(&acc->weight_through, wt);
}

GpuMedium make_medium(double g) {
	GpuMedium m{};
	double ss = SIGMA_T * ALBEDO, sa = SIGMA_T * (1.0 - ALBEDO);
	m.sigma_s = vec3(ss, ss, ss);
	m.sigma_a = vec3(sa, sa, sa);
	m.sigma_t = m.sigma_s + m.sigma_a;
	m.bmin    = vec3(0, 0, 0);
	m.bmax    = vec3(BOX_HI, BOX_HI, BOX_HI);
	m.g       = (float)g;
	m.active  = true;
	return m;
}

// Mean cosine of Henyey-Greenstein is exactly g. A sampler built in the wrong
// frame stays self-consistent with its own pdf but returns -g.
void test_phase(double g) {
	char tag[64];
	std::snprintf(tag, sizeof(tag), "hg(g=%.2f)", g);
	std::string name(tag);

	PhaseAcc h{}, *d;
	cudaMalloc(&d, sizeof(PhaseAcc));
	cudaMemcpy(d, &h, sizeof(PhaseAcc), cudaMemcpyHostToDevice);
	phase_kernel<<<64, 128>>>((float)g, 400, d);
	cudaDeviceSynchronize();
	cudaMemcpy(&h, d, sizeof(PhaseAcc), cudaMemcpyDeviceToHost);
	cudaFree(d);

	double n    = (double)h.n_sampled;
	double norm = h.norm_sum / n;
	double mcos = h.mean_cos_sum / n;
	double scos = h.sampled_cos / n;

	check(std::abs(norm - 1.0) < 0.01, name + ": GPU phase integrates to 1",
		  norm, 1.0, 0.01);
	check(std::abs(mcos - g) < 0.01, name + ": GPU phase mean cosine = g",
		  mcos, g, 0.01);
	check(std::abs(scos - g) < 0.01, name + ": GPU hg_sample mean cosine = g",
		  scos, g, 0.01);

	// Same three quantities for the CPU phase function.
	isotropic iso(color(1, 1, 1), g);
	hit_record rec;
	rec.p = point3(0, 0, 0); rec.normal = vec3(0, 0, 1);
	rec.u = rec.v = 0.0; rec.front_face = true;

	const vec3 fwd(0, 0, 1);
	const vec3 wo = -fwd;              // wo points back along the incoming ray
	double cnorm = 0.0, cmcos = 0.0, cscos = 0.0;
	const int N = 200000;
	for (int i = 0; i < N; ++i) {
		vec3   wi = random_unit_vector();
		double p  = iso.pdf_dir(wo, wi, rec);
		cnorm += p;
		cmcos += p * dot(fwd, wi);

		BSDFSample bs = iso.sample_dir(wo, rec);
		cscos += dot(fwd, bs.wi);
	}
	cnorm = cnorm * 4.0 * pi / N;
	cmcos = cmcos * 4.0 * pi / N;
	cscos /= N;

	check(std::abs(cnorm - 1.0) < 0.01, name + ": CPU phase integrates to 1",
		  cnorm, 1.0, 0.01);
	check(std::abs(cmcos - g) < 0.01, name + ": CPU phase mean cosine = g",
		  cmcos, g, 0.01);
	check(std::abs(cscos - g) < 0.01, name + ": CPU sample mean cosine = g",
		  cscos, g, 0.01);

	// Cross-backend: identical densities for identical geometry.
	const int M = 4096;
	std::vector<vec3> wis(M);
	for (int i = 0; i < M; ++i) wis[i] = random_unit_vector();

	vec3* d_wi; double* d_gp;
	cudaMalloc(&d_wi, M*sizeof(vec3)); cudaMalloc(&d_gp, M*sizeof(double));
	cudaMemcpy(d_wi, wis.data(), M*sizeof(vec3), cudaMemcpyHostToDevice);
	phase_eval_kernel<<<(M+127)/128, 128>>>((float)g, d_wi, M, d_gp);
	cudaDeviceSynchronize();
	std::vector<double> gp(M);
	cudaMemcpy(gp.data(), d_gp, M*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_wi); cudaFree(d_gp);

	double worst = 0.0;
	for (int i = 0; i < M; ++i) {
		double cp = iso.pdf_dir(wo, wis[i], rec);
		double m  = std::max(cp, gp[i]);
		if (m > 1e-12) worst = std::max(worst, std::abs(cp - gp[i]) / m);
	}
	check(worst < 1e-5, name + ": CPU/GPU phase agree", worst, 0.0, 1e-5);
}

// transmittance_seg must depend on world distance, not on the ray parameter,
// so scaling the direction must not change the answer.
void test_transmittance_scale_invariance() {
	GpuMedium med = make_medium(0.0);

	// Straight through the box along z: 555 units of medium whatever the
	// parameterisation. Also one origin inside the box.
	const int  N = 5;
	vec3 h_o[N] = { vec3(278,278,-800), vec3(278,278,-800), vec3(278,278,-800),
					vec3(278,278,-800), vec3(278,278, 100) };
	vec3 h_d[N] = { vec3(0,0,1), vec3(0,0,800), vec3(0,0,0.25),
					vec3(0,0,137.5), vec3(0,0,640) };
	double expect[N] = {
		std::exp(-SIGMA_T * BOX_HI), std::exp(-SIGMA_T * BOX_HI),
		std::exp(-SIGMA_T * BOX_HI), std::exp(-SIGMA_T * BOX_HI),
		std::exp(-SIGMA_T * (BOX_HI - 100.0))
	};

	vec3 *d_o, *d_d; double* d_out;
	cudaMalloc(&d_o, N*sizeof(vec3)); cudaMalloc(&d_d, N*sizeof(vec3));
	cudaMalloc(&d_out, N*sizeof(double));
	cudaMemcpy(d_o, h_o, N*sizeof(vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, h_d, N*sizeof(vec3), cudaMemcpyHostToDevice);
	tr_kernel<<<1, N>>>(med, d_o, d_d, N, d_out);
	cudaDeviceSynchronize();
	double got[N];
	cudaMemcpy(got, d_out, N*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_o); cudaFree(d_d); cudaFree(d_out);

	for (int i = 0; i < N; ++i) {
		char tag[80];
		std::snprintf(tag, sizeof(tag),
					  "transmittance |dir|=%.4g", h_d[i].length());
		check(std::abs(got[i] - expect[i]) < 1e-6, std::string(tag),
			  got[i], expect[i], 1e-6);
	}
}

// Free flight is sampled in world distance, so the scatter probability over a
// fixed span is fixed, and the weights are the single-scattering albedo and 1.
void test_free_flight() {
	GpuMedium med = make_medium(0.2);
	const double p_scatter = 1.0 - std::exp(-SIGMA_T * BOX_HI);

	const double lens[3] = { 1.0, 800.0, 0.25 };
	for (double L : lens) {
		MediumAcc h{}, *d;
		cudaMalloc(&d, sizeof(MediumAcc));
		cudaMemcpy(d, &h, sizeof(MediumAcc), cudaMemcpyHostToDevice);
		medium_kernel<<<64, 128>>>(med, vec3(278,278,-800), vec3(0,0,L),
								    200, d);
		cudaDeviceSynchronize();
		cudaMemcpy(&h, d, sizeof(MediumAcc), cudaMemcpyDeviceToHost);
		cudaFree(d);

		char tag[80];
		std::snprintf(tag, sizeof(tag), "free flight |dir|=%.4g", L);
		std::string name(tag);

		double frac = (double)h.scattered / (double)h.total;
		check(std::abs(frac - p_scatter) < 0.01, name + ": scatter fraction",
			  frac, p_scatter, 0.01);

		if (h.scattered > 0) {
			double w = h.weight_scatter / (double)h.scattered;
			check(std::abs(w - ALBEDO) < 1e-6, name + ": scatter weight = albedo",
				  w, ALBEDO, 1e-6);
		}
		unsigned long long thru = h.total - h.scattered;
		if (thru > 0) {
			double w = h.weight_through / (double)thru;
			check(std::abs(w - 1.0) < 1e-6, name + ": through weight = 1",
				  w, 1.0, 1e-6);
		}
	}
}

// The integrators branch on this to skip the cosine, which a phase function
// does not have.
void test_phase_flag() {
	isotropic iso(color(0.8,0.8,0.8), 0.2);
	lambertian lam(color(0.5,0.5,0.5));
	check(iso.is_phase_function(), "isotropic: is_phase_function", 1.0, 1.0, 0.0);
	check(!lam.is_phase_function(), "lambertian: is_phase_function false",
		  0.0, 0.0, 0.0);

	hit_record rec;
	rec.p = point3(0,0,0); rec.normal = vec3(1,0,0);
	rec.u = rec.v = 0.0; rec.front_face = true;

	// f/pdf is the single-scattering albedo, with no cosine anywhere.
	double worst = 0.0;
	for (int i = 0; i < 20000; ++i) {
		BSDFSample bs = iso.sample_dir(-vec3(0,0,1), rec);
		if (bs.pdf <= 0.0) continue;
		double r = 0.2126*bs.f.x() + 0.7152*bs.f.y() + 0.0722*bs.f.z();
		worst = std::max(worst, std::abs(r / bs.pdf - 0.8));
	}
	check(worst < 1e-9, "isotropic: f/pdf = albedo", worst, 0.0, 1e-9);
}

} // namespace

void run_medium_tests() {
	std::printf("Medium checks\n");
	for (double g : { 0.0, 0.2, -0.4, 0.7 }) test_phase(g);
	test_transmittance_scale_invariance();
	test_free_flight();
	test_phase_flag();
}
