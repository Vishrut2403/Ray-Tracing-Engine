#include "tests/test_util.h"
#include "materials/material.h"
#include "cuda/gpu_material.cuh"
#include "cuda/cuda_rand.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <memory>

namespace {

constexpr int N_DIRS = 512;

struct Acc {
	unsigned long long pdf_mismatch, drawn, rejected, density, refl_n;
	double refl_sum, mis_sum;
};

__global__ void eval_kernel(const GpuMaterial* mats, int n_mats,
							 const vec3* wos, const vec3* wis, int n_dirs,
							 vec3* out_f, double* out_pdf, vec3* out_f_swapped) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_mats * n_dirs) return;
	int m = i / n_dirs, d = i % n_dirs;
	vec3 n(0,0,1);
	out_f[i]         = gpu_f_dir  (mats[m], wos[d], wis[d], n);
	out_pdf[i]       = gpu_pdf_dir(mats[m], wos[d], wis[d], n);
	out_f_swapped[i] = gpu_f_dir  (mats[m], wis[d], wos[d], n);
}

__global__ void sample_kernel(const GpuMaterial* mats, int n_mats,
							   int iters_per_thread, vec3 wo_probe,
							   vec3 wo_oblique, Acc* acc) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int m   = blockIdx.y;
	if (m >= n_mats) return;

	curandState rng;
	curand_init(9876ULL, tid + m * 100003, 0, &rng);

	const vec3   n(0,0,1);
	const double q_uniform = 1.0 / (4.0 * GPU_PI);

	GpuHitRecord rec{};
	rec.p = vec3(0,0,0); rec.normal = n; rec.u = rec.v = 0.0;
	rec.front_face = true; rec.mat_id = 0;

	const GpuMaterial& mat = mats[m];

	for (int i = 0; i < iters_per_thread; ++i) {
		vec3   wi = rand_unit_vector(&rng);
		double p  = gpu_pdf_dir(mat, wo_probe, wi, n);
		if (p > 0.0) atomicAdd(&acc[m].mis_sum, p / (q_uniform + p));

		GpuBSDFSample bs = gpu_sample_dir(mat, wo_probe, rec, &rng);
		if (!bs.is_delta) {
			atomicAdd(&acc[m].drawn, 1ULL);
			if (bs.pdf <= 0.0) {
				atomicAdd(&acc[m].rejected, 1ULL);
			} else {
				double p2 = gpu_pdf_dir(mat, wo_probe, bs.wi, n);
				if (p2 > 0.0) {
					atomicAdd(&acc[m].mis_sum, p2 / (q_uniform + p2));
					atomicAdd(&acc[m].density, 1ULL);
				}
				double denom = rmax(p2, bs.pdf);
				if (denom > 0.0 && fabs(p2 - bs.pdf) / denom > 1e-4)
					atomicAdd(&acc[m].pdf_mismatch, 1ULL);
			}
		}

		GpuBSDFSample rs = gpu_sample_dir(mat, wo_oblique, rec, &rng);
		if (rs.pdf > 0.0) {
			double c   = fabs(dot(n, rs.wi));
			double lum = 0.2126*rs.f.x() + 0.7152*rs.f.y() + 0.0722*rs.f.z();
			atomicAdd(&acc[m].refl_sum, lum * c / rs.pdf);
		}
		atomicAdd(&acc[m].refl_n, 1ULL);
	}
}

double lum(const vec3& c) {
	return 0.2126*c.x() + 0.7152*c.y() + 0.0722*c.z();
}

struct Pair {
	std::string name;
	GpuMaterial gpu;
	std::shared_ptr<material> cpu;
	double expected_rho;   // negative = only the <= 1 bound
	double tol;
	bool   reciprocal;
};

GpuMaterial mk(MatType t, vec3 albedo, float fuzz, float ir,
			   float rough, float metallic, float mfp) {
	GpuMaterial m{}; m.type = t; m.albedo = albedo; m.fuzz = fuzz;
	m.ir = ir; m.roughness = rough; m.metallic = metallic; m.mfp = mfp;
	return m;
}

} // namespace

void run_gpu_tests() {
	int dev_count = 0;
	if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
		std::printf("  SKIP  no CUDA device — GPU checks not run\n");
		return;
	}
	std::printf("GPU material checks\n");

	// Exactly representable in float, so a CPU/GPU difference is a real
	// divergence rather than storage precision.
	std::vector<Pair> cases = {
		{ "lambertian(0.5)",
		  mk(MatType::LAMBERTIAN, vec3(0.5,0.5,0.5), 0,0,0,0,0),
		  std::make_shared<lambertian>(color(0.5,0.5,0.5)), 0.5, 0.01, true },
		{ "metal(0.75, fuzz 0)",
		  mk(MatType::METAL, vec3(0.75,0.75,0.75), 0.0f,0,0,0,0),
		  std::make_shared<metal>(color(0.75,0.75,0.75), 0.0), 0.75, 0.01, true },
		{ "dielectric(1.5)",
		  mk(MatType::DIELECTRIC, vec3(1,1,1), 0, 1.5f, 0,0,0),
		  std::make_shared<dielectric>(1.5), 1.0, 0.01, true },
		{ "ggx(rough 0.25, metallic 1)",
		  mk(MatType::GGX, vec3(0.75,0.5,0.25), 0,0, 0.25f, 1.0f, 0),
		  std::make_shared<ggx>(color(0.75,0.5,0.25), 0.25, 1.0), -1.0, 0.0, true },
		{ "ggx(rough 0.5, metallic 0)",
		  mk(MatType::GGX, vec3(0.25,0.5,0.75), 0,0, 0.5f, 0.0f, 0),
		  std::make_shared<ggx>(color(0.25,0.5,0.75), 0.5, 0.0), -1.0, 0.0, true },
		{ "isotropic(0.5)",
		  mk(MatType::ISOTROPIC, vec3(0.5,0.5,0.5), 0,0,0,0,0),
		  std::make_shared<isotropic>(color(0.5,0.5,0.5)), -1.0, 0.0, true },
		{ "rough_dielectric(rough 0.25, ior 1.5)",
		  mk(MatType::ROUGH_DIELECTRIC, vec3(1,1,1), 0, 1.5f, 0.25f, 0, 0),
		  std::make_shared<rough_dielectric>(color(1,1,1), 0.25, 1.5),
		  -1.0, 0.0, false },
		{ "subsurface(0.75, mfp 0.125)",
		  mk(MatType::SSS, vec3(0.75,0.75,0.75), 0, 1.5f, 0,0, 0.125f),
		  std::make_shared<subsurface>(color(0.75,0.75,0.75), 0.125, 1.5),
		  -1.0, 0.0, false },
	};
	const int M = (int)cases.size();

	std::vector<vec3> wos(N_DIRS), wis(N_DIRS);
	for (int i = 0; i < N_DIRS; ++i) {
		double a = 0.7548776662 * (i + 1), b = 0.5698402909 * (i + 1);
		a -= std::floor(a); b -= std::floor(b);
		double t1 = a * 2.0 * pi, z1 = 0.05 + 0.94 * b;
		double r1 = std::sqrt(1.0 - z1*z1);
		wos[i] = vec3(r1*std::cos(t1), r1*std::sin(t1), z1);
		double t2 = b * 2.0 * pi, z2 = 0.05 + 0.94 * a;
		double r2 = std::sqrt(1.0 - z2*z2);
		wis[i] = vec3(r2*std::cos(t2), r2*std::sin(t2), z2);
	}

	GpuMaterial* d_mats; vec3 *d_wo, *d_wi, *d_f, *d_fs; double* d_pdf;
	std::vector<GpuMaterial> host_mats;
	for (auto& c : cases) host_mats.push_back(c.gpu);

	cudaMalloc(&d_mats, M*sizeof(GpuMaterial));
	cudaMalloc(&d_wo, N_DIRS*sizeof(vec3));
	cudaMalloc(&d_wi, N_DIRS*sizeof(vec3));
	cudaMalloc(&d_f,  (size_t)M*N_DIRS*sizeof(vec3));
	cudaMalloc(&d_fs, (size_t)M*N_DIRS*sizeof(vec3));
	cudaMalloc(&d_pdf,(size_t)M*N_DIRS*sizeof(double));
	cudaMemcpy(d_mats, host_mats.data(), M*sizeof(GpuMaterial), cudaMemcpyHostToDevice);
	cudaMemcpy(d_wo, wos.data(), N_DIRS*sizeof(vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_wi, wis.data(), N_DIRS*sizeof(vec3), cudaMemcpyHostToDevice);

	int total = M*N_DIRS, thr = 128;
	eval_kernel<<<(total+thr-1)/thr, thr>>>(d_mats, M, d_wo, d_wi, N_DIRS,
											d_f, d_pdf, d_fs);
	cudaDeviceSynchronize();

	std::vector<vec3>   h_f(total), h_fs(total);
	std::vector<double> h_pdf(total);
	cudaMemcpy(h_f.data(),   d_f,   total*sizeof(vec3),   cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fs.data(),  d_fs,  total*sizeof(vec3),   cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pdf.data(), d_pdf, total*sizeof(double), cudaMemcpyDeviceToHost);

	for (int m = 0; m < M; ++m) {
		hit_record rec;
		rec.p = point3(0,0,0); rec.normal = vec3(0,0,1);
		rec.u = rec.v = 0.0; rec.front_face = true;

		double worst_f = 0.0, worst_p = 0.0, worst_recip = 0.0;
		for (int d = 0; d < N_DIRS; ++d) {
			int i = m*N_DIRS + d;
			double gf = lum(h_f[i]), gp = h_pdf[i];
			double cf = lum(cases[m].cpu->f_dir(wos[d], wis[d], rec));
			double cp = cases[m].cpu->pdf_dir(wos[d], wis[d], rec);

			double df = std::max(std::abs(gf), std::abs(cf));
			if (df > 1e-12) worst_f = std::max(worst_f, std::abs(gf-cf)/df);
			double dp = std::max(std::abs(gp), std::abs(cp));
			if (dp > 1e-12) worst_p = std::max(worst_p, std::abs(gp-cp)/dp);

			double fs = lum(h_fs[i]);
			double mx = std::max(gf, fs);
			if (mx > 1e-12) worst_recip = std::max(worst_recip, std::abs(gf-fs)/mx);
		}
		check(worst_f < kNumericTol, cases[m].name + ": CPU/GPU f_dir agree",
			  worst_f, 0.0, kNumericTol);
		check(worst_p < kNumericTol, cases[m].name + ": CPU/GPU pdf_dir agree",
			  worst_p, 0.0, kNumericTol);
		if (cases[m].reciprocal)
			check(worst_recip < kNumericTol, cases[m].name + ": GPU f is reciprocal",
				  worst_recip, 0.0, kNumericTol);
	}

	Acc* d_acc; cudaMalloc(&d_acc, M*sizeof(Acc));
	cudaMemset(d_acc, 0, M*sizeof(Acc));

	const int threads = 128, blocks = 64, iters = 32;
	const long long N = (long long)threads * blocks * iters;

	vec3 wo_probe   = unit_vector(vec3(0.3, 0.1, 0.9));
	vec3 wo_oblique = unit_vector(vec3(0.7, 0.0, 0.7));
	dim3 grid(blocks, M);
	sample_kernel<<<grid, threads>>>(d_mats, M, iters, wo_probe, wo_oblique, d_acc);
	cudaDeviceSynchronize();

	std::vector<Acc> acc(M);
	cudaMemcpy(acc.data(), d_acc, M*sizeof(Acc), cudaMemcpyDeviceToHost);

	for (int m = 0; m < M; ++m) {
		const auto& a = acc[m];
		check(a.pdf_mismatch == 0,
			  cases[m].name + ": GPU sample()/pdf() agree",
			  (double)a.pdf_mismatch, 0.0, 0.5);

		if (a.density > 0 && a.drawn > 0) {
			double reject   = (double)a.rejected / (double)a.drawn;
			double integral = a.mis_sum / (double)N;
			check(std::abs(integral + reject - 1.0) < 0.01,
				  cases[m].name + ": GPU pdf mass + rejection = 1",
				  integral + reject, 1.0, 0.01);
		}

		double rho = a.refl_sum / (double)a.refl_n;
		check(rho <= 1.0 + 1e-3, cases[m].name + ": GPU reflectance <= 1 (energy)",
			  rho, 1.0, 1e-3);
		if (cases[m].expected_rho >= 0.0)
			check(std::abs(rho - cases[m].expected_rho) < cases[m].tol,
				  cases[m].name + ": GPU reflectance matches albedo",
				  rho, cases[m].expected_rho, cases[m].tol);
	}

	cudaFree(d_mats); cudaFree(d_wo); cudaFree(d_wi);
	cudaFree(d_f); cudaFree(d_fs); cudaFree(d_pdf); cudaFree(d_acc);
}
