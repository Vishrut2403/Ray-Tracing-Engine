// Analytic BSDF checks — closed-form expectations, no reference render.
// Run: ./build/tests   (exit 0 = passed)

#include "tests/test_util.h"
#include "materials/material.h"
#include "core/onb.h"
#include "core/random.h"

#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

int g_checks = 0, g_failures = 0;

void check(bool ok, const std::string& what, double got, double want,
		   double tol) {
	++g_checks;
	if (ok) return;
	++g_failures;
	std::printf("  FAIL  %-52s got %.6f want %.6f (tol %.4f)\n",
				what.c_str(), got, want, tol);
}

static hit_record make_rec(const vec3& n, bool front_face = true) {
	hit_record rec;
	rec.p = point3(0,0,0);
	rec.normal = n;
	rec.u = rec.v = 0.0;
	rec.front_face = front_face;
	return rec;
}

static double lum(const color& c) {
	return 0.2126*c.x() + 0.7152*c.y() + 0.0722*c.z();
}

// sample()'s reported density must match an independent pdf() lookup.
static void test_sample_pdf_agree(const std::string& name,
								   const std::shared_ptr<material>& mat,
								   const vec3& n, int N = 20000) {
	hit_record rec = make_rec(n);
	double worst = 0.0;
	int compared = 0;

	for (int i = 0; i < N; ++i) {
		onb uvw; uvw.build_from_w(n);
		vec3 wo = unit_vector(uvw.local(random_cosine_direction()));

		BSDFSample bs = mat->sample_dir(wo, rec);
		if (bs.pdf <= 0.0 || bs.is_delta) continue;

		double p = mat->pdf_dir(wo, bs.wi, rec);
		double denom = std::max(p, bs.pdf);
		if (denom <= 0.0) continue;
		worst = std::max(worst, std::abs(p - bs.pdf) / denom);
		++compared;
	}
	if (compared == 0) return;   // delta-only material
	check(worst < 1e-6, name + ": sample()/pdf() agree", worst, 0.0, 1e-6);
}

// pdf mass + rejection probability = 1. VNDF sampling discards below-horizon
// reflections, so a rough GGX pdf integrates to less than 1 by exactly that
// rate. Integrated by MIS against the material's own sampler; uniform sampling
// alone cannot resolve a narrow lobe (roughness 0.05 covers ~2e-5 sr).
static void test_pdf_normalized(const std::string& name,
								 const std::shared_ptr<material>& mat,
								 const vec3& n, int N = 200000) {
	hit_record rec = make_rec(n);
	onb uvw; uvw.build_from_w(n);
	vec3 wo = unit_vector(uvw.local(vec3(0.3, 0.1, 0.9)));

	const double q_uniform = 1.0 / (4.0 * pi);
	double sum = 0.0;
	long   drawn = 0, rejected = 0, density_samples = 0;

	for (int i = 0; i < N; ++i) {
		vec3   wi = random_unit_vector();
		double p  = mat->pdf_dir(wo, wi, rec);
		if (p > 0.0) sum += p / (q_uniform + p);

		BSDFSample bs = mat->sample_dir(wo, rec);
		++drawn;
		if (bs.is_delta) { --drawn; continue; }
		if (bs.pdf <= 0.0) { ++rejected; continue; }

		double p2 = mat->pdf_dir(wo, bs.wi, rec);
		if (p2 > 0.0) { sum += p2 / (q_uniform + p2); ++density_samples; }
	}
	if (density_samples == 0) return;   // purely specular

	double reject = (double)rejected / (double)drawn;
	double integral = sum / N;
	check(std::abs(integral + reject - 1.0) < 0.01,
		  name + ": pdf mass + rejection = 1", integral + reject, 1.0, 0.01);
}

// f(wo->wi) == f(wi->wo). A cosine folded into f breaks this and nothing else
// here would notice.
static void test_reciprocity(const std::string& name,
							  const std::shared_ptr<material>& mat,
							  const vec3& n, int N = 20000) {
	hit_record rec = make_rec(n);
	onb uvw; uvw.build_from_w(n);
	double worst = 0.0;
	int compared = 0;

	for (int i = 0; i < N; ++i) {
		vec3 a = unit_vector(uvw.local(random_cosine_direction()));
		vec3 b = unit_vector(uvw.local(random_cosine_direction()));
		double f1 = lum(mat->f_dir(a, b, rec));
		double f2 = lum(mat->f_dir(b, a, rec));
		double m  = std::max(f1, f2);
		if (m < 1e-9) continue;
		worst = std::max(worst, std::abs(f1 - f2) / m);
		++compared;
	}
	if (compared == 0) return;
	check(worst < 1e-6, name + ": f is reciprocal", worst, 0.0, 1e-6);
}

// White furnace: E[f*cos/pdf] must match the known albedo and stay <= 1.
static void test_reflectance(const std::string& name,
							  const std::shared_ptr<material>& mat,
							  const vec3& n, double expected, double tol,
							  int N = 200000) {
	hit_record rec = make_rec(n);
	onb uvw; uvw.build_from_w(n);
	// Oblique: at normal incidence cos = 1 hides a missing 1/|cos|.
	vec3 wo = unit_vector(uvw.local(vec3(0.7, 0.0, 0.7)));

	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		BSDFSample bs = mat->sample_dir(wo, rec);
		if (bs.pdf <= 0.0) continue;
		double c = std::abs(dot(n, bs.wi));
		sum += lum(bs.f) * c / bs.pdf;
	}
	double rho = sum / N;
	check(rho <= 1.0 + 1e-3, name + ": reflectance <= 1 (energy)", rho, 1.0, 1e-3);
	if (expected >= 0.0)
		check(std::abs(rho - expected) < tol,
			  name + ": reflectance matches albedo", rho, expected, tol);
}

int main() {
	std::printf("BSDF analytic checks\n");

	const vec3 n(0,0,1);

	struct Case {
		std::string name;
		std::shared_ptr<material> mat;
		double expected_rho;   // negative = only check the <= 1 bound
		double tol;
		bool   reciprocal;   // subsurface averages entry Fresnel, so it is not
	};

	std::vector<Case> cases = {
		{ "lambertian(0.5)",
		  std::make_shared<lambertian>(color(0.5,0.5,0.5)), 0.5, 0.01, true },
		{ "lambertian(0.73)",
		  std::make_shared<lambertian>(color(0.73,0.73,0.73)), 0.73, 0.01, true },
		{ "metal(0.9, fuzz 0)",
		  std::make_shared<metal>(color(0.9,0.9,0.9), 0.0), 0.9, 0.01, true },
		{ "dielectric(1.5)",
		  std::make_shared<dielectric>(1.5), 1.0, 0.01, true },
		{ "ggx(rough 0.3, metallic 1)",
		  std::make_shared<ggx>(color(1.0,0.76,0.33), 0.3, 1.0), -1.0, 0.0, true },
		{ "ggx(rough 0.6, metallic 0)",
		  std::make_shared<ggx>(color(0.05,0.2,0.8), 0.6, 0.0), -1.0, 0.0, true },
		{ "ggx(rough 0.05, metallic 1)",
		  std::make_shared<ggx>(color(0.9,0.9,0.9), 0.05, 1.0), -1.0, 0.0, true },
		{ "subsurface(0.9, mfp 0.1)",
		  std::make_shared<subsurface>(color(0.9,0.9,0.9), 0.1, 1.4), -1.0, 0.0, false },
		// isotropic is a phase function: spherical and with no cosine, so the
		// albedo-reflectance check does not apply, only the <= 1 bound.
		{ "isotropic(0.8)",
		  std::make_shared<isotropic>(color(0.8,0.8,0.8)), -1.0, 0.0, true },
		{ "rough_dielectric(rough 0.3, ior 1.5)",
		  std::make_shared<rough_dielectric>(color(1,1,1), 0.3, 1.5), -1.0, 0.0, false },
		{ "rough_dielectric(rough 0.6, ior 1.5)",
		  std::make_shared<rough_dielectric>(color(1,1,1), 0.6, 1.5), -1.0, 0.0, false },
	};

	for (const auto& c : cases) {
		test_sample_pdf_agree(c.name, c.mat, n);
		test_pdf_normalized  (c.name, c.mat, n);
		test_reflectance     (c.name, c.mat, n, c.expected_rho, c.tol);
		if (c.reciprocal) test_reciprocity(c.name, c.mat, n);
	}

	run_gpu_tests();

	std::printf("%d checks, %d failed\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
