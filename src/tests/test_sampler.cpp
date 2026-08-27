// Sampling routine checks. These cover the two things the path tracer needs
// from them: the right distribution, and a fixed number of dimensions drawn in
// a fixed order -- a rejection loop gets the distribution right and the second
// part wrong, which costs the low-discrepancy sequence its stratification.

#include "tests/test_util.h"

#include "core/random.h"
#include "core/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

const double kPi = 3.14159265358979323846;

// How many dimensions of the sample sequence one call consumes.
template <typename F>
uint32_t dims_used(F&& f) {
	uint32_t before = g_sampler.dim;
	f();
	return g_sampler.dim - before;
}

void test_fixed_dimension_cost() {
	// A rejection loop draws again on every reject, so its cost varies with
	// the values it happens to draw. Sweep enough samples to catch that.
	uint32_t disk_min = ~0u, disk_max = 0;
	uint32_t sph_min  = ~0u, sph_max  = 0;
	uint32_t dir_min  = ~0u, dir_max  = 0;
	uint32_t cos_min  = ~0u, cos_max  = 0;

	for (uint32_t i = 0; i < 4096; ++i) {
		sampler_begin_sample(i * 7u + 3u, i);
		uint32_t d = dims_used([]{ random_in_unit_disk(); });
		uint32_t s = dims_used([]{ random_in_unit_sphere(); });
		uint32_t u = dims_used([]{ random_unit_vector(); });
		uint32_t c = dims_used([]{ random_cosine_direction(); });
		sampler_end_sample();

		disk_min = std::min(disk_min, d); disk_max = std::max(disk_max, d);
		sph_min  = std::min(sph_min,  s); sph_max  = std::max(sph_max,  s);
		dir_min  = std::min(dir_min,  u); dir_max  = std::max(dir_max,  u);
		cos_min  = std::min(cos_min,  c); cos_max  = std::max(cos_max,  c);
	}

	check(disk_min == 2 && disk_max == 2,
		  "dimensions: the unit disk always costs two", (double)disk_max, 2.0, 0.5);
	check(dir_min == 2 && dir_max == 2,
		  "dimensions: a unit vector always costs two", (double)dir_max, 2.0, 0.5);
	check(sph_min == 3 && sph_max == 3,
		  "dimensions: the unit ball always costs three", (double)sph_max, 3.0, 0.5);
	check(cos_min == 2 && cos_max == 2,
		  "dimensions: a cosine direction always costs two", (double)cos_max,
		  2.0, 0.5);
}

// The sequence has to be walked in a fixed order too: two runs from the same
// key must produce the same values, or nothing downstream is reproducible.
void test_order_is_fixed() {
	std::vector<double> a, b;
	for (int pass = 0; pass < 2; ++pass) {
		std::vector<double>& out = pass == 0 ? a : b;
		for (uint32_t i = 0; i < 64; ++i) {
			sampler_begin_sample(i, 11u);
			vec3 d = random_in_unit_disk();
			vec3 s = random_in_unit_sphere();
			vec3 c = random_cosine_direction();
			sampler_end_sample();
			for (int k = 0; k < 3; ++k)
				{ out.push_back(d[k]); out.push_back(s[k]); out.push_back(c[k]); }
		}
	}
	int differing = 0;
	for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++differing;
	check(differing == 0, "dimensions: the same key gives the same draws",
		  (double)differing, 0.0, 0.5);
}

void test_unit_disk() {
	const int N = 200000;
	int outside = 0, off_plane = 0;
	// Equal-area rings: a uniform disk puts an equal count in each.
	const int rings = 8, sectors = 8;
	std::vector<int> ring(rings, 0), sect(sectors, 0);

	for (int i = 0; i < N; ++i) {
		sampler_begin_sample((uint32_t)i, 5u);
		vec3 p = random_in_unit_disk();
		sampler_end_sample();

		double r2 = (double)(p.x()*p.x() + p.y()*p.y());
		if (r2 > 1.0 + 1e-9) ++outside;
		if (p.z() != 0) ++off_plane;
		ring[std::min(rings-1, (int)(r2 * rings))]++;
		double a = std::atan2((double)p.y(), (double)p.x()) + kPi;
		sect[std::min(sectors-1, (int)(a / (2*kPi) * sectors))]++;
	}

	check(outside == 0, "unit disk: every sample is inside the disk",
		  (double)outside, 0.0, 0.5);
	check(off_plane == 0, "unit disk: every sample is in the xy plane",
		  (double)off_plane, 0.0, 0.5);

	double want = (double)N / rings, worst = 0.0;
	for (int c : ring) worst = std::max(worst, std::abs(c - want) / want);
	check(worst < 0.02, "unit disk: equal area rings get equal counts", worst,
		  0.0, 0.02);

	want = (double)N / sectors; worst = 0.0;
	for (int c : sect) worst = std::max(worst, std::abs(c - want) / want);
	check(worst < 0.02, "unit disk: no angular bias", worst, 0.0, 0.02);
}

void test_unit_vector() {
	const int N = 200000;
	double off_unit = 0.0;
	vec3 sum(0,0,0);
	// Equal-area bands in cos(theta), the same trick as the rings above.
	const int bands = 8;
	std::vector<int> band(bands, 0);

	for (int i = 0; i < N; ++i) {
		sampler_begin_sample((uint32_t)i, 9u);
		vec3 d = random_unit_vector();
		sampler_end_sample();
		off_unit = std::max(off_unit, std::abs((double)d.length() - 1.0));
		sum += d;
		double z = std::clamp((double)d.z(), -1.0, 1.0);
		band[std::min(bands-1, (int)((z + 1.0) * 0.5 * bands))]++;
	}

	check(off_unit < 1e-6, "unit vector: every sample is on the sphere",
		  off_unit, 0.0, 1e-6);
	check((double)(sum / (real)N).length() < 0.01,
		  "unit vector: no net direction", (double)(sum/(real)N).length(), 0.0,
		  0.01);

	double want = (double)N / bands, worst = 0.0;
	for (int c : band) worst = std::max(worst, std::abs(c - want) / want);
	check(worst < 0.02, "unit vector: equal solid angle bands get equal counts",
		  worst, 0.0, 0.02);
}

void test_unit_ball() {
	const int N = 200000;
	int outside = 0;
	// Uniform in volume: the fraction within radius t is t^3.
	const int shells = 8;
	std::vector<int> shell(shells, 0);

	for (int i = 0; i < N; ++i) {
		sampler_begin_sample((uint32_t)i, 13u);
		vec3 p = random_in_unit_sphere();
		sampler_end_sample();
		double r = (double)p.length();
		if (r > 1.0 + 1e-9) ++outside;
		shell[std::min(shells-1, (int)(r*r*r * shells))]++;
	}

	check(outside == 0, "unit ball: every sample is inside the ball",
		  (double)outside, 0.0, 0.5);

	double want = (double)N / shells, worst = 0.0;
	for (int c : shell) worst = std::max(worst, std::abs(c - want) / want);
	check(worst < 0.02, "unit ball: equal volume shells get equal counts",
		  worst, 0.0, 0.02);
}

// The cosine density is what the lambertian BSDF's pdf claims it is.
void test_cosine_direction() {
	const int N = 200000;
	int below = 0;
	double sum_z = 0.0;
	for (int i = 0; i < N; ++i) {
		sampler_begin_sample((uint32_t)i, 17u);
		vec3 d = random_cosine_direction();
		sampler_end_sample();
		if (d.z() < 0) ++below;
		sum_z += (double)d.z();
	}
	check(below == 0, "cosine direction: nothing below the surface",
		  (double)below, 0.0, 0.5);
	// E[cos] over a cosine-weighted hemisphere is 2/3.
	check(std::abs(sum_z/N - 2.0/3.0) < 0.005,
		  "cosine direction: mean cosine is 2/3", sum_z/N, 2.0/3.0, 0.005);
}

}  // namespace

void run_sampler_tests() {
	std::printf("\nSampling routine checks\n");
	test_fixed_dimension_cost();
	test_order_is_fixed();
	test_unit_disk();
	test_unit_vector();
	test_unit_ball();
	test_cosine_direction();
}
