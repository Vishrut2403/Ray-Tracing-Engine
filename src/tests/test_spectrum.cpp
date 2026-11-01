// Colour science checks. Spectral rendering is easy to get subtly wrong in a
// way that looks like a shading bug, so the conversion is pinned here against
// published values before any wavelength reaches the integrators.

#include "tests/test_util.h"
#include "core/spectrum.h"
#include "core/dispersion.h"

#include <cmath>
#include <cstdio>

namespace {

// Trapezoid over the visible range at 1nm, the spacing the CIE tables use.
template <typename F>
double integrate(F&& f) {
	double sum = 0.0;
	for (real l = kLambdaMin; l <= kLambdaMax; l += 1)
		sum += (double)f(l);
	return sum;
}

void test_curves() {
	// The fit must reproduce the tabulated luminance integral, which is what
	// every normalization downstream divides by.
	double y_int = integrate(cie_y);
	check(std::abs(y_int - 106.857) / 106.857 < 0.02,
		  "cie: luminance curve integrates to the tabulated value",
		  y_int, 106.857, 0.02 * 106.857);

	// Photopic peak at 555nm is the one number everyone knows.
	real peak = kLambdaMin; double best = 0.0;
	for (real l = kLambdaMin; l <= kLambdaMax; l += 1)
		if ((double)cie_y(l) > best) { best = cie_y(l); peak = l; }
	check(std::abs((double)peak - 555.0) <= 2.0,
		  "cie: luminance peaks at 555nm", (double)peak, 555.0, 2.0);

	int negative = 0;
	for (real l = kLambdaMin; l <= kLambdaMax; l += 1)
		if (cie_y(l) < 0 || cie_z(l) < 0) ++negative;
	check(negative == 0, "cie: y and z are non-negative",
		  (double)negative, 0.0, 0.5);
}

// Chromaticity is independent of the primaries, so it tests the curves without
// dragging sRGB's gamut into it.
void chromaticity(const vec3& xyz, double& x, double& y) {
	double s = (double)(xyz.x() + xyz.y() + xyz.z());
	x = (double)xyz.x() / s;
	y = (double)xyz.y() / s;
}

void test_equal_energy_is_neutral() {
	vec3 sum(0,0,0);
	for (real l = kLambdaMin; l <= kLambdaMax; l += 1)
		sum += cie_xyz(l);

	double x, y;
	chromaticity(sum, x, y);
	// Illuminant E sits at the centre of the chromaticity diagram by
	// definition; a fit that drifts here is a fit with the wrong areas.
	check(std::abs(x - 1.0/3.0) < 0.01,
		  "equal energy: chromaticity x is 1/3", x, 1.0/3.0, 0.01);
	check(std::abs(y - 1.0/3.0) < 0.01,
		  "equal energy: chromaticity y is 1/3", y, 1.0/3.0, 0.01);
}

void test_monochromatic_locus() {
	// Points on the spectral locus, from the 1931 2-degree tables.
	struct { real lambda; double x, y; } pts[] = {
		{ 450, 0.1566, 0.0177 },
		{ 500, 0.0082, 0.5384 },
		{ 550, 0.3016, 0.6923 },
		{ 600, 0.6270, 0.3725 },
	};

	for (const auto& p : pts) {
		double x, y;
		chromaticity(cie_xyz(p.lambda), x, y);
		char tag[64];
		std::snprintf(tag, sizeof(tag), "locus %dnm", (int)p.lambda);
		// The analytic fit is a percent-level approximation, so the tolerance
		// is the fit's error, not the table's.
		check(std::hypot(x - p.x, y - p.y) < 0.03,
			  std::string(tag) + ": lands on the spectral locus",
			  std::hypot(x - p.x, y - p.y), 0.0, 0.03);
	}
}

// A flat reflectance lit by a flat illuminant has to come back grey, at the
// reflectance's own value. This is the end-to-end check the renderer will lean
// on: it fixes the normalization, which nothing else pins down.
void test_flat_spectrum_round_trip() {
	for (double rho : { 0.18, 0.5, 1.0 }) {
		vec3 xyz(0,0,0);
		for (real l = kLambdaMin; l <= kLambdaMax; l += 1)
			xyz += spectral_to_xyz(l, (real)rho);

		color rgb = xyz_to_rgb(xyz);
		char tag[64];
		std::snprintf(tag, sizeof(tag), "flat %.2f", rho);

		check(std::abs((double)xyz.y() - rho) < 0.01 * rho,
			  std::string(tag) + ": luminance equals the reflectance",
			  (double)xyz.y(), rho, 0.01 * rho);

		// Equal-energy white is illuminant E, and the sRGB matrix is referenced
		// to D65, so E comes out warm by construction -- that is colorimetry,
		// not a bug, and scenes under a non-D65 illuminant will want chromatic
		// adaptation. Pin the closed form so the matrix cannot drift.
		const double want[3] = { 1.2048, 0.9484, 0.9087 };
		for (int k = 0; k < 3; ++k)
			check(std::abs((double)rgb[k] / rho - want[k]) < 0.01,
				  std::string(tag) + ": equal-energy white hits its D65-referenced value",
				  (double)rgb[k] / rho, want[k], 0.01);
	}
}

// Abbe numbers are how glass is actually catalogued, so the conversion to
// Cauchy's coefficients is worth pinning: both defining relations are exact by
// construction, and a sign slip in either breaks them visibly.
void test_dispersion() {
	struct { const char* name; double n_d, V; } glasses[] = {
		{ "BK7",  1.5168, 64.17 },
		{ "SF10", 1.7283, 28.53 },
		{ "F2",   1.6200, 36.37 },
	};

	for (const auto& g : glasses) {
		Dispersion d = Dispersion::from_abbe((real)g.n_d, (real)g.V);

		double n_d = d.ior(kFraunhoferD * 1000);
		double n_F = d.ior(kFraunhoferF * 1000);
		double n_C = d.ior(kFraunhoferC * 1000);

		check(std::abs(n_d - g.n_d) < 1e-4,
			  std::string(g.name) + ": index at the d line", n_d, g.n_d, 1e-4);

		double V = (n_d - 1) / (n_F - n_C);
		check(std::abs(V - g.V) / g.V < 1e-3,
			  std::string(g.name) + ": Abbe number recovered", V, g.V, 1e-3*g.V);

		// Normal dispersion: blue bends more than red. The sign is the whole
		// visible effect, so it gets its own check.
		check(n_F > n_C,
			  std::string(g.name) + ": blue refracts more than red",
			  n_F - n_C, 1.0, 0.0);
	}

	// Zero Abbe is the opt-out, and has to be exactly constant.
	Dispersion flat = Dispersion::from_abbe(1.5, 0);
	check(flat.ior(400) == flat.ior(700) && flat.ior(550) == (real)1.5,
		  "no Abbe number: the glass is exactly non-dispersive",
		  (double)flat.ior(400), 1.5, 0.0);
}

// The adaptation has one defining property: it must send the source white
// exactly onto the destination white. Everything else follows from that.
void test_chromatic_adaptation() {
	vec3 e(kWhiteE[0], kWhiteE[1], kWhiteE[2]);
	vec3 d65(kWhiteD65[0], kWhiteD65[1], kWhiteD65[2]);

	vec3 got = chromatic_adapt(e, kWhiteE, kWhiteD65);
	check((double)(got - d65).length() < 1e-4,
		  "adaptation: equal-energy white lands on D65",
		  (double)(got - d65).length(), 0.0, 1e-4);

	// Adapting to the white you already have must change nothing.
	vec3 probe(0.31, 0.47, 0.22);
	vec3 same = chromatic_adapt(probe, kWhiteD65, kWhiteD65);
	check((double)(same - probe).length() < 1e-4,
		  "adaptation: same white in and out is the identity",
		  (double)(same - probe).length(), 0.0, 1e-4);

	// And it must be invertible: there and back is where a transposed matrix
	// or a swapped white point shows up.
	vec3 back = chromatic_adapt(chromatic_adapt(probe, kWhiteE, kWhiteD65),
								kWhiteD65, kWhiteE);
	check((double)(back - probe).length() < 1e-4,
		  "adaptation: the round trip returns the colour",
		  (double)(back - probe).length(), 0.0, 1e-4);

	// A flat spectrum is the case the renderer actually hits, and after
	// adaptation it has to be neutral rather than warm.
	vec3 xyz(0,0,0);
	for (real l = kLambdaMin; l <= kLambdaMax; l += 1)
		xyz += spectral_to_xyz(l, (real)0.5);
	color rgb = xyz_to_rgb(chromatic_adapt(xyz, kWhiteE, kWhiteD65));
	real lo = rt_min(rgb.x(), rt_min(rgb.y(), rgb.z()));
	real hi = rt_max(rgb.x(), rt_max(rgb.y(), rgb.z()));
	check((double)(hi - lo) < 0.01,
		  "adaptation: an adapted flat spectrum is neutral",
		  (double)(hi - lo), 0.0, 0.01);
	check(std::abs((double)rgb.y() - 0.5) < 0.01,
		  "adaptation: and keeps its reflectance", (double)rgb.y(), 0.5, 0.01);
}

}  // namespace

void run_spectrum_tests() {
	std::printf("\nColour science checks\n");
	test_curves();
	test_equal_energy_is_neutral();
	test_monochromatic_locus();
	test_flat_spectrum_round_trip();
	test_chromatic_adaptation();
	test_dispersion();
}
