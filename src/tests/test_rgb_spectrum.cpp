// Upsampling checks. Turning an RGB albedo back into a spectrum is
// underdetermined, so what can be pinned is the round trip: integrate the
// spectrum the renderer will actually use, and the colour has to come back.

#include "tests/test_util.h"
#include "core/rgb_spectrum.h"
#include "core/spectrum.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

// What the renderer does to a reflectance, done densely: light it with an
// equal-energy illuminant, integrate against the CIE curves, convert to RGB.
color round_trip(const color& rgb) {
	vec3 xyz(0, 0, 0);
	for (real l = kLambdaMin; l <= kLambdaMax; l += 1)
		xyz += cie_xyz(l) * rgb_to_reflectance(rgb, l);
	return xyz_to_rgb(xyz / kCieYIntegral);
}

void test_round_trip() {
	// White under illuminant E is the reference the others divide by, which
	// takes the E-to-D65 cast out and leaves the upsampling error alone.
	color white = round_trip(color(1, 1, 1));

	struct { color rgb; const char* name; } cases[] = {
		{ color(1.00, 1.00, 1.00), "white"            },
		{ color(0.50, 0.50, 0.50), "grey"             },
		{ color(0.65, 0.05, 0.05), "cornell red"      },
		{ color(0.12, 0.45, 0.15), "cornell green"    },
		{ color(0.73, 0.73, 0.73), "cornell white"    },
		{ color(0.80, 0.60, 0.20), "gold"             },
		{ color(0.20, 0.40, 0.80), "blue"             },
	};

	for (const auto& c : cases) {
		color got = round_trip(c.rgb);
		double worst = 0.0;
		for (int k = 0; k < 3; ++k)
			worst = std::max(worst,
				std::abs((double)(got[k] / white[k]) - (double)c.rgb[k]));

		// Smits trades round-trip accuracy for smoothness; a few percent is
		// the price, and it is the reason Jakob and Hanika exists.
		check(worst < 0.05,
			  std::string("upsample ") + c.name + ": round trips to its RGB",
			  worst, 0.0, 0.05);
	}
}

void test_physical_bounds() {
	int out_of_range = 0, samples = 0;
	for (real r = 0; r <= 1.0001; r += 0.25)
		for (real g = 0; g <= 1.0001; g += 0.25)
			for (real b = 0; b <= 1.0001; b += 0.25)
				for (real l = kLambdaMin; l <= kLambdaMax; l += 5) {
					real v = rgb_to_reflectance(color(r,g,b), l);
					++samples;
					// A reflectance above one is a surface that emits.
					if (v < 0 || v > 1.02) ++out_of_range;
				}
	check(out_of_range == 0,
		  "upsample: reflectance stays within [0,1] everywhere",
		  (double)out_of_range, 0.0, 0.5);
	check(samples > 0, "upsample: the sweep ran", (double)samples, 1.0, 0.0);
}

// Grey is the case the furnace tests depend on, so it has to be exactly flat
// rather than approximately so.
void test_grey_is_flat() {
	for (double rho : { 0.18, 0.5, 0.73 }) {
		double lo = 1e30, hi = -1e30;
		for (real l = kLambdaMin; l <= kLambdaMax; l += 1) {
			double v = (double)rgb_to_reflectance(color(rho,rho,rho), l);
			lo = std::min(lo, v); hi = std::max(hi, v);
		}
		check(hi - lo < 0.002 && std::abs(lo - rho) < 0.002,
			  "upsample: grey gives a flat spectrum at its own value",
			  hi - lo, 0.0, 0.002);
	}
}

// RGB mode must be untouched: a zero wavelength passes the colour straight on.
void test_rgb_passthrough() {
	color c(0.65, 0.05, 0.05);
	check((spectral_albedo(c, vec3(0,0,0)) - c).length() == 0.0,
		  "upsample: zero wavelength leaves RGB alone",
		  (double)(spectral_albedo(c, vec3(0,0,0)) - c).length(), 0.0, 0.0);

	// Emission keeps its magnitude; only the hue goes through the bases.
	color e(15, 15, 15);
	color se = spectral_emission(e, vec3(550, 550, 550));
	check(std::abs((double)se.x() - 15.0) < 0.1,
		  "upsample: a neutral emitter keeps its radiance",
		  (double)se.x(), 15.0, 0.1);
}

}  // namespace

void run_rgb_spectrum_tests() {
	std::printf("\nSpectral upsampling checks\n");
	test_round_trip();
	test_physical_bounds();
	test_grey_is_flat();
	test_rgb_passthrough();
}
