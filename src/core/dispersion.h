#pragma once

#include "core/vec3.h"

// Glass is specified by its refractive index at the sodium d line and its Abbe
// number, which is how every catalogue lists it. Cauchy's two-term form then
// follows in closed form, and reproduces the visible range to a few units in
// the fourth decimal -- well inside what the d/V pair itself pins down.
//
//   n(lambda) = A + B / lambda^2        (lambda in micrometres)
//   V         = (n_d - 1) / (n_F - n_C)

inline constexpr real kFraunhoferF = 0.4861;   // hydrogen F, micrometres
inline constexpr real kFraunhoferD = 0.5876;   // sodium d
inline constexpr real kFraunhoferC = 0.6563;   // hydrogen C

struct Dispersion {
	real a = 1.5;
	real b = 0;      // zero is a non-dispersive glass

	// An Abbe number of zero means the caller wants no dispersion at all; a
	// real glass runs from about 20 (dense flint) to 65 (crown).
	static Dispersion from_abbe(real n_d, real abbe) {
		Dispersion d;
		if (!(abbe > 0)) { d.a = n_d; d.b = 0; return d; }

		const real inv_f2 = 1 / (kFraunhoferF * kFraunhoferF);
		const real inv_c2 = 1 / (kFraunhoferC * kFraunhoferC);
		d.b = (n_d - 1) / (abbe * (inv_f2 - inv_c2));
		d.a = n_d - d.b / (kFraunhoferD * kFraunhoferD);
		return d;
	}

	// lambda in nanometres, which is what the renderer carries.
	real ior(real lambda_nm) const {
		if (b == 0) return a;
		real um = lambda_nm * (real)1e-3;
		return a + b / (um * um);
	}
};
