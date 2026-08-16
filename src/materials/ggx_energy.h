#pragma once

// Kulla-Conty energy compensation for the single-scattering GGX lobe.
// Smith GGX drops the energy that would have bounced a second time between
// microfacets; at roughness 1 that is ~65% of it. This adds it back as a
// separate lobe. Shared by both backends so they stay identical.

#include "core/vec3.h"
#include "materials/ggx_energy_table.h"

// Device code cannot address a plain host array, so instantiate one of each and
// pick per compilation pass.
#ifdef __CUDACC__
__device__ static const float GGX_E_DEV[GGX_E_N][GGX_E_N] = GGX_E_DATA;
__device__ static const float GGX_E_AVG_DEV[GGX_E_N]      = GGX_E_AVG_DATA;
#endif
static const float GGX_E_HST[GGX_E_N][GGX_E_N] = GGX_E_DATA;
static const float GGX_E_AVG_HST[GGX_E_N]      = GGX_E_AVG_DATA;

#if defined(__CUDA_ARCH__)
#  define GGX_E_TAB     GGX_E_DEV
#  define GGX_E_AVG_TAB GGX_E_AVG_DEV
#else
#  define GGX_E_TAB     GGX_E_HST
#  define GGX_E_AVG_TAB GGX_E_AVG_HST
#endif

#ifndef GGX_E_PI
#define GGX_E_PI 3.1415926535897932385
#endif

HD inline float ggx_E_at(float mu, float rough) {
	float fm = mu    * (GGX_E_N - 1);
	float fr = rough * (GGX_E_N - 1);
	fm = fm < 0.0f ? 0.0f : (fm > GGX_E_N - 1 ? GGX_E_N - 1 : fm);
	fr = fr < 0.0f ? 0.0f : (fr > GGX_E_N - 1 ? GGX_E_N - 1 : fr);
	int   i0 = (int)fm, j0 = (int)fr;
	int   i1 = i0 + 1 < GGX_E_N ? i0 + 1 : i0;
	int   j1 = j0 + 1 < GGX_E_N ? j0 + 1 : j0;
	float ti = fm - i0, tj = fr - j0;
	float a = GGX_E_TAB[j0][i0] + (GGX_E_TAB[j0][i1] - GGX_E_TAB[j0][i0]) * ti;
	float b = GGX_E_TAB[j1][i0] + (GGX_E_TAB[j1][i1] - GGX_E_TAB[j1][i0]) * ti;
	return a + (b - a) * tj;
}

HD inline float ggx_E_avg_at(float rough) {
	float fr = rough * (GGX_E_N - 1);
	fr = fr < 0.0f ? 0.0f : (fr > GGX_E_N - 1 ? GGX_E_N - 1 : fr);
	int   j0 = (int)fr;
	int   j1 = j0 + 1 < GGX_E_N ? j0 + 1 : j0;
	float t  = fr - j0;
	return GGX_E_AVG_TAB[j0] + (GGX_E_AVG_TAB[j1] - GGX_E_AVG_TAB[j0]) * t;
}

// Cosine-weighted average Schlick Fresnel: f0 + (1 - f0)/21.
HD inline vec3 ggx_F_avg(const vec3& f0) {
	return f0 + (vec3(1,1,1) - f0) * (1.0/21.0);
}

// Kulla-Conty multiple-scattering lobe, without the trailing cosine.
HD inline vec3 ggx_ms_brdf(const vec3& f0, real rough,
						    real ndotv, real ndotl) {
	real Ea = (real)ggx_E_avg_at((float)rough);
	real om = 1.0 - Ea;
	if (om < 1e-4) return vec3(0,0,0);

	real Ev = (real)ggx_E_at((float)ndotv, (float)rough);
	real El = (real)ggx_E_at((float)ndotl, (float)rough);

	vec3   Fa = ggx_F_avg(f0);
	real scale = (1.0 - Ev) * (1.0 - El) / (GGX_E_PI * om);

	vec3 Fms;
	for (int c = 0; c < 3; ++c) {
		real fa = Fa[c];
		real d  = 1.0 - fa * om;
		Fms[c] = d > 1e-9 ? fa * fa * Ea / d : 0.0;
	}
	return Fms * scale;
}
