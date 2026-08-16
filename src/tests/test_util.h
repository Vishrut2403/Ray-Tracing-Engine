#pragma once
#include <string>
#include <limits>
#include "core/vec3.h"

// Tolerances for checks that compare two algebraically identical expressions.
// They track the working precision instead of being fixed: at `real = float`
// a bound of 1e-6 is below the rounding floor and can never pass, while at
// double these keep the historical tightness.
inline constexpr double kEpsReal = (double)std::numeric_limits<real>::epsilon();
// Same quantity down two independent code paths, several operations deep.
inline constexpr double kNumericTol = 1000.0*kEpsReal > 1e-6 ? 1000.0*kEpsReal : 1e-6;
// A ratio that cancels exactly on paper, so only rounding separates the two.
inline constexpr double kExactTol   = 100.0*kEpsReal  > 1e-9 ? 100.0*kEpsReal  : 1e-9;

extern int g_checks, g_failures;

void check(bool ok, const std::string& what, double got, double want, double tol);
void run_gpu_tests();
void run_medium_tests();
void run_render_tests();
