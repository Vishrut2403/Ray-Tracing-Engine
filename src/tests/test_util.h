#pragma once
#include <string>

extern int g_checks, g_failures;

void check(bool ok, const std::string& what, double got, double want, double tol);
void run_gpu_tests();
void run_medium_tests();
void run_render_tests();
