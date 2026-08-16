#pragma once

#include "cuda/gpu_env_map.cuh"
#include "lights/env_light.h"
#include <vector>
#include <cmath>
#include <cstdio>

inline GpuEnvMap* upload_env_map(const env_light& env) {
	auto* hdr = env.tex.get();
	if (!hdr || hdr->get_width() == 0) return nullptr;

	int W = hdr->get_width();
	int H = hdr->get_height();
	float* cpu_pixels = hdr->get_data();

	std::vector<float> pixel_pdf(W * H);
	std::vector<float> marginal_cdf(H + 1, 0.0f);
	std::vector<float> cond_cdf((size_t)H * (W + 1), 0.0f);

	const real pi = 3.14159265358979;

	for (int j = 0; j < H; ++j) {
		real theta   = pi * (j + 0.5) / H;
		real sin_t   = sin(theta);
		real row_sum = 0.0;
		cond_cdf[j * (W+1) + 0] = 0.0f;

		for (int i = 0; i < W; ++i) {
			float* px  = cpu_pixels + (j * W + i) * 3;
			real lum = 0.2126*px[0] + 0.7152*px[1] + 0.0722*px[2];
			real w   = lum * sin_t;
			pixel_pdf[j * W + i] = (float)w;
			row_sum += w;
			cond_cdf[j * (W+1) + i + 1] = (float)row_sum;
		}

		if (row_sum > 0.0) {
			for (int i = 1; i <= W; ++i)
				cond_cdf[j * (W+1) + i] /= (float)row_sum;
		}

		marginal_cdf[j + 1] = marginal_cdf[j] + (float)row_sum;
	}

	float total = marginal_cdf[H];
	if (total > 0.0f) {
		for (int j = 0; j <= H; ++j) marginal_cdf[j] /= total;
		for (auto& p : pixel_pdf) p /= total;
	}

	GpuEnvMap* d_env;
	cudaMalloc(&d_env, sizeof(GpuEnvMap));

	GpuEnvMap host_env;
	host_env.width  = W;
	host_env.height = H;
	host_env.valid  = true;

	size_t pixel_bytes    = (size_t)W * H * 3 * sizeof(float);
	size_t marginal_bytes = (size_t)(H + 1)   * sizeof(float);
	size_t cond_bytes     = (size_t)H * (W+1) * sizeof(float);
	size_t pdf_bytes      = (size_t)W * H     * sizeof(float);

	cudaMalloc(&host_env.d_pixels,       pixel_bytes);
	cudaMalloc(&host_env.d_marginal_cdf, marginal_bytes);
	cudaMalloc(&host_env.d_cond_cdf,     cond_bytes);
	cudaMalloc(&host_env.d_pixel_pdf,    pdf_bytes);

	cudaMemcpy(host_env.d_pixels,       cpu_pixels,          pixel_bytes,    cudaMemcpyHostToDevice);
	cudaMemcpy(host_env.d_marginal_cdf, marginal_cdf.data(), marginal_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(host_env.d_cond_cdf,     cond_cdf.data(),     cond_bytes,     cudaMemcpyHostToDevice);
	cudaMemcpy(host_env.d_pixel_pdf,    pixel_pdf.data(),    pdf_bytes,      cudaMemcpyHostToDevice);

	cudaMemcpy(d_env, &host_env, sizeof(GpuEnvMap), cudaMemcpyHostToDevice);

	printf("[GpuEnvMap] uploaded %dx%d HDR (%.1f MB)\n",
		   W, H, (pixel_bytes + marginal_bytes + cond_bytes + pdf_bytes) / 1e6f);

	return d_env;
}