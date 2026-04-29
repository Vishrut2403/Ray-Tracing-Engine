#pragma once

#include "hittables/hittable.h"
#include "textures/hdr_texture.h"
#include "core/onb.h"
#include <vector>
#include <cmath>
#include <memory>

class env_light : public hittable {
public:
	std::shared_ptr<hdr_texture> tex;

	env_light(std::shared_ptr<hdr_texture> t) : tex(t) {
		build_cdf();
	}

	virtual bool hit(const ray&, const interval&, hit_record&) const override {
		return false;
	}
	virtual bool bounding_box(double, double, aabb&) const override {
		return false;
	}

	virtual vec3 random(const point3&) const override {
		double r1 = random_double();
		double r2 = random_double();

		int row = lower_bound(marginal_cdf, r1);
		row = clamp_idx(row, tex->get_height() - 1);

		int col = lower_bound(conditional_cdf[row], r2);
		col = clamp_idx(col, tex->get_width() - 1);

		double u   = (col + 0.5) / tex->get_width();
		double v   = (row + 0.5) / tex->get_height();
		double phi = 2.0 * pi * u;
		double theta = pi * (1.0 - v);

		return vec3(
			sin(theta) * cos(phi),
			cos(theta),
			sin(theta) * sin(phi)
		);
	}

	virtual double pdf_value(const point3&, const vec3& dir) const override {
		vec3 d = unit_vector(dir);
		double u = 0.5 + atan2(d.z(), d.x()) / (2.0 * pi);
		double v = 0.5 + asin(clamp(d.y(), -1.0, 1.0)) / pi;

		int col = clamp_idx((int)(u * tex->get_width()),  tex->get_width()  - 1);
		int row = clamp_idx((int)(v * tex->get_height()), tex->get_height() - 1);

		double sin_theta = sqrt(1.0 - d.y() * d.y());
		if (sin_theta < 1e-8) return 0.0;

		double p = pixel_pdf[row * tex->get_width() + col];
		return p / (sin_theta * 2.0 * pi * pi);
	}

	color Le(const vec3& dir) const {
		return tex->sample_dir(dir);
	}

private:
	std::vector<double>              marginal_cdf;
	std::vector<std::vector<double>> conditional_cdf;
	std::vector<double>              pixel_pdf;

	void build_cdf() {
		int W = tex->get_width();
		int H = tex->get_height();
		float* data = tex->get_data();

		pixel_pdf.resize(W * H);
		conditional_cdf.resize(H);
		marginal_cdf.resize(H + 1, 0.0);

		for (int j = 0; j < H; ++j) {
			double theta   = pi * (j + 0.5) / H;
			double sin_t   = sin(theta);
			double row_sum = 0.0;
			conditional_cdf[j].resize(W + 1, 0.0);

			for (int i = 0; i < W; ++i) {
				float* px = data + (j * W + i) * 3;
				double lum = 0.2126*px[0] + 0.7152*px[1] + 0.0722*px[2];
				double w   = lum * sin_t;
				pixel_pdf[j * W + i] = w;
				row_sum += w;
				conditional_cdf[j][i + 1] = row_sum;
			}

			if (row_sum > 0.0)
				for (int i = 1; i <= W; ++i)
					conditional_cdf[j][i] /= row_sum;

			marginal_cdf[j + 1] = marginal_cdf[j] + row_sum;
		}

		double total = marginal_cdf[H];
		if (total > 0.0) {
			for (int j = 0; j <= H; ++j)
				marginal_cdf[j] /= total;
			for (auto& p : pixel_pdf) p /= total;
		}
	}

	static int lower_bound(const std::vector<double>& cdf, double u) {
		int lo = 0, hi = (int)cdf.size() - 1;
		while (lo < hi) {
			int mid = (lo + hi) / 2;
			if (cdf[mid] < u) lo = mid + 1;
			else              hi = mid;
		}
		return lo > 0 ? lo - 1 : 0;
	}

	static int clamp_idx(int v, int max_v) {
		return v < 0 ? 0 : (v > max_v ? max_v : v);
	}
};