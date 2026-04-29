#include "image_writer.h"
#include <fstream>
#include <cmath>
#include <vector>
#include <iostream>
#include "core/rtweekend.h"

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 1
#include "external/miniz.h"
#include "external/tinyexr.h"

static inline double aces_filmic(double x) {
	const double a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
	return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

static constexpr double EXPOSURE      = 1.0;
static constexpr double FIREFLY_CLAMP = 20.0;

static inline color firefly_clamp(color c) {
	double lum = 0.2126*c.x() + 0.7152*c.y() + 0.0722*c.z();
	if (lum > FIREFLY_CLAMP && lum > 0.0) c = c * (FIREFLY_CLAMP / lum);
	return c;
}

static inline double gamma_encode(double x) {
	return std::pow(clamp(x, 0.0, 1.0), 1.0/2.2);
}

void ImageWriter::write_ppm(
	const std::string& path,
	const Framebuffer& fb
) {
	std::ofstream out(path);
	int W = fb.get_width(), H = fb.get_height();
	out << "P3\n" << W << " " << H << "\n255\n";

	for (int j = H-1; j >= 0; --j) {
		for (int i = 0; i < W; ++i) {
			color c = firefly_clamp(fb.get(i, j)) * EXPOSURE;
			int ir = (int)(256 * clamp(gamma_encode(aces_filmic(c.x())), 0.0, 0.999));
			int ig = (int)(256 * clamp(gamma_encode(aces_filmic(c.y())), 0.0, 0.999));
			int ib = (int)(256 * clamp(gamma_encode(aces_filmic(c.z())), 0.0, 0.999));
			out << ir << " " << ig << " " << ib << "\n";
		}
	}
}

void ImageWriter::write_exr(
	const std::string& path,
	const Framebuffer& fb
) {
	int W = fb.get_width(), H = fb.get_height();
	int N = W * H;

	std::vector<float> r_buf(N), g_buf(N), b_buf(N);

	for (int j = 0; j < H; ++j) {
		for (int i = 0; i < W; ++i) {
			int exr_idx = j * W + i;
			int fb_j    = (H - 1 - j);
			color c     = fb.get(i, fb_j);
			r_buf[exr_idx] = (float)c.x();
			g_buf[exr_idx] = (float)c.y();
			b_buf[exr_idx] = (float)c.z();
		}
	}

	EXRHeader header;
	InitEXRHeader(&header);
	EXRImage image;
	InitEXRImage(&image);

	image.num_channels = 3;
	float* image_ptr[3] = { b_buf.data(), g_buf.data(), r_buf.data() };
	image.images = (unsigned char**)image_ptr;
	image.width  = W;
	image.height = H;

	header.num_channels = 3;
	header.channels = new EXRChannelInfo[3];
	strncpy(header.channels[0].name, "B", 255);
	strncpy(header.channels[1].name, "G", 255);
	strncpy(header.channels[2].name, "R", 255);

	header.pixel_types         = new int[3];
	header.requested_pixel_types = new int[3];
	for (int i = 0; i < 3; ++i) {
		header.pixel_types[i]           = TINYEXR_PIXELTYPE_FLOAT;
		header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;
	}

	const char* err = nullptr;
	int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
	if (ret != TINYEXR_SUCCESS) {
		std::cerr << "[EXR] write failed: " << (err ? err : "unknown") << "\n";
		if (err) FreeEXRErrorMessage(err);
	} else {
		std::cerr << "[EXR] saved: " << path << "\n";
	}

	delete[] header.channels;
	delete[] header.pixel_types;
	delete[] header.requested_pixel_types;
}