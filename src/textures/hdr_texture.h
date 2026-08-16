#pragma once

#include "core/rtweekend.h"
#include <iostream>  
#include "textures/texture.h"
#include "external/stb_image.h"

class hdr_texture : public texture {
public:
	hdr_texture(const char* filename) {
		data = stbi_loadf(filename, &width, &height, &channels, 3);
		if (!data) {
			std::cerr << "[hdr_texture] failed to load: " << filename << "\n";
			width = height = 0;
		}
	}

	~hdr_texture() {
		stbi_image_free(data);
	}

	virtual color value(real u, real v, const point3&) const override {
		if (!data) return color(0, 1, 1);
		u = clamp(u, 0.0, 1.0);
		v = clamp(1.0 - v, 0.0, 1.0);

		int i = std::min<real>((int)(u * width),  width  - 1);
		int j = std::min<real>((int)(v * height), height - 1);

		float* px = data + (j * width + i) * 3;
		return color(px[0], px[1], px[2]);
	}

	color sample_dir(const vec3& dir) const {
		vec3 d = unit_vector(dir);
		real u = 0.5 + atan2(d.z(), d.x()) / (2.0 * pi);
		real v = 0.5 + asin(clamp(d.y(), -1.0, 1.0)) / pi;
		return value(u, v, point3(0,0,0));
	}

	int get_width()  const { return width;  }
	int get_height() const { return height; }
	float* get_data() const { return data;  }

private:
	float* data   = nullptr;
	int    width  = 0;
	int    height = 0;
	int    channels = 3;
};