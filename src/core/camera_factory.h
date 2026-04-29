#pragma once
#include "core/camera.h"
#include "app/render_config.h"

class CameraFactory {
public:
	static camera build(const RenderConfig& config) {

		if (config.feature == "furnace")
			return camera(point3(0,0,3), point3(0,0,0), vec3(0,1,0),
						  90.0, double(config.width)/config.height,
						  0.0, 3.0, 0.0, 1.0);

		if (config.feature == "ggx" || config.feature == "hdr")
			return camera(point3(0,0.3,7), point3(0,0.3,0), vec3(0,1,0),
						  65.0, double(config.width)/config.height,
						  0.0, 7.0, 0.0, 1.0);

		if (config.feature == "bunny")
			return camera(
				point3(0, 1.2, 3.5),
				point3(0, 0.8, 0),
				vec3(0, 1, 0),
				45.0,
				double(config.width)/config.height,
				0.0, 3.5, 0.0, 1.0
			);

		if (config.feature == "glass")
			return camera(point3(0, 1.2, 10), point3(0, 0.2, 0), vec3(0,1,0),
				60.0, double(config.width)/config.height,
				0.0, 10.0, 0.0, 1.0);

		if (config.feature == "caustics")
			return camera(point3(278,278,-800), point3(278,278,0), vec3(0,1,0),
				40.0, double(config.width)/config.height,
				0.0, 10.0, 0.0, 1.0);

		if (config.feature == "helmet")
			return camera(
				point3(0.0, 4.0, 0.5),
				point3(0.0, 0.0, 0.0),  
				vec3(0, 0, -1),
				45.0,
				double(config.width)/config.height,
				0.0, 4.0, 0.0, 1.0
			);

		if (config.feature == "sss")
    		return camera(point3(278,278,-800), point3(278,278,0), vec3(0,1,0),
                  40.0, double(config.width)/config.height,
                  0.0, 10.0, 0.0, 1.0
	);

		if (config.feature == "ppm")
			return camera(point3(278,278,-800), point3(278,278,0), vec3(0,1,0),
				40.0, double(config.width)/config.height,
				0.0, 10.0, 0.0, 1.0);

		return camera(point3(278,278,-800), point3(278,278,0), vec3(0,1,0),
					  40.0, double(config.width)/config.height,
					  0.0, 10.0, 0.0, 1.0);
	}
};