#pragma once
#include "core/camera.h"
#include "app/render_config.h"

class CameraFactory {
public:
    static camera build(const RenderConfig& config) {

        if (config.feature == "furnace") {
            return camera(
                point3(0, 0,  3), point3(0, 0, 0), vec3(0,1,0),
                90.0, double(config.width)/config.height,
                0.0, 3.0, 0.0, 1.0
            );
        }

        if (config.feature == "ggx") {
            return camera(
                point3(0, 0.3, 7), point3(0, 0.3, 0), vec3(0,1,0),
                65.0, double(config.width)/config.height,
                0.0, 7.0, 0.0, 1.0
            );
        }

        if (config.feature == "hdr") {
            return camera(
                point3(0, 0.3, 7), point3(0, 0.3, 0), vec3(0,1,0),
                65.0, double(config.width)/config.height,
                0.0, 7.0, 0.0, 1.0
            );
        }

        // Default — Cornell box
        return camera(
            point3(278, 278, -800), point3(278, 278, 0), vec3(0,1,0),
            40.0, double(config.width)/config.height,
            0.0, 10.0, 0.0, 1.0
        );
    }
};