#pragma once
#include "core/camera.h"
#include "app/render_config.h"

class CameraFactory {
public:
    static camera build(const RenderConfig& config)
    {
        if (config.feature == "furnace") {
            return camera(
                point3(0, 0,  3),   // lookfrom
                point3(0, 0,  0),   // lookat
                vec3  (0, 1,  0),   // vup
                90.0,               // wide FOV
                double(config.width) / config.height,
                0.0,                // no depth of field
                3.0,                // focus dist
                0.0, 1.0
            );
        }

        return camera(
            point3(278, 278, -800),
            point3(278, 278,  0),
            vec3  (0,   1,    0),
            40.0,
            double(config.width) / config.height,
            0.0,
            10.0,
            0.0, 1.0
        );
    }
};