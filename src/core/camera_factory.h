#pragma once
#include "camera.h"
#include "app/render_config.h"

class CameraFactory {
public:
    static camera build(const RenderConfig& config)
    {
        point3 lookfrom(278,278,-800);
        point3 lookat(278,278,0);
        vec3 vup(0,1,0);

        return camera(
            lookfrom,
            lookat,
            vup,
            40,
            double(config.width)/config.height,
            0.0,
            10.0,
            0.0,
            1.0
        );
    }
};