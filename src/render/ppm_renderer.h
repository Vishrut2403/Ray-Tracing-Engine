#pragma once

#include "render/framebuffer.h"
#include "core/camera.h"
#include "scenes/cornell_scene.h"
#include "photon/photon_map.h"

class PPMRenderer {
public:
    int    n_iterations;
    int    photons_per_iter;
    int    max_depth;  
    double initial_radius;
    double alpha;    

    PPMRenderer(int iterations    = 64,
                int photons       = 100000,
                int depth         = 10,
                double radius     = 15.0,
                double alpha_val  = 0.7)
        : n_iterations(iterations)
        , photons_per_iter(photons)
        , max_depth(depth)
        , initial_radius(radius)
        , alpha(alpha_val) {}

    void render(
        const Scene& scene,
        Framebuffer& fb,
        const camera& cam,
        const color& background
    );

private:
    void camera_pass(
        const ray& r,
        const color& beta,
        const std::shared_ptr<hittable>& world,
        const std::shared_ptr<hittable_list>& lights,
        const std::shared_ptr<env_light>& env,
        const color& background,
        int px, int py,
        VisiblePoint& vp
    );

    void photon_pass(
        const std::shared_ptr<hittable_list>& lights,
        const std::shared_ptr<hittable>& world,
        std::vector<VisiblePoint>& vps,
        PhotonHashGrid& grid,
        long long& total_photons
    );

    void accumulate_photon(
        const Photon& photon,
        std::vector<VisiblePoint>& vps,
        const PhotonHashGrid& grid
    );
};