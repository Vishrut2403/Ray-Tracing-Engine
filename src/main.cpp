#include "rtweekend.h"

#include "hittable_list.h"
#include "camera.h"
#include "bvh.h"

#include "xz_rect.h"
#include "diffuse_light.h"

#include <iostream>

color ray_color(const ray& r, const hittable& world, int depth) {

    if (depth <= 0)
        return color(0,0,0);

    hit_record rec;

    if (world.hit(r, 0.001, infinity, rec)) {

        ray scattered;
        color attenuation;

        color emitted =
            rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

        if (!rec.mat_ptr->scatter(
                r, rec, attenuation, scattered))
            return emitted;

        return emitted +
               attenuation *
               ray_color(scattered, world, depth - 1);
    }

    return color(0,0,0);
}

int main() {

    //Image Settings 

    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 800;
    const int image_height =
        static_cast<int>(image_width / aspect_ratio);

    const int samples_per_pixel = 200;
    const int max_depth = 50;

    std::cout << "P3\n"
              << image_width << " "
              << image_height << "\n255\n";

    // World 
    hittable_list world;

    auto light =
        std::make_shared<diffuse_light>(
            color(4,4,4));

    world.add(
        std::make_shared<xz_rect>(
            -2, 2,   // x0, x1
            -2, 2,   // z0, z1
            2,       // y
            light
        )
    );

    // Wrap world with BVH
    world = hittable_list(
        std::make_shared<bvh_node>(
            world.objects,
            0,
            world.objects.size(),
            0.0,
            1.0
        )
    );

    //Camera

    point3 lookfrom(0,4,5);
    point3 lookat(0,2,0);
    vec3 vup(0,1,0);

    auto dist_to_focus = 10.0;
    auto aperture = 0.0;

    camera cam(
        lookfrom,
        lookat,
        vup,
        20,
        aspect_ratio,
        aperture,
        dist_to_focus,
        0.0,
        1.0
    );

    //Render 

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {

            color pixel_color(0,0,0);

            for (int s = 0; s < samples_per_pixel; ++s) {

                auto u =
                    (i + random_double())
                    / (image_width - 1);

                auto v =
                    (j + random_double())
                    / (image_height - 1);

                ray r = cam.get_ray(u, v);

                pixel_color +=
                    ray_color(r, world, max_depth);
            }

            auto scale = 1.0 / samples_per_pixel;

            auto r_col = sqrt(scale * pixel_color.x());
            auto g_col = sqrt(scale * pixel_color.y());
            auto b_col = sqrt(scale * pixel_color.z());

            int ir = static_cast<int>(
                256 * clamp(r_col, 0.0, 0.999));
            int ig = static_cast<int>(
                256 * clamp(g_col, 0.0, 0.999));
            int ib = static_cast<int>(
                256 * clamp(b_col, 0.0, 0.999));

            std::cout << ir << " "
                      << ig << " "
                      << ib << "\n";
        }
    }
}