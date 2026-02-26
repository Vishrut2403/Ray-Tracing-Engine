#include "rtweekend.h"

#include "hittable_list.h"
#include "sphere.h"
#include "material.h"
#include "camera.h"

#include <iostream>

color ray_color(const ray& r, const hittable& world, int depth) {

    if (depth <= 0)
        return color(0,0,0);

    hit_record rec;

    if (world.hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;

        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);

        return color(0,0,0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0 - t)*color(1.0,1.0,1.0)
         + t*color(0.5,0.7,1.0);
}

int main() {

    // Image settings
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;
    const int max_depth = 50;

    std::cout << "P3\n"
              << image_width << " "
              << image_height << "\n255\n";

    // World
    hittable_list world;

    auto material_ground = std::make_shared<lambertian>(
        color(0.8, 0.8, 0.0));

    auto material_center = std::make_shared<dielectric>(1.5);

    auto material_left = std::make_shared<metal>(
        color(0.8, 0.8, 0.8), 0.0);

    auto material_right = std::make_shared<metal>(
        color(0.8, 0.6, 0.2), 0.3);

    world.add(std::make_shared<sphere>(
        point3(0,-100.5,-1), 100, material_ground));

    // Glass sphere (outer)
    world.add(std::make_shared<sphere>(
        point3(0,0,-1), 0.5, material_center));

    // Hollow inner sphere
    world.add(std::make_shared<sphere>(
        point3(0,0,-1), -0.45, material_center));

    world.add(std::make_shared<sphere>(
        point3(-1,0,-1), 0.5, material_left));

    world.add(std::make_shared<sphere>(
        point3(1,0,-1), 0.5, material_right));

    // Camera (Depth of Field)
    point3 lookfrom(3,3,2);
    point3 lookat(0,0,-1);
    vec3 vup(0,1,0);

    auto dist_to_focus = (lookfrom - lookat).length();
    auto aperture = 2.0;

    camera cam(
        lookfrom,
        lookat,
        vup,
        20,
        aspect_ratio,
        aperture,
        dist_to_focus
    );

    // Render
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {

            color pixel_color(0,0,0);

            for (int s = 0; s < samples_per_pixel; ++s) {

                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);

                ray r = cam.get_ray(u, v);

                pixel_color += ray_color(r, world, max_depth);
            }

            auto scale = 1.0 / samples_per_pixel;

            auto r_col = sqrt(scale * pixel_color.x());
            auto g_col = sqrt(scale * pixel_color.y());
            auto b_col = sqrt(scale * pixel_color.z());

            int ir = static_cast<int>(256 * clamp(r_col, 0.0, 0.999));
            int ig = static_cast<int>(256 * clamp(g_col, 0.0, 0.999));
            int ib = static_cast<int>(256 * clamp(b_col, 0.0, 0.999));

            std::cout << ir << " "
                      << ig << " "
                      << ib << "\n";
        }
    }
}