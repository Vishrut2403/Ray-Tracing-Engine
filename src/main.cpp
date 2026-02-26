#include "rtweekend.h"

#include "hittable_list.h"
#include "sphere.h"
#include "moving_sphere.h"
#include "material.h"
#include "camera.h"
#include "bvh.h"

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

hittable_list random_scene() {
    hittable_list world;

    auto checker =
        std::make_shared<checker_texture>(
            color(0.2, 0.3, 0.1),
            color(0.9, 0.9, 0.9)
        );

    auto ground_material =
        std::make_shared<lambertian>(checker);

        world.add(std::make_shared<sphere>(
            point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {

            auto choose_mat = random_double();
            point3 center(
                a + 0.9*random_double(),
                0.2,
                b + 0.9*random_double()
            );

            if ((center - point3(4,0.2,0)).length() > 0.9) {

                std::shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // Diffuse (MOVING)
                    auto albedo =
                        color::random() * color::random();

                    sphere_material =
                        std::make_shared<lambertian>(albedo);

                    auto center2 =
                        center + vec3(0,
                                      random_double(0,0.5),
                                      0);

                    world.add(std::make_shared<moving_sphere>(
                        center,
                        center2,
                        0.0,
                        1.0,
                        0.2,
                        sphere_material));

                } else if (choose_mat < 0.95) {
                    // Metal (STATIC)
                    auto albedo =
                        color::random(0.5,1);
                    auto fuzz =
                        random_double(0,0.5);

                    sphere_material =
                        std::make_shared<metal>(albedo, fuzz);

                    world.add(std::make_shared<sphere>(
                        center, 0.2, sphere_material));

                } else {
                    // Glass (STATIC)
                    sphere_material =
                        std::make_shared<dielectric>(1.5);

                    world.add(std::make_shared<sphere>(
                        center, 0.2, sphere_material));
                }
            }
        }
    }

    // Three large spheres

    auto material1 =
        std::make_shared<dielectric>(1.5);

    world.add(std::make_shared<sphere>(
        point3(0,1,0), 1.0, material1));

    auto material2 =
        std::make_shared<lambertian>(
            color(0.4,0.2,0.1));

    world.add(std::make_shared<sphere>(
        point3(-4,1,0), 1.0, material2));

    auto material3 =
        std::make_shared<metal>(
            color(0.7,0.6,0.5), 0.0);

    world.add(std::make_shared<sphere>(
        point3(4,1,0), 1.0, material3));

    return world;
}

int main() {

    // Image settings
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 300;
    const int image_height =
        static_cast<int>(image_width / aspect_ratio);

    const int samples_per_pixel = 100;   // ↑ smoother blur
    const int max_depth = 50;

    std::cout << "P3\n"
              << image_width << " "
              << image_height << "\n255\n";

    auto world = random_scene();
    world = hittable_list(
        make_shared<bvh_node>(
            world.objects, 0,
            world.objects.size(),
            0.0, 1.0
        )
    );

    // Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);

    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(
        lookfrom,
        lookat,
        vup,
        20,
        aspect_ratio,
        aperture,
        dist_to_focus,
        0.0,   // shutter open
        1.0    // shutter close
    );

    // Render
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