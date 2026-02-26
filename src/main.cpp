#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include <iostream>
#include <memory>

color ray_color(const ray& r, const hittable& world, int depth) {

    if (depth <= 0)
        return color(0, 0, 0);

    hit_record rec;

    if (world.hit(r, 0.001, 1000.0, rec)) {

        vec3 target = rec.p + rec.normal + random_in_unit_sphere();

        return 0.5 * ray_color(
            ray(rec.p, target - rec.p),
            world,
            depth - 1
        );
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0)
         + t * color(0.5, 0.7, 1.0);
}

int main() {

    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    std::cout << "P3\n"
              << image_width << " " << image_height << "\n255\n";

    // World
    hittable_list world;

    world.add(std::make_shared<sphere>(point3(0, 0, -1), 0.5));
    world.add(std::make_shared<sphere>(point3(0, -100.5, -1), 100));

    // Camera
    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner =
        origin
        - horizontal/2
        - vertical/2
        - vec3(0, 0, focal_length);

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {

            auto u = double(i) / (image_width - 1);
            auto v = double(j) / (image_height - 1);

            ray r(origin,
                  lower_left_corner + u*horizontal + v*vertical - origin);

            color pixel_color = ray_color(r, world, 50);

            int ir = static_cast<int>(255.999 * pixel_color.x());
            int ig = static_cast<int>(255.999 * pixel_color.y());
            int ib = static_cast<int>(255.999 * pixel_color.z());

            std::cout << ir << " "
                      << ig << " "
                      << ib << "\n";
        }
    }
}