#include "renderer.h"

#include "rtweekend.h"
#include <vector>
#include <atomic>
#include <iostream>
#include <omp.h>

void Renderer::render(
    std::ostream& out,
    const hittable& world,
    const std::shared_ptr<hittable>& lights,
    const camera& cam,
    const color& background,
    const PathIntegrator& integrator,
    int image_width,
    int image_height,
    int samples_per_pixel,
    int max_depth
) const
{
    std::vector<color> framebuffer(image_width * image_height);
    std::atomic<int> rows_done = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < image_height; ++j) {

        for (int i = 0; i < image_width; ++i) {

            color pixel_color(0,0,0);

            for (int s = 0; s < samples_per_pixel; ++s) {

                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);

                ray r = cam.get_ray(u, v);

                pixel_color += integrator.Li(
                    r,
                    background,
                    world,
                    lights,
                    max_depth
                );
            }

            framebuffer[j * image_width + i] = pixel_color;
        }

        int done = ++rows_done;

        #pragma omp critical
        {
            std::cerr << "\rScanlines completed: "
                      << done << " / "
                      << image_height
                      << std::flush;
        }
    }

    std::cerr << "\nRendering finished.\n";

    // Write gamma-corrected image
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {

            color pixel_color =
                framebuffer[j * image_width + i];

            auto scale = 1.0 / samples_per_pixel;

            auto r_col = sqrt(scale * pixel_color.x());
            auto g_col = sqrt(scale * pixel_color.y());
            auto b_col = sqrt(scale * pixel_color.z());

            int ir = static_cast<int>(256 * clamp(r_col, 0.0, 0.999));
            int ig = static_cast<int>(256 * clamp(g_col, 0.0, 0.999));
            int ib = static_cast<int>(256 * clamp(b_col, 0.0, 0.999));

            out << ir << " "
                << ig << " "
                << ib << "\n";
        }
    }
}