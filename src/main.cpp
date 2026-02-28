#include "rtweekend.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <atomic>
#include <algorithm>
#include "hittable_list.h"
#include "hittable_pdf.h"
#include "mixture_pdf.h"
#include "camera.h"
#include "bvh.h"
#include "core/interval.h"
#include "constant_medium.h"

#include "xy_rect.h"
#include "xz_rect.h"
#include "yz_rect.h"
#include "flip_face.h"
#include "box.h"
#include "translate.h"
#include "rotate_y.h"

#include "material.h"
#include "diffuse_light.h"

color ray_color(
    const ray& r,
    const color& background,
    const hittable& world,
    const shared_ptr<hittable>& lights,
    int depth
) {
    hit_record rec;

    if (depth <= 0)
        return color(0,0,0);

    if (!world.hit(r, interval(0.001, infinity), rec))
        return background;

    color emitted =
        rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

    scatter_record srec;

    if (!rec.mat_ptr->scatter(r, rec, srec))
        return emitted;

    if (srec.is_specular) {
        return emitted +
               srec.attenuation *
               ray_color(
                   srec.specular_ray,
                   background,
                   world,
                   lights,
                   depth - 1
               );
    }

    if (depth < 5) {
        // Always survive early bounces
    } else {
            double max_component =
                std::max({ srec.attenuation.x(),
                    srec.attenuation.y(),
                    srec.attenuation.z() });

            double survival_prob =
                std::min(0.95, max_component);

        if (random_double() > survival_prob)
            return emitted;

        srec.attenuation /= survival_prob;
    }

    // ------------------------------
    // Light sampling (MIS)
    // ------------------------------
    auto light_pdf =
        make_shared<hittable_pdf>(lights, rec.p);

    mixture_pdf mixed_pdf(light_pdf, srec.pdf_ptr);

    ray scattered(
        rec.p,
        mixed_pdf.generate(),
        r.time()
    );

    double pdf_val =
        mixed_pdf.value(scattered.direction());

    if (pdf_val <= 1e-8)
        return emitted;

    double scattering_pdf =
        rec.mat_ptr->scattering_pdf(
            r, rec, scattered);

    return emitted +
           srec.attenuation *
           scattering_pdf *
           ray_color(
               scattered,
               background,
               world,
               lights,
               depth - 1
           ) / pdf_val;
}


int main(int argc, char** argv) {

    std::cout << "Max Threads: "
              << omp_get_max_threads()
              << "\n";

    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Threads in parallel region: "
                  << omp_get_num_threads()
                  << "\n";
    }

    const double aspect_ratio = 1.0;
    const int image_width  = 800;
    const int image_height = 800;
    const int samples_per_pixel = 800;
    const int max_depth = 50;  

    std::string filename = "cornell.ppm";

    if (argc > 1) {
        filename = argv[1];
        if (filename.size() < 4 ||
            filename.substr(filename.size() - 4) != ".ppm") {
            filename += ".ppm";
        }
    }

    namespace fs = std::filesystem;

    fs::path build_dir = fs::current_path();
    fs::path project_root = build_dir.parent_path();
    fs::path render_dir =
        project_root / "Ray-Tracing-Engine" / "renders" / "book3";

    fs::create_directories(render_dir);
    fs::path filepath = render_dir / filename;

    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "Error: Could not open file: "
                  << filepath << "\n";
        return 1;
    }

    out << "P3\n"
        << image_width << " "
        << image_height << "\n255\n";

    // Scene: Cornell Box with Participating Media

    hittable_list world;
    hittable_list lights;

    auto red   = std::make_shared<lambertian>(color(.65, .05, .05));
    auto white = std::make_shared<lambertian>(color(.73, .73, .73));
    auto green = std::make_shared<lambertian>(color(.12, .45, .15));
    auto light = std::make_shared<diffuse_light>(color(15, 15, 15));

    world.add(std::make_shared<yz_rect>(0,555,0,555,555, green));
    world.add(std::make_shared<flip_face>(
        std::make_shared<yz_rect>(0,555,0,555,0, red)));

    auto ceiling_light =
        std::make_shared<xz_rect>(213,343,227,332,554, light);

    world.add(std::make_shared<flip_face>(ceiling_light));
    lights.add(ceiling_light);

    world.add(std::make_shared<xz_rect>(0,555,0,555,0, white));
    world.add(std::make_shared<flip_face>(
        std::make_shared<xz_rect>(0,555,0,555,555, white)));
    world.add(std::make_shared<flip_face>(
        std::make_shared<xy_rect>(0,555,0,555,555, white)));

    std::shared_ptr<hittable> box1 =
        std::make_shared<box>(
            point3(0,0,0),
            point3(165,330,165),
            white);

    box1 = std::make_shared<rotate_y>(box1, 15);
    box1 = std::make_shared<translate>(box1, vec3(265,0,295));

    world.add(box1);

    world.add(std::make_shared<constant_medium>(
        box1,
        0.01,
        color(0,0,0)
    ));

    std::shared_ptr<hittable> box2 =
        std::make_shared<box>(
            point3(0,0,0),
            point3(165,165,165),
            white);

    box2 = std::make_shared<rotate_y>(box2, -18);
    box2 = std::make_shared<translate>(box2, vec3(130,0,65));

    world.add(std::make_shared<constant_medium>(
        box2,
        0.01,
        color(1,1,1)
    ));

    world = hittable_list(
        std::make_shared<bvh_node>(
            world.objects,
            0,
            world.objects.size(),
            0.0,
            1.0
        )
    );

    auto lights_ptr =
        std::make_shared<hittable_list>(lights);

    point3 lookfrom(278,278,-800);
    point3 lookat(278,278,0);
    vec3 vup(0,1,0);

    camera cam(
        lookfrom,
        lookat,
        vup,
        40,
        aspect_ratio,
        0.0,
        10.0,
        0.0,
        1.0
    );

    color background(0,0,0);

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

                pixel_color += ray_color(
                    r,
                    background,
                    world,
                    lights_ptr,
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

    std::cerr << "\nRender complete.\n";
    out.close();

    std::cout << "Saved to: "
              << filepath.string() << "\n";

    return 0;
}