#include "rtweekend.h"
#include <filesystem>

#include <fstream>
#include <iostream>
#include <string>

#include "hittable_list.h"
#include "camera.h"
#include "bvh.h"
#include "core/interval.h"

#include "xy_rect.h"
#include "xz_rect.h"
#include "yz_rect.h"
#include "flip_face.h"
#include "box.h"
#include "translate.h"
#include "rotate_y.h"

#include "material.h"
#include "diffuse_light.h"

//////////////////////////////////////////////////////////////
// Path Tracing Integrator (Book 2 Architecture)
//////////////////////////////////////////////////////////////

color ray_color(
    const ray& r,
    const color& background,
    const hittable& world,
    int depth
) {
    hit_record rec;

    if (depth <= 0)
        return color(0,0,0);

    if (!world.hit(r, interval(0.001, infinity), rec))
        return background;

    color emitted =
        rec.mat_ptr->emitted(
            r, rec, rec.u, rec.v, rec.p);

    scatter_record srec;

    if (!rec.mat_ptr->scatter(r, rec, srec))
        return emitted;

    // Delta distribution (metal / dielectric)
    if (srec.is_specular) {
        return srec.attenuation *
               ray_color(
                   srec.specular_ray,
                   background,
                   world,
                   depth - 1);
    }

    // PDF-based scattering (Lambertian)
    ray scattered(
        rec.p,
        srec.pdf_ptr->generate(),
        r.time()
    );

    double pdf_val =
        srec.pdf_ptr->value(
            scattered.direction()
        );

    // Prevent division by zero
    if (pdf_val <= 0)
        return emitted;

    return emitted +
           srec.attenuation *
           rec.mat_ptr->scattering_pdf(
               r, rec, scattered
           ) *
           ray_color(
               scattered,
               background,
               world,
               depth - 1
           ) / pdf_val;
}

//////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

    // Render Parameters
    const double aspect_ratio = 1.0;
    const int image_width  = 400;
    const int image_height = 400;
    const int samples_per_pixel = 10;
    const int max_depth = 20;

    // Output file
    // ============================================================
    // Output File Setup (CLI + Robust Path Handling)
    // ============================================================

    std::string filename = "cornell.ppm";

    // CLI filename override
    if (argc > 1) {
        filename = argv[1];

        // Auto-append .ppm if missing
        if (filename.size() < 4 ||
            filename.substr(filename.size() - 4) != ".ppm") {
            filename += ".ppm";
        }
    }

    namespace fs = std::filesystem;

    // We are executing from the build directory
    fs::path build_dir = fs::current_path();

    // Project root is one level above build/
    fs::path project_root = build_dir.parent_path();

    // Target directory: renders/book2
    fs::path render_dir = project_root / "renders" / "book2";

    // Create directory if it doesn't exist
    fs::create_directories(render_dir);

    // Final output path
    fs::path filepath = render_dir / filename;

    // Open file
    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "Error: Could not open file: "
                << filepath << "\n";
        return 1;
    }

    // Write PPM header
    out << "P3\n"
        << image_width << " "
        << image_height << "\n255\n";

    //////////////////////////////////////////////////////////
    // Cornell Box Scene
    //////////////////////////////////////////////////////////

    hittable_list world;

    auto red   = std::make_shared<lambertian>(color(.65, .05, .05));
    auto white = std::make_shared<lambertian>(color(.73, .73, .73));
    auto green = std::make_shared<lambertian>(color(.12, .45, .15));
    auto light = std::make_shared<diffuse_light>(color(15, 15, 15));

    world.add(std::make_shared<yz_rect>(0,555,0,555,555, green));
    world.add(std::make_shared<flip_face>(
        std::make_shared<yz_rect>(0,555,0,555,0, red)));

    world.add(std::make_shared<flip_face>(
        std::make_shared<xz_rect>(213,343,227,332,554, light)));

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

    std::shared_ptr<hittable> box2 =
        std::make_shared<box>(
            point3(0,0,0),
            point3(165,165,165),
            white);

    box2 = std::make_shared<rotate_y>(box2, -18);
    box2 = std::make_shared<translate>(box2, vec3(130,0,65));
    world.add(box2);

    // Wrap world in BVH
    world = hittable_list(
        std::make_shared<bvh_node>(
            world.objects,
            0,
            world.objects.size(),
            0.0,
            1.0
        )
    );

    //////////////////////////////////////////////////////////
    // Camera
    //////////////////////////////////////////////////////////

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

    //////////////////////////////////////////////////////////
    // Render Loop
    //////////////////////////////////////////////////////////

    for (int j = image_height - 1; j >= 0; --j) {

        std::cerr << "\rScanlines remaining: "
                  << j << " / " << image_height
                  << std::flush;

        for (int i = 0; i < image_width; ++i) {

            color pixel_color(0,0,0);

            for (int s = 0; s < samples_per_pixel; ++s) {

                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);

                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, background, world, max_depth);
            }

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