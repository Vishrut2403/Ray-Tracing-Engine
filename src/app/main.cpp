#include "rtweekend.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "renderer/renderer.h"
#include "integrator/path_integrator.h"

#include "hittable_list.h"
#include "camera.h"
#include "bvh.h"
#include "constant_medium.h"
#include "sphere.h"

#include "xy_rect.h"
#include "xz_rect.h"
#include "yz_rect.h"
#include "flip_face.h"
#include "box.h"
#include "translate.h"
#include "rotate_y.h"

#include "material.h"
#include "diffuse_light.h"

int main(int argc, char** argv) {

    const double aspect_ratio = 1.0;
    const int image_width  = 600;
    const int image_height = 600;
    const int samples_per_pixel = 100;
    const int max_depth = 40;

    std::string filename = "cornell_volume_box2.ppm";

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
        std::cerr << "Error: Could not open file\n";
        return 1;
    }

    out << "P3\n"
        << image_width << " "
        << image_height << "\n255\n";

    // Scene construction (unchanged)
    hittable_list world;
    hittable_list lights;

    auto red   = std::make_shared<lambertian>(color(.65, .05, .05));
    auto white = std::make_shared<lambertian>(color(.73, .73, .73));
    auto green = std::make_shared<lambertian>(color(.12, .45, .15));
    auto light = std::make_shared<diffuse_light>(color(20, 20, 20));

    world.add(std::make_shared<yz_rect>(0,555,0,555,555, green));
    world.add(std::make_shared<flip_face>(
        std::make_shared<yz_rect>(0,555,0,555,0, red)));

    auto sphere_light =
        std::make_shared<sphere>(
            point3(278, 540, 278),
            30,
            light
        );

    world.add(sphere_light);
    lights.add(sphere_light);

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
        0.08,
        color(0,0,0)
    ));

    std::shared_ptr<hittable> box2 =
        std::make_shared<box>(
            point3(0,0,0),
            point3(165,165,165),
            white);

    box2 = std::make_shared<rotate_y>(box2, -18);
    box2 = std::make_shared<translate>(box2, vec3(130,0,65));

    world.add(box2);

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

    // Integrator + Renderer
    PathIntegrator integrator;
    Renderer renderer;

    renderer.render(
        out,
        world,
        lights_ptr,
        cam,
        background,
        integrator,
        image_width,
        image_height,
        samples_per_pixel,
        max_depth
    );

    out.close();

    std::cout << "Saved to: "
              << filepath.string() << "\n";

    return 0;
}