#include "cornell_scene.h"

#include "hittables/sphere.h"
#include "hittables/xy_rect.h"
#include "hittables/xz_rect.h"
#include "hittables/yz_rect.h"
#include "hittables/flip_face.h"
#include "hittables/box.h"
#include "hittables/translate.h"
#include "hittables/rotate_y.h"

#include "materials/material.h"
#include "materials/diffuse_light.h"
#include "acceleration/bvh.h"

Scene build_cornell_volume_scene()
{
    Scene scene;

    hittable_list world_objects;
    auto lights = std::make_shared<hittable_list>();

    // Materials
    auto red   = std::make_shared<lambertian>(color(.65, .05, .05));
    auto white = std::make_shared<lambertian>(color(.73, .73, .73));
    auto green = std::make_shared<lambertian>(color(.12, .45, .15));
    auto light = std::make_shared<diffuse_light>(color(20,20,20));

    // Left wall (green)
    world_objects.add(
        std::make_shared<yz_rect>(0, 555, 0, 555, 555, green)
    );

    // Right wall (red)
    world_objects.add(
        std::make_shared<flip_face>(
            std::make_shared<yz_rect>(0, 555, 0, 555, 0, red)
        )
    );

    // Floor
    world_objects.add(
        std::make_shared<xz_rect>(0, 555, 0, 555, 0, white)
    );

    // Ceiling
    world_objects.add(
        std::make_shared<flip_face>(
            std::make_shared<xz_rect>(0, 555, 0, 555, 555, white)
        )
    );

    // Back wall
    world_objects.add(
        std::make_shared<flip_face>(
            std::make_shared<xy_rect>(0, 555, 0, 555, 555, white)
        )
    );

    auto raw_light =
        std::make_shared<xz_rect>(
            213, 343,
            227, 332,
            554,
            light
        );

    auto ceiling_light =
        std::make_shared<flip_face>(raw_light);

    world_objects.add(ceiling_light);

    // Add the RAW rect to the lights list
    lights->add(raw_light);

    // Tall box
    std::shared_ptr<hittable> box1 =
        std::make_shared<box>(
            point3(0,0,0),
            point3(165,330,165),
            white
        );

    box1 = std::make_shared<rotate_y>(box1, 15);
    box1 = std::make_shared<translate>(box1, vec3(265,0,295));
    world_objects.add(box1);

    // Short box
    std::shared_ptr<hittable> box2 =
        std::make_shared<box>(
            point3(0,0,0),
            point3(165,165,165),
            white
        );

    box2 = std::make_shared<rotate_y>(box2, -18);
    box2 = std::make_shared<translate>(box2, vec3(130,0,65));
    world_objects.add(box2);

    // Build BVH
    scene.world = std::make_shared<bvh_node>(
        world_objects.objects,
        0,
        world_objects.objects.size(),
        0.0,
        1.0
    );

    scene.lights = lights;

    return scene;
}