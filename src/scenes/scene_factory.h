#pragma once
#include "scenes/cornell_scene.h"
#include "scenes/furnace_scene.h"
#include "scenes/ggx_scene.h"

class SceneFactory {
public:
    static Scene build(const std::string& name) {
        if (name == "cornell")  return build_cornell_volume_scene();
        if (name == "furnace")  return build_furnace_scene();
        if (name == "ggx")      return build_ggx_scene();

        std::cerr << "Unknown scene '" << name << "'. Using cornell.\n";
        return build_cornell_volume_scene();
    }
};