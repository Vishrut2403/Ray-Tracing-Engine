#pragma once
#include "scenes/cornell_scene.h"

class SceneFactory {
public:
    static Scene build(const std::string& name) {
        if (name == "cornell")
            return build_cornell_volume_scene();

        std::cerr << "Unknown scene. Using cornell.\n";
        return build_cornell_volume_scene();
    }
};