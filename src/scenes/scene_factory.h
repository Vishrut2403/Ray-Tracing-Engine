#pragma once
#include "scenes/cornell_scene.h"
#include "scenes/furnace_scene.h"
#include "scenes/ggx_scene.h"
#include "scenes/hdr_scene.h"
#include "scenes/bunny_scene.h"
#include "scenes/rough_glass_scene.h"
#include "scenes/caustics_scene.h"
#include "scenes/helmet_scene.h"
#include "scenes/ppm_scene.h"
#include "scenes/sss_scene.h"
#include "scenes/volume_scene.h"

class SceneFactory {
public:
	static Scene build(const std::string& name) {
		if (name == "cornell")  return build_cornell_volume_scene();
		if (name == "furnace")  return build_furnace_scene();
		if (name == "ggx")      return build_ggx_scene();
		if (name == "hdr")      return build_hdr_scene();
		if (name == "bunny")    return build_bunny_scene();
		if (name == "glass")    return build_rough_glass_scene();
		if (name == "caustics") return build_caustics_scene();
		if (name == "helmet") return build_helmet_scene();
		if (name == "ppm") return build_ppm_scene();
		if (name == "sss")      return build_sss_scene();
		if (name == "volume")   return build_volume_scene();

		std::cerr << "Unknown scene '" << name << "'. Using cornell.\n";
		return build_cornell_volume_scene();
	}
};