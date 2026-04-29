#pragma once

#include "core/vec3.h"

struct BSDFSample {
	vec3 wi;          
	color f;          
	double pdf;       
	bool is_delta;    
};