#pragma once
#include "core/vec3.h"

struct GpuTriangle {
	vec3  v0, v1, v2;    
	vec3  n0, n1, n2;     
	int   mat_id;
};

struct GpuTriBVHNode {
	float aabb_min[3];
	float aabb_max[3];
	int   left;
	int   right;   
};