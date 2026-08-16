#ifndef RTWEEKEND_H
#define RTWEEKEND_H
 
#include "ray.h"
#include "vec3.h"
 
#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include "random.h"
 
using std::shared_ptr;
using std::make_shared;
 
const real infinity = std::numeric_limits<real>::infinity();
const real pi = 3.1415926535897932385;
 
inline real degrees_to_radians(real degrees) {
	return degrees * pi / 180.0;
}
 
inline real clamp(real x, real min, real max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

#endif