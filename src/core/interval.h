#ifndef INTERVAL_H
#define INTERVAL_H

#include "vec3.h"

class interval {
public:
	real min, max;

	HD interval() : min(0), max(0) {}
	HD interval(real _min, real _max) : min(_min), max(_max) {}

	HD bool contains(real x)  const { return min <= x && x <= max; }
	HD bool surrounds(real x) const { return min <  x && x <  max; }
};

#endif