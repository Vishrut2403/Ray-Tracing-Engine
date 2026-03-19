#ifndef INTERVAL_H
#define INTERVAL_H

#include "vec3.h"

class interval {
public:
    double min, max;

    HD interval() : min(0), max(0) {}
    HD interval(double _min, double _max) : min(_min), max(_max) {}

    HD bool contains(double x)  const { return min <= x && x <= max; }
    HD bool surrounds(double x) const { return min <  x && x <  max; }
};

#endif