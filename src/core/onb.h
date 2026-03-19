#ifndef ONB_H
#define ONB_H

#include "vec3.h"

class onb {
public:
    vec3 axis[3];

    HD onb() {}

    HD vec3 operator[](int i) const { return axis[i]; }
    HD vec3 u() const { return axis[0]; }
    HD vec3 v() const { return axis[1]; }
    HD vec3 w() const { return axis[2]; }

    HD vec3 local(double a, double b, double c) const { return a*u()+b*v()+c*w(); }
    HD vec3 local(const vec3& a) const { return a.x()*u()+a.y()*v()+a.z()*w(); }

    HD void build_from_w(const vec3& n) {
        axis[2] = unit_vector(n);
        vec3 a  = (fabs(w().x()) > 0.9) ? vec3(0,1,0) : vec3(1,0,0);
        axis[1] = unit_vector(cross(w(), a));
        axis[0] = cross(w(), v());
    }
};

#endif