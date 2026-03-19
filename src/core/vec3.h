#ifndef VEC3_H
#define VEC3_H

#include <cmath>

#ifdef __CUDACC__
#  define HD __host__ __device__
#else
#  define HD
#endif

class vec3 {
public:
    double e[3];

    HD vec3() : e{0.0, 0.0, 0.0} {}
    HD vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    HD double x() const { return e[0]; }
    HD double y() const { return e[1]; }
    HD double z() const { return e[2]; }

    HD vec3  operator-()          const { return vec3(-e[0], -e[1], -e[2]); }
    HD double  operator[](int i)  const { return e[i]; }
    HD double& operator[](int i)        { return e[i]; }

    HD vec3& operator+=(const vec3& v) {
        e[0]+=v.e[0]; e[1]+=v.e[1]; e[2]+=v.e[2]; return *this;
    }
    HD vec3& operator*=(double t) {
        e[0]*=t; e[1]*=t; e[2]*=t; return *this;
    }
    HD vec3& operator*=(const vec3& v) {
        e[0]*=v.e[0]; e[1]*=v.e[1]; e[2]*=v.e[2]; return *this;
    }
    HD vec3& operator/=(double t) { return *this *= (1.0/t); }

    HD double length()          const { return sqrt(length_squared()); }
    HD double length_squared()  const { return e[0]*e[0]+e[1]*e[1]+e[2]*e[2]; }
    HD double max_component()   const { return fmax(e[0], fmax(e[1], e[2])); }
};

using point3 = vec3;
using color  = vec3;

#ifndef __CUDA_ARCH__
#include <iostream>
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << " " << v.e[1] << " " << v.e[2];
}
#endif

HD inline vec3   operator+(const vec3& u, const vec3& v) { return vec3(u.e[0]+v.e[0], u.e[1]+v.e[1], u.e[2]+v.e[2]); }
HD inline vec3   operator-(const vec3& u, const vec3& v) { return vec3(u.e[0]-v.e[0], u.e[1]-v.e[1], u.e[2]-v.e[2]); }
HD inline vec3   operator*(const vec3& u, const vec3& v) { return vec3(u.e[0]*v.e[0], u.e[1]*v.e[1], u.e[2]*v.e[2]); }
HD inline vec3   operator*(double t, const vec3& v)      { return vec3(t*v.e[0], t*v.e[1], t*v.e[2]); }
HD inline vec3   operator*(const vec3& v, double t)      { return t*v; }
HD inline vec3   operator/(const vec3& v, double t)      { return (1.0/t)*v; }
HD inline double dot(const vec3& u, const vec3& v)       { return u.e[0]*v.e[0]+u.e[1]*v.e[1]+u.e[2]*v.e[2]; }
HD inline vec3   cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1]*v.e[2]-u.e[2]*v.e[1],
                u.e[2]*v.e[0]-u.e[0]*v.e[2],
                u.e[0]*v.e[1]-u.e[1]*v.e[0]);
}
HD inline vec3   unit_vector(const vec3& v)               { return v/v.length(); }
HD inline vec3   reflect(const vec3& v, const vec3& n)    { return v - 2*dot(v,n)*n; }
HD inline vec3   refract(const vec3& uv, const vec3& n, double eta) {
    double cos_t = fmin(dot(-uv,n), 1.0);
    vec3 perp  =  eta*(uv + cos_t*n);
    vec3 par   = -sqrt(fabs(1.0 - perp.length_squared()))*n;
    return perp + par;
}
HD inline double reflectance(double cosine, double ref_idx) {
    double r0 = (1-ref_idx)/(1+ref_idx); r0*=r0;
    return r0 + (1-r0)*pow((1-cosine),5.0);
}

#endif