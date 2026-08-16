#ifndef VEC3_H
#define VEC3_H

#include <cmath>

#ifdef __CUDACC__
#  define HD __host__ __device__
#else
#  define HD
#endif

// Scalar type for all geometry and shading. Single precision by default:
// FP64 runs at a small fraction of FP32 throughput on consumer GPUs, and the
// scene scales here have no need for it. Define RT_USE_DOUBLE to switch back.
#ifdef RT_USE_DOUBLE
using real = double;
#else
using real = float;
#endif

class vec3 {
public:
	real e[3];

	HD vec3() : e{0.0, 0.0, 0.0} {}
	HD vec3(real e0, real e1, real e2) : e{e0, e1, e2} {}

	HD real x() const { return e[0]; }
	HD real y() const { return e[1]; }
	HD real z() const { return e[2]; }

	HD vec3  operator-()          const { return vec3(-e[0], -e[1], -e[2]); }
	HD real  operator[](int i)  const { return e[i]; }
	HD real& operator[](int i)        { return e[i]; }

	HD vec3& operator+=(const vec3& v) {
		e[0]+=v.e[0]; e[1]+=v.e[1]; e[2]+=v.e[2]; return *this;
	}
	HD vec3& operator*=(real t) {
		e[0]*=t; e[1]*=t; e[2]*=t; return *this;
	}
	HD vec3& operator*=(const vec3& v) {
		e[0]*=v.e[0]; e[1]*=v.e[1]; e[2]*=v.e[2]; return *this;
	}
	HD vec3& operator/=(real t) { return *this *= (1.0/t); }

	HD real length()          const { return sqrt(length_squared()); }
	HD real length_squared()  const { return e[0]*e[0]+e[1]*e[1]+e[2]*e[2]; }
	HD real max_component()   const { return fmax(e[0], fmax(e[1], e[2])); }
};

using point3 = vec3;
using color  = vec3;

#ifndef __CUDA_ARCH__
#include <iostream>
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
	return out << v.e[0] << " " << v.e[1] << " " << v.e[2];
}
#endif

// Precision-agnostic min/max. std::fmin/fmax are constexpr in libstdc++ and
// nvcc will not call them from device code, while the float-only fminf/fmaxf
// silently round their arguments when `real` is double.
HD inline real rmin(real a, real b) { return a < b ? a : b; }
HD inline real rmax(real a, real b) { return a > b ? a : b; }

HD inline vec3   operator+(const vec3& u, const vec3& v) { return vec3(u.e[0]+v.e[0], u.e[1]+v.e[1], u.e[2]+v.e[2]); }
HD inline vec3   operator-(const vec3& u, const vec3& v) { return vec3(u.e[0]-v.e[0], u.e[1]-v.e[1], u.e[2]-v.e[2]); }
HD inline vec3   operator*(const vec3& u, const vec3& v) { return vec3(u.e[0]*v.e[0], u.e[1]*v.e[1], u.e[2]*v.e[2]); }
HD inline vec3   operator*(real t, const vec3& v)      { return vec3(t*v.e[0], t*v.e[1], t*v.e[2]); }
HD inline vec3   operator*(const vec3& v, real t)      { return t*v; }
HD inline vec3   operator/(const vec3& v, real t)      { return (1.0/t)*v; }
HD inline real dot(const vec3& u, const vec3& v)       { return u.e[0]*v.e[0]+u.e[1]*v.e[1]+u.e[2]*v.e[2]; }
HD inline vec3   cross(const vec3& u, const vec3& v) {
	return vec3(u.e[1]*v.e[2]-u.e[2]*v.e[1],
				u.e[2]*v.e[0]-u.e[0]*v.e[2],
				u.e[0]*v.e[1]-u.e[1]*v.e[0]);
}
HD inline vec3   unit_vector(const vec3& v)               { return v/v.length(); }
HD inline vec3   reflect(const vec3& v, const vec3& n)    { return v - 2*dot(v,n)*n; }
HD inline vec3   refract(const vec3& uv, const vec3& n, real eta) {
	real d     = dot(-uv,n);
	real cos_t = d < (real)1.0 ? d : (real)1.0;
	vec3 perp  =  eta*(uv + cos_t*n);
	vec3 par   = -sqrt(fabs((real)1.0 - perp.length_squared()))*n;
	return perp + par;
}
HD inline real reflectance(real cosine, real ref_idx) {
	real r0 = (1-ref_idx)/(1+ref_idx); r0*=r0;
	real c  = 1-cosine;
	return r0 + (1-r0)*c*c*c*c*c;
}

#endif