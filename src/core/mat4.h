#pragma once

#include <cmath>
#include "core/vec3.h"

// Column-major, which is what glUniformMatrix4fv reads with transpose = FALSE.
struct mat4 {
	real m[16] = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};

	real&       operator()(int row, int col)       { return m[col*4 + row]; }
	const real& operator()(int row, int col) const { return m[col*4 + row]; }
};

inline mat4 operator*(const mat4& a, const mat4& b) {
	mat4 r;
	for (int col = 0; col < 4; ++col)
		for (int row = 0; row < 4; ++row) {
			real s = 0;
			for (int k = 0; k < 4; ++k) s += a(row,k) * b(k,col);
			r(row,col) = s;
		}
	return r;
}

inline mat4 mat4_transpose(const mat4& a) {
	mat4 r;
	for (int c = 0; c < 4; ++c)
		for (int row = 0; row < 4; ++row) r(row,c) = a(c,row);
	return r;
}

// w_out carries the clip-space w, which a projection needs and a view does not.
inline point3 mat4_mul_point(const mat4& a, const point3& p, real* w_out = nullptr) {
	real x = a(0,0)*p[0] + a(0,1)*p[1] + a(0,2)*p[2] + a(0,3);
	real y = a(1,0)*p[0] + a(1,1)*p[1] + a(1,2)*p[2] + a(1,3);
	real z = a(2,0)*p[0] + a(2,1)*p[1] + a(2,2)*p[2] + a(2,3);
	if (w_out)
		*w_out = a(3,0)*p[0] + a(3,1)*p[1] + a(3,2)*p[2] + a(3,3);
	return point3(x, y, z);
}

inline vec3 mat4_mul_dir(const mat4& a, const vec3& v) {
	return vec3(a(0,0)*v[0] + a(0,1)*v[1] + a(0,2)*v[2],
				a(1,0)*v[0] + a(1,1)*v[1] + a(1,2)*v[2],
				a(2,0)*v[0] + a(2,1)*v[1] + a(2,2)*v[2]);
}

// Right-handed: the camera ends up at the origin looking down -z, the
// convention the projection below and OpenGL's clip volume both assume.
inline mat4 mat4_look_at(const point3& eye, const point3& center, const vec3& up) {
	vec3 f = unit_vector(center - eye);
	vec3 s = unit_vector(cross(f, up));
	vec3 u = cross(s, f);

	mat4 r;
	r(0,0) = s[0]; r(0,1) = s[1]; r(0,2) = s[2]; r(0,3) = -dot(s, eye);
	r(1,0) = u[0]; r(1,1) = u[1]; r(1,2) = u[2]; r(1,3) = -dot(u, eye);
	r(2,0) =-f[0]; r(2,1) =-f[1]; r(2,2) =-f[2]; r(2,3) =  dot(f, eye);
	r(3,0) = 0;    r(3,1) = 0;    r(3,2) = 0;    r(3,3) =  1;
	return r;
}

// Maps the frustum onto OpenGL's [-1,1] clip cube, near plane to -1.
inline mat4 mat4_perspective(real vfov_degrees, real aspect, real znear, real zfar) {
	const real deg2rad = (real)3.14159265358979323846 / (real)180;
	real t = (real)1 / std::tan(vfov_degrees * deg2rad / (real)2);

	mat4 r;
	r(0,0) = t / aspect; r(1,1) = t;
	r(2,2) = (zfar + znear) / (znear - zfar);
	r(2,3) = (real)2 * zfar * znear / (znear - zfar);
	r(3,2) = -1;
	r(3,3) = 0;
	return r;
}

// Same framing as the perspective above at the plane being looked at, so
// toggling between them keeps the subject the same size.
inline mat4 mat4_orthographic(real half_height, real aspect, real znear, real zfar) {
	real half_width = half_height * aspect;

	mat4 r;
	r(0,0) = (real)1 / half_width;
	r(1,1) = (real)1 / half_height;
	r(2,2) = (real)-2 / (zfar - znear);
	r(2,3) = -(zfar + znear) / (zfar - znear);
	return r;
}

inline void mat4_to_float(const mat4& a, float out[16]) {
	for (int i = 0; i < 16; ++i) out[i] = (float)a.m[i];
}
