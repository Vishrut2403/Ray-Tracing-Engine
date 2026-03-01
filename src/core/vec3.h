#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class vec3 {
public:
    double e[3];

    // Constructors
    vec3() : e{0.0, 0.0, 0.0} {}
    vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    // Accessors
    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    // Unary minus
    vec3 operator-() const {
        return vec3(-e[0], -e[1], -e[2]);
    }

    // Indexing
    double operator[](int i) const { return e[i]; }
    double& operator[](int i) { return e[i]; }

    // Compound operators
    vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    // Component-wise multiply (important for throughput)
    vec3& operator*=(const vec3& v) {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    vec3& operator/=(double t) {
        return *this *= (1.0 / t);
    }

    // Length
    double length() const {
        return std::sqrt(length_squared());
    }

    double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
};

// Type aliases
using point3 = vec3;
using color  = vec3;

///////////////////////////////////////////////////////////
// Non-member operators
///////////////////////////////////////////////////////////

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << " " << v.e[1] << " " << v.e[2];
}

inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(
        u.e[0] + v.e[0],
        u.e[1] + v.e[1],
        u.e[2] + v.e[2]
    );
}

inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(
        u.e[0] - v.e[0],
        u.e[1] - v.e[1],
        u.e[2] - v.e[2]
    );
}

// Component-wise multiply
inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(
        u.e[0] * v.e[0],
        u.e[1] * v.e[1],
        u.e[2] * v.e[2]
    );
}

inline vec3 operator*(double t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

inline vec3 operator/(const vec3& v, double t) {
    return (1.0 / t) * v;
}

///////////////////////////////////////////////////////////
// Math helpers
///////////////////////////////////////////////////////////

inline double dot(const vec3& u, const vec3& v) {
    return u.e[0]*v.e[0]
         + u.e[1]*v.e[1]
         + u.e[2]*v.e[2];
}

inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(
        u.e[1]*v.e[2] - u.e[2]*v.e[1],
        u.e[2]*v.e[0] - u.e[0]*v.e[2],
        u.e[0]*v.e[1] - u.e[1]*v.e[0]
    );
}

inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

///////////////////////////////////////////////////////////
// Reflection / Refraction
///////////////////////////////////////////////////////////

inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    double cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel =
        -std::sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// Schlick approximation
inline double reflectance(double cosine, double ref_idx) {
    double r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::pow((1 - cosine), 5);
}

#endif