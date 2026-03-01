#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "core/rtweekend.h"
#include "core/onb.h"
#include <memory>

inline void get_sphere_uv(
    const point3& p,
    double& u,
    double& v
) {
    auto theta = acos(-p.y());
    auto phi   = atan2(-p.z(), p.x()) + pi;

    u = phi / (2*pi);
    v = theta / pi;
}

class sphere : public hittable {
public:
    sphere() {}

    sphere(point3 cen, double r, std::shared_ptr<material> m)
        : center(cen), radius(r), mat_ptr(m) {}

    // ----------------------------------------------------
    // HIT
    // ----------------------------------------------------
    virtual bool hit(
        const ray& r,
        const interval& ray_t,
        hit_record& rec
    ) const override {

        vec3 oc = r.origin() - center;

        auto a = r.direction().length_squared();
        auto half_b = dot(oc, r.direction());
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = half_b*half_b - a*c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        auto root = (-half_b - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (-half_b + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);

        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;

        get_sphere_uv(outward_normal, rec.u, rec.v);

        return true;
    }

    // ----------------------------------------------------
    // BOUNDING BOX
    // ----------------------------------------------------
    virtual bool bounding_box(
        double time0,
        double time1,
        aabb& output_box
    ) const override {

        output_box = aabb(
            center - vec3(radius, radius, radius),
            center + vec3(radius, radius, radius)
        );

        return true;
    }

    // ----------------------------------------------------
    // SOLID ANGLE PDF
    // ----------------------------------------------------
    virtual double pdf_value(
        const point3& origin,
        const vec3& direction
    ) const override {

        hit_record rec;

        if (!this->hit(ray(origin, direction),
                       interval(0.001, infinity),
                       rec))
            return 0.0;

        vec3 to_center = center - origin;
        double distance_squared = to_center.length_squared();
        double radius_squared   = radius * radius;

        // If shading point is inside sphere
        if (distance_squared <= radius_squared)
            return 1.0 / (4.0 * pi);

        double cos_theta_max =
            sqrt(1.0 - radius_squared / distance_squared);

        double solid_angle =
            2.0 * pi * (1.0 - cos_theta_max);

        return 1.0 / solid_angle;
    }

    // ----------------------------------------------------
    // SOLID ANGLE SAMPLING
    // ----------------------------------------------------
    virtual vec3 random(
        const point3& origin
    ) const override {

        vec3 direction = center - origin;
        double distance_squared = direction.length_squared();
        double radius_squared   = radius * radius;

        // Build orthonormal basis toward sphere
        onb uvw;
        uvw.build_from_w(direction);

        // If inside sphere, fallback to uniform sphere sampling
        if (distance_squared <= radius_squared) {
            return random_unit_vector();
        }

        double cos_theta_max =
            sqrt(1.0 - radius_squared / distance_squared);

        double r1 = random_double();
        double r2 = random_double();

        double cos_theta =
            1.0 - r2 * (1.0 - cos_theta_max);

        double sin_theta =
            sqrt(1.0 - cos_theta * cos_theta);

        double phi = 2.0 * pi * r1;

        vec3 local_dir(
            cos(phi) * sin_theta,
            sin(phi) * sin_theta,
            cos_theta
        );

        return uvw.local(local_dir);
    }

public:
    point3 center;
    double radius;
    std::shared_ptr<material> mat_ptr;
};

#endif