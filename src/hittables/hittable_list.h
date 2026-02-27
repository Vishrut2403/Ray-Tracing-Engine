#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <memory>
#include <vector>
#include "hittable.h"
#include "interval.h"

class hittable_list : public hittable {
public:
    std::vector<std::shared_ptr<hittable>> objects;

    hittable_list() {}

    hittable_list(std::shared_ptr<hittable> object) {
        add(object);
    }

    void clear() {
        objects.clear();
    }

    void add(std::shared_ptr<hittable> object) {
        objects.push_back(object);
    }

    virtual bool hit(
        const ray& r,
        const interval& ray_t,
        hit_record& rec
    ) const override {

        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto& object : objects) {
            if (object->hit(r,
                            interval(ray_t.min, closest_so_far),
                            temp_rec)) {

                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    virtual bool bounding_box(
        double time0,
        double time1,
        aabb& output_box
    ) const override {

        if (objects.empty())
            return false;

        aabb temp_box;
        bool first_box = true;

        for (const auto& object : objects) {

            if (!object->bounding_box(time0, time1, temp_box))
                return false;

            output_box = first_box
                ? temp_box
                : surrounding_box(output_box, temp_box);

            first_box = false;
        }

        return true;
    }

    virtual double pdf_value(
        const point3& origin,
        const vec3& direction
    ) const override {

        if (objects.empty())
            return 0.0;

        double weight = 1.0 / objects.size();
        double sum = 0.0;

        for (const auto& object : objects)
            sum += weight * object->pdf_value(origin, direction);

        return sum;
    }

    virtual vec3 random(
        const point3& origin
    ) const override {

        if (objects.empty())
            return vec3(1, 0, 0);

        int index = static_cast<int>(
            random_double(0, objects.size())
        );

        return objects[index]->random(origin);
    }
};

#endif