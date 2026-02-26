#ifndef TEXTURE_H
#define TEXTURE_H

#include "rtweekend.h"
#include "perlin.h"

class texture {
public:
    virtual color value(
        double u,
        double v,
        const point3& p
    ) const = 0;
};

class solid_color : public texture {
public:
    solid_color() {}
    solid_color(color c) : color_value(c) {}
    solid_color(double r, double g, double b)
        : solid_color(color(r,g,b)) {}

    virtual color value(
        double u,
        double v,
        const point3& p
    ) const override {
        return color_value;
    }

private:
    color color_value;
};

class checker_texture : public texture {
public:
    checker_texture() {}

    checker_texture(
        std::shared_ptr<texture> _even,
        std::shared_ptr<texture> _odd
    ) : even(_even), odd(_odd) {}

    checker_texture(
        const color& c1,
        const color& c2
    ) : even(std::make_shared<solid_color>(c1)),
        odd(std::make_shared<solid_color>(c2)) {}

    virtual color value(
        double u,
        double v,
        const point3& p
    ) const override {

        auto sines =
            sin(10*p.x()) *
            sin(10*p.y()) *
            sin(10*p.z());

        if (sines < 0)
            return odd->value(u,v,p);
        else
            return even->value(u,v,p);
    }

private:
    std::shared_ptr<texture> even;
    std::shared_ptr<texture> odd;
};

class noise_texture : public texture {
public:
    noise_texture() : scale(1) {}
    noise_texture(double sc) : scale(sc) {}

    virtual color value(
        double u,
        double v,
        const point3& p
    ) const override {

        return color(1,1,1) *
               0.5 *
               (1 + sin(scale * p.z() + 10 * noise.turb(p)));
    }

private:
    perlin noise;
    double scale;
};

#endif