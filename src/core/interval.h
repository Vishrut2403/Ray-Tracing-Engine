#ifndef INTERVAL_H
#define INTERVAL_H

class interval {
public:
    double min, max;

    interval() : min(0), max(0) {}

    interval(double _min, double _max)
        : min(_min), max(_max) {}

    bool contains(double x) const {
        return min <= x && x <= max;
    }

    bool surrounds(double x) const {
        return min < x && x < max;
    }
};

#endif