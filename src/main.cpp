#include "vec3.h"
#include <iostream>

int main() {

    vec3 v(1, 2, 3);
    vec3 w(4, 5, 6);

    std::cout << "v + w = " << v + w << "\n";
    std::cout << "dot(v, w) = " << dot(v, w) << "\n";
    std::cout << "cross(v, w) = " << cross(v, w) << "\n";
}