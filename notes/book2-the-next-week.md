# Book 2 — The Next Week

---

# Motion Blur

Book 2 introduces time as a dimension.

## Core Idea

Rays now carry a time value:

```cpp
class ray {
public:
    ray(const point3& origin,
        const vec3& direction,
        double time = 0.0);
};
```

Camera generates rays with random time:

```cpp
ray get_ray(double s, double t) const {
    return ray(origin,
               lower_left_corner + s*horizontal + t*vertical - origin,
               random_double(time0, time1));
}
```

Moving objects interpolate position:

```cpp
center = center0 + ((time - time0)/(time1-time0)) * (center1-center0);
```

This produces physically correct motion blur.

---

# Axis-Aligned Bounding Boxes (AABB)

## Problem

Without acceleration:

For each ray:
    test intersection with every object

Time complexity: O(N)

## AABB Structure

```cpp
class aabb {
public:
    point3 minimum;
    point3 maximum;

    bool hit(const ray& r,
             double t_min,
             double t_max) const;
};
```

Each object implements:

```cpp
virtual bool bounding_box(
    double time0,
    double time1,
    aabb& output_box
) const = 0;
```

Moving objects return a box that encloses motion over time.

---

# Bounding Volume Hierarchy (BVH)

## Problem

Brute-force intersection is too slow.

## Solution

Group objects into a tree.

Each node contains:
- left child
- right child
- bounding box enclosing both

Algorithm:

```
If ray misses node box:
    skip entire subtree

If ray hits:
    recurse into children
```

## BVH Construction

1. Choose random axis (x/y/z)
2. Sort objects along that axis
3. Split in half
4. Recursively build left and right nodes

```cpp
auto axis = random_int(0,2);
std::sort(objects.begin()+start,
          objects.begin()+end,
          comparator);
```

Leaf nodes:
- 1 object → left = right
- 2 objects → ordered pair

Each node stores:

```cpp
aabb box = surrounding_box(box_left, box_right);
```

---

# Performance Impact

Without BVH:
- O(N) intersections per ray

With BVH:
- O(log N) average behavior

In large scenes:
- 5–20x speed improvement

---

# Rendering Workflow (Book 2 Branch)

Compile:

    g++ -O3 -std=c++17 src/main.cpp -o render

Preview:

    ./render > renders/book2/preview.ppm

Final:

    ./render > renders/book2/final.ppm

Note:
Renders are stored locally and not pushed to GitHub.
They are deterministic and regeneratable.

---

# Architectural State After Book 2

Renderer now includes:

- Physically based materials
- Depth of field
- Motion blur
- AABB bounding volumes
- BVH acceleration structure

This is now a structurally scalable ray tracer.