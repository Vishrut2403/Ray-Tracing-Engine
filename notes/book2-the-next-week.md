# Book 2 — The Next Week

This book upgrades the renderer from a correct path tracer
to a scalable and extensible rendering system.

---

# 1. Motion Blur (Time as a Dimension)

## Concept

Light transport occurs over time.
A real camera shutter is open for a duration, not an instant.

Instead of tracing rays at a fixed time,
we sample time randomly within a shutter interval.

---

## Ray Upgrade

```cpp
class ray {
public:
    ray(const point3& origin,
        const vec3& direction,
        double time = 0.0);
};
```

Each ray now carries:

- origin
- direction
- time

---

## Camera Upgrade

```cpp
ray get_ray(double s, double t) const {
    return ray(origin,
               lower_left_corner + s*horizontal + t*vertical - origin,
               random_double(time0, time1));
}
```

The camera samples time uniformly between:

- time0 → shutter open
- time1 → shutter close

---

## Moving Objects

Objects interpolate position over time:

```cpp
center = center0
       + ((time - time0)/(time1-time0))
         * (center1-center0);
```

This produces physically correct motion blur.

---

# 2. Axis-Aligned Bounding Boxes (AABB)

## Problem

Without acceleration:

For each ray:
    test intersection with every object

Time complexity per ray: O(N)

This does not scale.

---

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

Bounding boxes allow:

- Fast early rejection
- Spatial partitioning
- Hierarchical grouping

---

## Ray–Box Intersection (Slab Method)

For each axis:

1. Compute intersection interval
2. Shrink valid t range
3. Exit early if interval collapses

This makes box tests extremely cheap.

---

# 3. Surrounding Boxes

When combining two objects:

```cpp
aabb surrounding_box(box0, box1);
```

This creates a bounding box that encloses both children.

Used to construct tree nodes.

---

# 4. Bounding Volume Hierarchy (BVH)

## Core Idea

Replace flat object list with a binary tree.

Each node stores:

- left child
- right child
- bounding box enclosing both

---

## BVH Construction Algorithm

1. Pick random axis (x, y, or z)
2. Sort objects along that axis
3. Split in half
4. Recursively build subtrees

Leaf conditions:
- 1 object → left = right
- 2 objects → ordered pair

Each node stores:

```cpp
box = surrounding_box(box_left, box_right);
```

---

## BVH Traversal Logic

```cpp
if (!box.hit(ray)) return false;

hit_left  = left->hit(...)
hit_right = right->hit(...)
```

Optimization:

```cpp
hit_right = right->hit(ray,
                       t_min,
                       hit_left ? rec.t : t_max,
                       rec);
```

If left child hits closer,
right subtree search is clipped.

This reduces unnecessary intersection tests.

---

# 5. Performance Impact

Without BVH:
- O(N) intersections per ray

With BVH:
- Expected O(log N)

Large scenes:
- 5× to 20× speed improvement
- Dramatic scalability increase

---

# 6. Architectural Upgrade

After BVH, the renderer now includes:

- Monte Carlo path tracing
- Diffuse / Metal / Dielectric materials
- Depth of field
- Motion blur
- AABB spatial bounds
- BVH acceleration

This is now a scalable physically based renderer.

---

# Rendering Workflow (Book 2 Branch)

Compile:

    g++ -O3 -std=c++17 src/main.cpp -o render

Preview:

    ./render > renders/book2/preview.ppm

Final:

    ./render > renders/book2/final.ppm

Renders are stored locally and not pushed to GitHub.
They are reproducible and regeneratable.

---

# 7. Texture Abstraction

## Motivation

Constant color is insufficient for realism.

We abstract surface color into a polymorphic texture system.

---

## Texture Interface

```cpp
class texture {
public:
    virtual color value(
        double u,
        double v,
        const point3& p
    ) const = 0;
};
```

Textures may represent:

- Solid color
- Checker pattern
- Procedural noise
- Image maps

Materials now depend on textures,
not fixed colors.

---

# 8. Checker Texture

## Motivation

Introduce spatially varying surface color without using images.

---

## Principle

We alternate between two textures based on position.

Let:

    s = sin(10x) * sin(10y) * sin(10z)

If s < 0:
    use odd texture
Else:
    use even texture

This creates a 3D checker pattern.

---

## Key Idea

Textures can contain other textures.

checker_texture stores:

- even texture
- odd texture

This demonstrates composability of texture systems.

---

# 9. Procedural Textures — Checker Pattern

## Motivation

Constant color is unrealistic.
We introduce spatial variation using procedural math.

---

## Checker Texture

A checker pattern is created using:

    s = sin(10x) * sin(10y) * sin(10z)

If s < 0:
    use odd texture
Else:
    use even texture

This creates a 3D alternating pattern.

---

## Design Pattern

checker_texture contains two textures:

- even
- odd

Textures are composable.

Material → Texture → Color

This decouples surface appearance from material logic.

---

## Architectural Upgrade

We transitioned from:

    material → constant color

to:

    material → texture abstraction → color

This enables:
- Checker textures
- Perlin noise
- Image textures
- Procedural materials

---

# 10. Perlin Noise

## Motivation

Checker patterns are artificial.
Natural materials exhibit smooth stochastic variation.

Perlin noise provides coherent randomness.

Unlike pure random numbers,
Perlin noise is spatially correlated.

---

# 11. Perlin Noise — Core Idea

Perlin noise generates smooth, coherent randomness.

Unlike pure random values, nearby points produce similar outputs.

---

## Structure

Perlin noise consists of:

- A 3D lattice grid
- Random gradient vectors at lattice points
- A permutation table
- Smooth interpolation between gradients

---

## Why Needed

Used for:
- Marble
- Wood
- Clouds
- Organic surface variation

It provides natural-looking stochastic patterns.

---

## Perlin Class Structure

Stores:

- 256 random gradient vectors
- 3 permutation tables (x, y, z)

Permutation tables are used to hash lattice coordinates
into pseudo-random gradient indices.

---

# 12. Perlin — Constructor and Permutation Tables

## Gradient Vectors

256 random unit vectors are generated.
These represent gradient directions at lattice points.

---

## Permutation Tables

Three permutation arrays (x, y, z) are created.

They:

- Contain shuffled values 0–255
- Act as hash tables
- Map lattice coordinates to gradient indices

Hashing approach:

    perm_x[x] ^ perm_y[y] ^ perm_z[z]

This produces deterministic pseudo-random gradient selection.