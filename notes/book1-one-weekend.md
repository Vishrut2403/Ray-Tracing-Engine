# Chapter 2 – Output an Image

## Goal
Generate first PPM image manually.

## Key Learnings

- PPM format is simple ASCII image format.
- Pixels written left-to-right, top-to-bottom.
- RGB stored as integers 0–255.
- Colors internally represented as doubles (0.0–1.0).

## Important Insight

We are manually controlling every pixel.
This is the foundation of the renderer.

---

# Chapter 3 – vec3 Class

## Why vec3 Exists

Everything in a ray tracer is vector math:
- Points
- Rays
- Normals
- Colors

## Key Implementations

- Operator overloading
- Dot product
- Cross product
- Unit vector normalization

## Important Insight

The renderer is fundamentally linear algebra.

---

# Chapter 4 – Rays & Camera

## Key Formula

Ray equation:
P(t) = A + tB

## Concepts Learned

- Ray parameterization
- Virtual viewport
- Camera origin
- Ray per pixel
- Linear interpolation for gradient sky

## Important Insight

Ray tracing = shooting rays through a virtual viewport into a scene.

---

# Chapter 5 – Ray-Sphere Intersection

## Goal
Detect if a ray intersects a sphere.

## Math Used

Substituted ray equation into sphere equation,
resulting in a quadratic:

a t² + b t + c = 0

If discriminant ≥ 0 → intersection exists.

## Insight

Rendering is solving geometry analytically.

---

# Chapter 6 – Surface Normals

## What Changed

Instead of returning just hit/no-hit,
we compute:

- Exact intersection parameter t
- Hit point P = A + tB
- Surface normal N = (P - C) / |P - C|

## Why It Matters

Surface normals are essential for:
- Lighting
- Reflections
- Shading
- Material systems

## Key Insight

Geometry gives us orientation.
Orientation enables lighting.

---

# Chapter 7 – Hittable Abstraction

## Why Refactored

Hardcoding sphere intersection does not scale.

We introduced:
- hit_record struct
- hittable abstract class
- sphere inheriting from hittable

## Key Concept

Polymorphism allows multiple object types to share a common interface.

## Insight

We are transitioning from demo code to engine architecture.

---

# Chapter 8 – Multiple Objects

## What is Added

- hittable_list container
- Support for multiple objects
- Closest-hit logic

## Important Concept

For each ray:
- Test all objects
- Keep nearest intersection
- Return that hit

## Insight

This is the foundation of real scene rendering.

---

# Chapter 9 – Diffuse Reflection

## What Changed

Instead of coloring by normal,
we simulate light bouncing.

## Key Ideas

- Random hemisphere scattering
- Recursive ray tracing
- Energy attenuation (0.5 factor)
- Depth limiting

## Insight

Rendering is solving the light transport equation using Monte Carlo methods.

---

# Chapter 10 – Material System

## What Changed

Removed hardcoded diffuse logic.
Introduced:

- material base class
- lambertian implementation
- material pointer in hit_record

## Why It Matters

Objects now control light interaction.
Renderer is modular and extensible.

## Insight

This is how real PBR engines are structured.

---

# Chapter 11 – Anti-Aliasing & Gamma Correction

## What is Added

- Multiple samples per pixel
- Random subpixel jitter
- Averaging
- Gamma correction (sqrt)

## Why It Matters

Rendering is Monte Carlo integration.
Multiple samples reduce variance.
Gamma correction ensures physically correct display.

## Insight

Path tracing = stochastic light integration.

---

# Chapter 12 – Metal Material

## What is Added

- reflect() function
- metal material class
- fuzz parameter for roughness

## Key Physics

Reflection vector:
R = V - 2(V·N)N

## Insight

Specular reflection models mirror-like surfaces.
Fuzz introduces micro-surface roughness.

---

# Chapter 13 – Dielectric (Glass)

## What is Added

- refract() function
- Schlick reflectance approximation
- dielectric material class
- Front/back face detection

## Physics Concepts

- Snell’s Law
- Total internal reflection
- Fresnel reflectance

## Insight

Glass combines reflection and refraction probabilistically.

# Chapter – Hollow Glass Sphere

## What is Added

- Inner sphere with negative radius
- Simulated thickness for glass

## Insight

Negative radius flips normal direction.
Allows modeling of dielectric boundaries.

# rtweekend.h

## Purpose
Central utility header for:
- constants
- random helpers
- clamp
- common includes

## Benefit
Reduces duplication and improves structure.

---

# Final Scene

## Features
- Random sphere generation
- Mixed materials
- Depth of field
- Monte Carlo path tracing

## Result
Successfully reproduced final scene of Ray Tracing in One Weekend.