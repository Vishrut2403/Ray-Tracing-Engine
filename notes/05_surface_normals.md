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