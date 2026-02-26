# Ray Tracing: The Next Week

## Branch
book2-next-week

## Goal
Extend the Book 1 renderer with:
- Motion blur
- Bounding Volume Hierarchy (BVH)
- Textures
- Rectangles and boxes
- Volumes

---

# 1. Motion Blur

## Problem
Book 1 rendered a static world.
Real cameras integrate light over a time interval (shutter duration).

## Solution
1. Add time parameter to ray.
2. Add shutter interval (time0, time1) to camera.
3. Implement moving_sphere with linear interpolation.

## Mathematical Model

Center interpolation:

center(t) = center0 + ((t - time0)/(time1 - time0)) * (center1 - center0)

## Key Insight

Motion blur is temporal sampling.
Each pixel averages multiple time samples.

## Files Modified

- ray.h (added time)
- camera.h (added shutter interval)
- moving_sphere.h (new class)
- main.cpp (random_scene updated)