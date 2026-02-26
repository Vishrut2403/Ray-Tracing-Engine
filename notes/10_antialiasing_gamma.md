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