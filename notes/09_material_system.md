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