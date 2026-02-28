# Book 3 – Monte Carlo Integration

## Rendering as Integration

The renderer solves:

L_o = ∫_Ω f_r L_i cosθ dω

This integral is approximated via Monte Carlo sampling.

---

## Estimator Used In Engine

L ≈ f_r * Li * cosθ / p(ω)

Implemented inside ray_color().

---

## Importance Sampling

Using cosine-weighted sampling reduces variance because:

f_r cosθ / p(ω) simplifies to albedo.

This dramatically improves convergence.

---

## Russian Roulette Termination

Instead of fixed path depth truncation, paths are terminated
probabilistically after several bounces.

If survival probability is s:

Contribution is scaled by 1/s to preserve unbiasedness.

This removes truncation bias and matches the formulation
presented in Book 3.

---

# Book 3 – Geometry as a Probability Distribution

## Motivation

In Book 2, rays randomly bounced and sometimes hit light sources.
This produced high variance and noisy renders.

Book 3 introduces **explicit light sampling** by treating geometry as a probability distribution.

---

## Extending the `hittable` Interface

Added two new virtual functions:

```cpp
virtual double pdf_value(const point3& origin,
                         const vec3& direction) const;

virtual vec3 random(const point3& origin) const;
```

---

## `pdf_value(origin, direction)`

Returns the probability density (in solid angle measure) of sampling `direction`
toward this object from `origin`.

This requires converting:

**Area PDF → Solid Angle PDF**

\[
p(\omega) = \frac{\text{distance}^2}{\cos(\theta) \cdot \text{area}}
\]

---

## `random(origin)`

Generates a random direction from `origin`
toward a randomly chosen point on the object.

Used for:

- Direct light sampling  
- Multiple Importance Sampling (MIS)

---

## Why This Matters

Instead of hoping a BRDF sample hits a light,
we now directly sample lights.

This dramatically reduces variance.

---

## Integration with MIS

We combine:

- Light sampling PDF  
- BRDF sampling PDF  

\[
p_{\text{mix}} = \frac{1}{2} p_{\text{light}} + \frac{1}{2} p_{\text{brdf}}
\]

---

This is the foundation of efficient Monte Carlo path tracing.

---

## Rectangle Light PDF (Solid Angle Conversion)

When sampling a rectangle light, we first sample uniformly over its surface:

\[
p_A = \frac{1}{Area}
\]

However, rendering integrates over solid angle, not area.

Using the geometric relationship:

\[
d\omega = \frac{\cos\theta \, dA}{distance^2}
\]

We convert area density into solid angle density:

\[
p_\omega = \frac{distance^2}{\cos\theta \cdot Area}
\]

### Implementation

```cpp
double area = (x1 - x0) * (z1 - z0);

double distance_squared =
    rec.t * rec.t *
    direction.length_squared();

double cosine =
    fabs(dot(direction, rec.normal)
         / direction.length());

return distance_squared / (cosine * area);
```

---

# Book 3 – Importance Sampling (Theory)

Based on:
Ray Tracing: The Rest of Your Life

---

## 1. Monte Carlo Estimator

We want to compute:

\[
I = \int f(x) dx
\]

Monte Carlo approximation:

\[
I \approx \frac{1}{N} \sum_{i=1}^{N} \frac{f(x_i)}{p(x_i)}
\]

Where:
- \( p(x) \) is the sampling PDF
- \( x_i \sim p(x) \)

The estimator is unbiased if:

\[
p(x) > 0 \quad \text{whenever} \quad f(x) > 0
\]

---

## 2. Why Uniform Sampling Fails

If we sample uniformly:

\[
p(x) = constant
\]

but \( f(x) \) varies strongly, then:

- Many samples contribute almost nothing
- A few samples contribute large values
- Variance becomes high

This produces noise in rendered images.

---

## 3. Importance Sampling

Instead of sampling uniformly, we choose:

\[
p(x) \propto f(x)
\]

Then:

\[
\frac{f(x)}{p(x)} \approx constant
\]

This reduces variance dramatically.

---

## 4. Cosine-Weighted Sampling

For Lambertian surfaces:

\[
f(\omega) = \frac{albedo}{\pi}
\]

Rendering equation includes:

\[
f(\omega) \cos\theta
\]

So ideal sampling PDF is:

\[
p(\omega) = \frac{\cos\theta}{\pi}
\]

This is why we use cosine-weighted hemisphere sampling.

---

## 5. Explicit Light Sampling

Instead of hoping a random bounce hits a light:

We directly sample the light’s surface.

Area PDF:

\[
p_A = \frac{1}{Area}
\]

Converted to solid angle PDF:

\[
p_\omega = \frac{distance^2}{\cos\theta \cdot Area}
\]

This drastically reduces variance for direct lighting.

---

## 6. Multiple Importance Sampling (MIS)

We combine two PDFs:

\[
p_{mix} = \frac{1}{2} p_{light} + \frac{1}{2} p_{brdf}
\]

Final estimator:

\[
L = L_e + \frac{f(\omega) L_i(\omega)}{p_{mix}(\omega)}
\]

MIS prevents either PDF from dominating poorly in difficult lighting configurations.

---

## 7. Russian Roulette

To prevent infinite recursion:

We terminate paths probabilistically:

If survival probability = \( p \),

Then scale contribution by:

\[
\frac{1}{p}
\]

This keeps the estimator unbiased.

---

## 8. Participating Media

For isotropic scattering:

\[
f(\omega) = \frac{1}{4\pi}
\]

We sample uniformly over the sphere:

\[
p(\omega) = \frac{1}{4\pi}
\]

Since \( f = p \), estimator simplifies.

Free-flight distance is sampled via exponential distribution:

\[
d = -\frac{1}{\sigma} \ln(\xi)
\]

---

## 9. Key Insight of Book 3

Rendering quality is not about more samples.

It is about choosing the correct PDF.

Variance reduction comes from matching the sampling distribution to the integrand.

---

## Sphere Light PDF Derivation

Given shading point P and sphere center C with radius R:

Let:

\[
d = |C - P|
\]

Maximum visible angle:

\[
\sin\theta_{max} = \frac{R}{d}
\]

\[
\cos\theta_{max} = \sqrt{1 - \frac{R^2}{d^2}}
\]

The sphere subtends solid angle:

\[
\Omega = 2\pi(1 - \cos\theta_{max})
\]

Uniform sampling over this solid angle gives:

\[
p(\omega) =
\frac{1}
{2\pi(1 - \cos\theta_{max})}
\]

This samples only directions that intersect the sphere,
reducing variance compared to uniform surface sampling.

---


Final estimator:

\[
L = \frac{f(x) \cdot L_i \cdot \cos\theta}{p_{\text{mixed}}}
\]

This is unbiased but not optimal.

---

# Upgrade 1 — Power Heuristic MIS

## Problem

Equal-weight mixture can still produce:

- Fireflies
- High variance near sharp lights
- Instability in volumetrics

## Solution

Use the **Power Heuristic**:

\[
w_i = \frac{(n_i p_i)^2}{\sum_j (n_j p_j)^2}
\]

Since we use 1 sample per technique:

\[
w_i = \frac{p_i^2}{p_1^2 + p_2^2}
\]

---

## Implementation Strategy

Instead of only using:

```cpp
mixture_pdf