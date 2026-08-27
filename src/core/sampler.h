#pragma once

#include <cstdint>
#include <limits>
#include "core/vec3.h"

// Low-discrepancy sampling for the CPU integrators.
//
// Samples come from a scrambled (0,2)-sequence taken in consecutive dimension
// pairs, which stratifies far better than white noise for the same count. Call
// sites keep using random_double(); it routes here whenever a pixel sample is
// active, so nothing else has to know about dimensions. Anything outside a
// sample -- scene building, the photon pass -- falls back to the RNG.
//
// Beyond the first few dimensions the pairs are only decorrelated by their
// scramble, so quality degrades gracefully towards white noise rather than
// developing structure.

inline uint32_t sampler_hash(uint32_t x) {
	x ^= x >> 16; x *= 0x7feb352du;
	x ^= x >> 15; x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

inline uint32_t sampler_mix(uint32_t a, uint32_t b) {
	return sampler_hash(a ^ (b * 0x9e3779b9u));
}

inline uint32_t sampler_reverse_bits(uint32_t n) {
	n = (n << 16) | (n >> 16);
	n = ((n & 0x00ff00ffu) << 8) | ((n & 0xff00ff00u) >> 8);
	n = ((n & 0x0f0f0f0fu) << 4) | ((n & 0xf0f0f0f0u) >> 4);
	n = ((n & 0x33333333u) << 2) | ((n & 0xccccccccu) >> 2);
	n = ((n & 0x55555555u) << 1) | ((n & 0xaaaaaaaau) >> 1);
	return n;
}

// Laine-Karras permutation: a cheap stand-in for a nested Owen scramble.
inline uint32_t sampler_lk_permute(uint32_t x, uint32_t seed) {
	x += seed;
	x ^= x * 0x6c50b47cu;
	x ^= x * 0xb82f1e52u;
	x ^= x * 0xc7afe638u;
	x ^= x * 0x8d22f6e6u;
	return x;
}

inline uint32_t sampler_owen(uint32_t x, uint32_t seed) {
	x = sampler_reverse_bits(x);
	x = sampler_lk_permute(x, seed);
	return sampler_reverse_bits(x);
}

// Second dimension of the (0,2)-sequence, Gray-code form.
inline uint32_t sampler_sobol2(uint32_t i, uint32_t scramble) {
	for (uint32_t v = 1u << 31; i; i >>= 1, v ^= v >> 1)
		if (i & 1u) scramble ^= v;
	return scramble;
}

struct SamplerState {
	bool     active      = false;
	uint32_t pixel       = 0;
	uint32_t index       = 0;
	uint32_t dim         = 0;
	real   pending     = 0.0;
	bool     has_pending = false;
};

inline thread_local SamplerState g_sampler;

inline void sampler_begin_sample(uint32_t pixel, uint32_t index) {
	g_sampler.active      = true;
	g_sampler.pixel       = sampler_hash(pixel + 1u);
	g_sampler.index       = index;
	g_sampler.dim         = 0;
	g_sampler.has_pending = false;
}

inline void sampler_end_sample() { g_sampler.active = false; }

// Handing each bounce a fixed block of dimensions was tried, so that a given
// decision always read the same dimension whatever the path before it did. It
// was measured worse: at 32 spp on cornell it cost 16-20% relMSE at every block
// size from 4 to 12, and gained nothing significant on ggx or glass. The reason
// is above -- pairs past the first few are decorrelated only by their scramble,
// so a fixed block pushes every decision into a higher, weaker dimension, and
// that costs more than the alignment is worth. Dimensions stay consecutive.

// Samples must stay strictly below 1: callers scale them by a count and index
// with the result, and in single precision the top of the 2^32 range rounds up
// to exactly 1.0.
inline real sampler_unit(uint32_t bits) {
	const real inv32 = (real)2.3283064365386963e-10;   // 1 / 2^32
	const real hi    = (real)1.0
					 - (real)0.5*std::numeric_limits<real>::epsilon();
	real x = (real)bits * inv32;
	return x < hi ? x : hi;
}

inline real sampler_next() {
	SamplerState& s = g_sampler;
	if (s.has_pending) {
		s.has_pending = false;
		++s.dim;
		return s.pending;
	}

	uint32_t pair = s.dim >> 1;

	// Shuffling the sample index per dimension pair is what decorrelates the
	// pairs. Scrambling only the output leaves every pair running the same
	// underlying sequence, which stops the estimator converging.
	uint32_t idx = sampler_owen(s.index, sampler_mix(s.pixel, pair + 0x51u));
	uint32_t sx  = sampler_mix(s.pixel, pair * 2u + 1u);
	uint32_t sy  = sampler_mix(s.pixel, pair * 2u + 2u);

	real u = sampler_unit(sampler_owen(sampler_reverse_bits(idx), sx));
	real v = sampler_unit(sampler_owen(sampler_sobol2(idx, 0u),   sy));

	s.pending     = v;
	s.has_pending = true;
	++s.dim;
	return u;
}
