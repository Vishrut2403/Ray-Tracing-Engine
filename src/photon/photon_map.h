#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "core/rtweekend.h"
#include "hittables/hittable.h"
#include "materials/material.h"

#include <vector>
#include <unordered_map>
#include <cmath>
#include <atomic>
#include <mutex>

struct Photon {
	point3 p;     
	vec3   wi;      
	color  flux;    
};

struct VisiblePoint {
	point3 p;      
	vec3   n;      
	vec3   wo;    
	color  beta;    
	double radius;   
	color  flux;     
	long long n_photons;
	double u, v;  
	const material* mat = nullptr;
	bool   valid = false;

	int px = 0, py = 0;
};

class PhotonHashGrid {
public:
	double cell_size;

	void build(std::vector<VisiblePoint>& vps, double max_radius) {
		cell_size = max_radius;
		cells.clear();
		for (int i = 0; i < (int)vps.size(); ++i) {
			if (!vps[i].valid) continue;
			auto key = hash_point(vps[i].p);
			cells[key].push_back(i);
		}
	}

	void query(const point3& p, double radius,
			   std::vector<int>& result) const {
		result.clear();
		int r = (int)std::ceil(radius / cell_size);
		for (int dx = -r; dx <= r; ++dx)
		for (int dy = -r; dy <= r; ++dy)
		for (int dz = -r; dz <= r; ++dz) {
			int64_t key = hash_cell(grid_pos(p) + ivec3{dx,dy,dz});
			auto it = cells.find(key);
			if (it != cells.end()) {
				for (int idx : it->second)
					result.push_back(idx);
			}
		}
	}

private:
	struct ivec3 { int x,y,z;
		ivec3 operator+(const ivec3& o) const { return {x+o.x,y+o.y,z+o.z}; }
	};

	ivec3 grid_pos(const point3& p) const {
		return { (int)std::floor(p.x()/cell_size),
				 (int)std::floor(p.y()/cell_size),
				 (int)std::floor(p.z()/cell_size) };
	}

	int64_t hash_cell(const ivec3& c) const {
		return (int64_t)c.x * 73856093LL
			 ^ (int64_t)c.y * 19349663LL
			 ^ (int64_t)c.z * 83492791LL;
	}

	int64_t hash_point(const point3& p) const {
		return hash_cell(grid_pos(p));
	}

	std::unordered_map<int64_t, std::vector<int>> cells;
};