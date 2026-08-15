#include <omp.h>

#include "render/ppm_renderer.h"
#include "materials/material.h"
#include "lights/env_light.h"
#include "core/onb.h"

#include <iostream>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>

static constexpr double pi_val = 3.14159265358979323846;

static color trace_camera_ray(
	const ray& r,
	const std::shared_ptr<hittable>& world,
	const std::shared_ptr<env_light>& env,
	const color& background,
	int max_depth,
	VisiblePoint& vp
) {
	ray   cur  = r;
	color beta(1,1,1);
	color direct(0,0,0);

	for (int depth = 0; depth < max_depth; ++depth) {
		hit_record rec;
		if (!world->hit(cur, interval(1e-4, infinity), rec)) {
			direct = direct + beta*(env ? env->Le(cur.direction()) : background);
			return direct;
		}
		color Le = rec.mat_ptr->emitted(cur,rec,rec.u,rec.v,rec.p);
		if (Le.length_squared()>0.0){direct=direct+beta*Le;return direct;}

		BSDFSample bs = rec.mat_ptr->sample(cur,rec);

		if (bs.pdf>0.0 && bs.is_delta){
			// A delta lobe carries its whole weight in f; no cosine/pdf.
			beta=beta*bs.f;
			cur=ray(rec.p,bs.wi,cur.time());
			continue;
		}

		// Record the visible point even if the sample failed — bailing out here
		// leaves nothing for photons to reach and the surface renders black.
		vp.p=rec.p; vp.n=rec.normal; vp.wo=-unit_vector(cur.direction());
		vp.beta=beta; vp.u=rec.u; vp.v=rec.v; vp.mat=rec.mat_ptr; vp.valid=true;
		return direct;
	}
	return direct;
}

void PPMRenderer::render(
	const Scene& scene,
	Framebuffer& fb,
	const camera& cam,
	const color& background
) {
	auto world = scene.world;
	auto env   = scene.env;
	const int W = fb.get_width(), H = fb.get_height();

	struct PixelState {
		VisiblePoint vp;
		color  direct{0,0,0};
		long long N = 0;     
		color  tau{0,0,0}; 
		double R = 0;   
	};

	std::vector<PixelState> pixels(W*H);
	for (auto& p : pixels) p.R = initial_radius;

	std::cerr << "[PPM] " << W << "x" << H
			  << "  iters=" << n_iterations
			  << "  photons/iter=" << photons_per_iter
			  << "  r0=" << initial_radius << "\n";

	// Emitter taken from the scene. Hardcoding it made this renderer correct
	// only for Cornell-shaped scenes.
	std::shared_ptr<hittable> light_obj;
	if (scene.lights)
		for (const auto& o : scene.lights->objects)
			if (o->area() > 0.0) { light_obj = o; break; }
	if (!light_obj) {
		std::cerr << "[PPM] no area light in scene; nothing to trace\n";
		return;
	}

	double light_area = light_obj->area();
	vec3   probe_ng;
	point3 probe_p = light_obj->sample_area(0.5, 0.5, probe_ng);
	vec3   light_n = probe_ng;
	color  Le_light(0,0,0);
	for (int side = 0; side < 2 && Le_light.length_squared() == 0.0; ++side) {
		vec3   nn = (side == 0) ? probe_ng : -probe_ng;
		ray    probe(probe_p + nn*1e-3, -nn, 0.0);
		hit_record lr;
		if (world->hit(probe, interval(1e-6, 1e-2), lr) && lr.mat_ptr) {
			color e = lr.mat_ptr->emitted(probe, lr, lr.u, lr.v, lr.p);
			if (e.length_squared() > 0.0) { Le_light = e; light_n = nn; }
		}
	}

	color photon_flux = Le_light * light_area / (double)photons_per_iter;

	std::cerr << "[PPM] light area=" << light_area
			  << " Le=(" << Le_light.x() << "," << Le_light.y() << ","
			  << Le_light.z() << ") flux=(" << photon_flux.x() << ","
			  << photon_flux.y() << "," << photon_flux.z() << ")\n";

	for (int iter = 0; iter < n_iterations; ++iter) {

		std::vector<color>     phi(W*H, color(0,0,0));
		std::vector<long long> M(W*H, 0);

#pragma omp parallel for schedule(dynamic,8)
		for (int j=0;j<H;++j) for (int i=0;i<W;++i) {
			int idx=j*W+i;
			pixels[idx].vp.valid=false;
			color dir = trace_camera_ray(
				cam.get_ray((i+random_double())/(W-1),
							(j+random_double())/(H-1)),
				world, env, background, max_depth, pixels[idx].vp);
			pixels[idx].direct = pixels[idx].direct + dir;
		}

		double cell = initial_radius;
		auto hash3 = [](int x,int y,int z) -> int64_t {
			return (int64_t)x*73856093LL^(int64_t)y*19349663LL^(int64_t)z*83492791LL;
		};
		auto cell_of = [&](const point3& p, int&cx,int&cy,int&cz){
			cx=(int)std::floor(p.x()/cell);
			cy=(int)std::floor(p.y()/cell);
			cz=(int)std::floor(p.z()/cell);
		};

		std::unordered_map<int64_t,std::vector<int>> grid;
		grid.reserve(W*H*2);
		for (int idx=0;idx<W*H;++idx){
			if (!pixels[idx].vp.valid) continue;
			int x,y,z; cell_of(pixels[idx].vp.p,x,y,z);
			grid[hash3(x,y,z)].push_back(idx);
		}

		// The grid and every VisiblePoint are read-only here, so the only shared
		// writes are phi/M (atomic below). random_double() is thread_local.
#pragma omp parallel for schedule(dynamic,1024)
		for (int pi=0;pi<photons_per_iter;++pi){
			vec3   ng_unused;
			point3 lpos = light_obj->sample_area(random_double(),
												 random_double(), ng_unused);

			onb uvw; uvw.build_from_w(light_n);
			vec3 emit=uvw.local(random_cosine_direction());
			double ce=std::max(dot(light_n,emit),0.0);
			if (ce<1e-6) continue;

			color flux = photon_flux;
			ray cur(lpos,emit,0.0);

			for (int depth=0;depth<max_depth;++depth){
				hit_record rec;
				if (!world->hit(cur,interval(1e-4,infinity),rec)) break;
				if (rec.mat_ptr->emitted(cur,rec,rec.u,rec.v,rec.p).length_squared()>0) break;

				BSDFSample bs=rec.mat_ptr->sample(cur,rec);
				if (bs.pdf<=0.0) break;

				if (!bs.is_delta){
					int cx,cy,cz; cell_of(rec.p,cx,cy,cz);
					for (int dx=-1;dx<=1;++dx)
					for (int dy=-1;dy<=1;++dy)
					for (int dz=-1;dz<=1;++dz){
						auto it=grid.find(hash3(cx+dx,cy+dy,cz+dz));
						if (it==grid.end()) continue;
						for (int idx:it->second){
							auto& px=pixels[idx];
							if (!px.vp.valid) continue;
							if ((px.vp.p-rec.p).length_squared()>px.R*px.R) continue;
							if (dot(px.vp.n,-unit_vector(cur.direction()))<0) continue;

							// material::f takes -ray.direction() as the view
							// vector, so this ray must travel into the surface.
							ray dr(px.vp.p+px.vp.wo*1e-3,-px.vp.wo,0.0);
							hit_record drec;
							drec.p=px.vp.p; drec.normal=px.vp.n;
							drec.u=px.vp.u; drec.v=px.vp.v;
							drec.front_face=true; drec.mat_ptr=px.vp.mat;
							color f=px.vp.mat->f(dr,-unit_vector(cur.direction()),drec);
							if (f.length_squared()<1e-12) continue;

							color contrib = px.vp.beta*f*flux;
#pragma omp atomic
							phi[idx].e[0] += contrib.x();
#pragma omp atomic
							phi[idx].e[1] += contrib.y();
#pragma omp atomic
							phi[idx].e[2] += contrib.z();
#pragma omp atomic
							M[idx]++;
						}
					}
				}

				double surv=std::clamp(flux.max_component(),0.05,0.95);
				if (random_double()>surv) break;
				flux=flux/surv;
				flux = bs.is_delta ? flux*bs.f
					 : bs.is_phase ? flux*bs.f/bs.pdf
					 : flux*bs.f*std::abs(dot(rec.normal,bs.wi))/bs.pdf;
				cur=ray(rec.p,bs.wi,0.0);
			}
		}

		for (int idx=0;idx<W*H;++idx){
			auto& px=pixels[idx];
			if (!px.vp.valid||M[idx]==0) continue;
			double Nn=px.N, Mn=M[idx];
			double ratio=(Nn+alpha*Mn)/(Nn+Mn);
			px.tau=(px.tau+phi[idx])*ratio;
			px.R=px.R*std::sqrt(ratio);
			px.N=(long long)(px.N+alpha*Mn);
		}

		std::cerr<<"\r[PPM] iter "<<(iter+1)<<"/"<<n_iterations<<std::flush;
	}
	std::cerr<<"\n";

	int dbg=0;
	for (int idx=0;idx<W*H&&dbg<3;++idx){
		auto& px=pixels[idx];
		if (px.N>0&&px.vp.valid){
			double area=pi_val*px.R*px.R;
			color L=px.tau/area;
			std::cerr<<"  dbg px "<<idx<<" N="<<px.N<<" R="<<px.R
					 <<" tau=("<<px.tau.x()<<")"
					 <<" L=("<<L.x()<<","<<L.y()<<","<<L.z()<<")\n";
			dbg++;
		}
	}

	std::cerr<<"[PPM] done\n";

	std::lock_guard<std::mutex> lock(fb.mtx);
	for (int j=0;j<H;++j) for (int i=0;i<W;++i){
		int idx=j*W+i;
		auto& px=pixels[idx];

		color L=px.direct/(double)n_iterations;

		if (px.vp.valid&&px.N>0){
			double area=pi_val*px.R*px.R;
			L=L+px.tau/area;
		}

		fb.set(i,j,L);
	}
}
