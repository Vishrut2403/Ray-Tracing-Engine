#pragma once

#include <cstddef>
#include "geometry/tri_soup.h"
#include "viewer/viewport_camera.h"

struct GLFWwindow;

// Rasterises the scene's display geometry. It draws the same triangles the
// tessellation walk hands the tests, so what is on screen is the geometry the
// integrators intersect and not a second description of the scene.
class Viewport {
public:
	Viewport(const TriSoup& tris, const ViewportCamera& cam, int width, int height);
	~Viewport();

	Viewport(const Viewport&)            = delete;
	Viewport& operator=(const Viewport&) = delete;

	bool should_close() const;
	void wait_events(double timeout);
	void draw();

	// Puts the whole scene in frame, as Home does in Blender.
	void frame_all();

	ViewportCamera camera;
	// Solid shading draws material colours; off gives Blender's flat grey.
	bool           material_color = true;

private:
	// What a drag is doing, decided by the modifiers held when it started.
	enum class Drag { None, Orbit, Pan, Zoom };

	void on_cursor(double x, double y);
	void on_button(int button, int action, int mods);
	void on_scroll(double dy);
	void on_key(int key, int action, int mods);
	float aspect() const;

	GLFWwindow*  window = nullptr;
	unsigned int vao = 0, vbo = 0;
	unsigned int shader_program = 0;
	// Opaque triangles come first in the buffer so the two passes are two
	// draw calls over one range rather than a per-triangle sort.
	std::size_t  opaque_count = 0;
	std::size_t  total_count  = 0;

	point3 bounds_center;
	real   bounds_radius = 1;

	Drag   drag = Drag::None;
	double last_x = 0, last_y = 0;
};
