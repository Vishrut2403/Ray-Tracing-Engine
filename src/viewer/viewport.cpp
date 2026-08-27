#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "viewer/viewport.h"
#include "viewer/gl_util.h"
#include "viewer/solid_shading.h"
#include "render/tonemap.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* kVertexShader = R"(
	#version 330 core
	layout(location = 0) in vec3  aPos;
	layout(location = 1) in vec3  aNormal;
	layout(location = 2) in vec4  aColor;
	layout(location = 3) in float aEmissive;
	uniform mat4 uViewProj;
	uniform mat4 uView;
	out vec3  vViewNormal;
	out vec3  vViewPos;
	out vec4  vColor;
	out float vEmissive;
	void main() {
		// The view's rotation is orthonormal, so it carries normals as-is.
		vViewNormal = mat3(uView) * aNormal;
		vViewPos    = (uView * vec4(aPos, 1.0)).xyz;
		vColor      = aColor;
		vEmissive   = aEmissive;
		gl_Position = uViewProj * vec4(aPos, 1.0);
	}
)";

// A fixed studio, in view space so the lights follow the camera and every
// surface stays readable however the view is turned. Nothing here is a
// physical quantity -- it exists to show shape, not to be correct.
const char* kFragmentShader = R"(
	#version 330 core
	in vec3  vViewNormal;
	in vec3  vViewPos;
	in vec4  vColor;
	in float vEmissive;
	uniform vec3  uFlatColor;
	uniform float uUseMaterialColor;
	out vec4 FragColor;

	const vec3 kKeyDir   = normalize(vec3(-0.35,  0.55,  0.75));
	const vec3 kFillDir  = normalize(vec3( 0.75, -0.15,  0.45));
	const vec3 kRimDir   = normalize(vec3( 0.10,  0.55, -0.85));
	const vec3 kKeyTint  = vec3(1.00, 0.98, 0.94);
	const vec3 kFillTint = vec3(0.72, 0.78, 0.92);
	const vec3 kRimTint  = vec3(0.88, 0.90, 1.00);

	void main() {
		vec3 base = mix(uFlatColor, vColor.rgb, uUseMaterialColor);
		if (vEmissive > 0.5) { FragColor = vec4(base, vColor.a); return; }

		vec3 n = normalize(vViewNormal);
		vec3 v = normalize(-vViewPos);
		// Two-sided: a soup gathered from closed boxes and inward-facing walls
		// has no single correct side to light.
		if (dot(n, v) < 0.0) n = -n;

		vec3 lit = vec3(0.18)
				 + 0.90 * kKeyTint  * max(dot(n, kKeyDir),  0.0)
				 + 0.35 * kFillTint * max(dot(n, kFillDir), 0.0)
				 + 0.30 * kRimTint  * max(dot(n, kRimDir),  0.0);

		// One narrow highlight off the key, which is what reads curvature.
		float spec = pow(max(dot(n, normalize(kKeyDir + v)), 0.0), 48.0);

		FragColor = vec4(base * lit + 0.25 * spec * kKeyTint, vColor.a);
	}
)";

const char* kQuadVertexShader = R"(
	#version 330 core
	layout(location = 0) in vec2 aPos;
	out vec2 uv;
	void main() {
		uv = (aPos + 1.0) * 0.5;
		gl_Position = vec4(aPos, 0.0, 1.0);
	}
)";

// One numpad press of orbit, the 15 degrees Blender turns.
const real kStep = (real)15 * (real)3.14159265358979323846 / (real)180;

struct Vertex {
	float px, py, pz, nx, ny, nz;
	float r, g, b, a;
	float emissive;
};

void push(std::vector<Vertex>& out, const Tri& t) {
	color c = display_rgb(t.mat);
	float e = (t.mat && t.mat->display_emissive() && !t.translucent) ? 1.f : 0.f;
	// Media are shells standing in for a volume, so they are drawn thin enough
	// to see what is inside them.
	float a = t.translucent ? 0.28f : 1.0f;

	const point3* v[3] = { &t.v0, &t.v1, &t.v2 };
	const vec3*   n[3] = { &t.n0, &t.n1, &t.n2 };
	for (int i = 0; i < 3; ++i)
		out.push_back({(float)(*v[i])[0], (float)(*v[i])[1], (float)(*v[i])[2],
					   (float)(*n[i])[0], (float)(*n[i])[1], (float)(*n[i])[2],
					   (float)c.x(), (float)c.y(), (float)c.z(), a, e});
}

}  // namespace

Viewport::Viewport(const TriSoup& tris, const ViewportCamera& cam,
				   int width, int height)
	: camera(cam)
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_DEPTH_BITS, 24);

	window = glfwCreateWindow(width, height, "Viewport", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize GLAD\n";
		std::exit(-1);
	}

	glfwSetWindowUserPointer(window, this);
	glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
		((Viewport*)glfwGetWindowUserPointer(w))->on_cursor(x, y);
	});
	glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int b, int a, int m) {
		((Viewport*)glfwGetWindowUserPointer(w))->on_button(b, a, m);
	});
	glfwSetScrollCallback(window, [](GLFWwindow* w, double, double dy) {
		((Viewport*)glfwGetWindowUserPointer(w))->on_scroll(dy);
	});
	glfwSetKeyCallback(window, [](GLFWwindow* w, int k, int, int a, int m) {
		((Viewport*)glfwGetWindowUserPointer(w))->on_key(k, a, m);
	});

	tri_soup_bounds(tris, bounds_center, bounds_radius);
	// Without this the clip planes fall back to the framing distance, which is
	// wrong as soon as the view is dollied away from where it started.
	camera.scene_radius = bounds_radius;
	camera.pivot_at(bounds_center, bounds_radius * (real)2);

	std::vector<Vertex> verts;
	verts.reserve(tris.size() * 3);
	for (const Tri& t : tris) if (!t.translucent) push(verts, t);
	opaque_count = verts.size();
	for (const Tri& t : tris) if (t.translucent)  push(verts, t);
	total_count = verts.size();

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex),
				 verts.data(), GL_STATIC_DRAW);
	const GLsizei stride = sizeof(Vertex);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
						  (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride,
						  (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride,
						  (void*)(10 * sizeof(float)));

	shader_program = gl_create_program(kVertexShader, kFragmentShader);

	// The traced frame arrives as a texture, so it needs a quad of its own and
	// the same display transform the image writer applies.
	const float quad[] = { -1.f,-1.f,  3.f,-1.f,  -1.f,3.f };
	glGenVertexArrays(1, &quad_vao);
	glGenBuffers(1, &quad_vbo);
	glBindVertexArray(quad_vao);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), nullptr);

	std::string quad_fs = std::string(
		"#version 330 core\n"
		"in vec2 uv;\n"
		"out vec4 FragColor;\n"
		"uniform sampler2D screenTex;\n")
		+ tonemap_glsl() +
		"void main() {\n"
		"    FragColor = vec4(tonemap_display(texture(screenTex, uv).rgb), 1.0);\n"
		"}\n";
	quad_program = gl_create_program(kQuadVertexShader, quad_fs.c_str());
	glUseProgram(quad_program);
	glUniform1i(glGetUniformLocation(quad_program, "screenTex"), 0);

	glGenTextures(1, &render_tex);
	glBindTexture(GL_TEXTURE_2D, render_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	rendered_view = camera;

	glEnable(GL_DEPTH_TEST);
	glfwSwapInterval(1);   // else the redraw loop spins on a core
}

Viewport::~Viewport() {
	stop_render();
	glDeleteTextures(1, &render_tex);
	glDeleteProgram(quad_program);
	glDeleteBuffers(1, &quad_vbo);
	glDeleteVertexArrays(1, &quad_vao);
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
	glDeleteProgram(shader_program);
	glfwDestroyWindow(window);
	glfwTerminate();
}

bool Viewport::should_close() const { return glfwWindowShouldClose(window); }

void Viewport::wait_events(double timeout) { glfwWaitEventsTimeout(timeout); }

float Viewport::aspect() const {
	int w, h;
	glfwGetFramebufferSize(window, &w, &h);
	return h > 0 ? (float)w / (float)h : 1.0f;
}

void Viewport::frame_all() {
	camera.frame(bounds_center, bounds_radius, (real)aspect());
}

void Viewport::on_button(int button, int action, int mods) {
	// Blender's bindings, plus its three-button emulation for the trackpad:
	// alt with the left button stands in for the middle one.
	bool middle = button == GLFW_MOUSE_BUTTON_MIDDLE;
	bool emulated = button == GLFW_MOUSE_BUTTON_LEFT && (mods & GLFW_MOD_ALT);
	if (!middle && !emulated) return;

	if (action == GLFW_PRESS) {
		if      (mods & GLFW_MOD_SHIFT)   drag = Drag::Pan;
		else if (mods & GLFW_MOD_CONTROL) drag = Drag::Zoom;
		else                              drag = Drag::Orbit;
		glfwGetCursorPos(window, &last_x, &last_y);
	} else if (action == GLFW_RELEASE) {
		drag = Drag::None;
	}
}

void Viewport::on_cursor(double x, double y) {
	double dx = x - last_x, dy = y - last_y;
	last_x = x; last_y = y;
	if (drag == Drag::None) return;

	int h; glfwGetFramebufferSize(window, nullptr, &h);
	switch (drag) {
		case Drag::Orbit: camera.orbit((real)dx, (real)dy); break;
		case Drag::Pan:   camera.pan((real)dx, (real)dy, h); break;
		// Vertical only, so the zoom does not fight the pointer drifting sideways.
		case Drag::Zoom:  camera.dolly((real)(-dy * 0.02)); break;
		default: break;
	}
}

void Viewport::on_scroll(double dy) { camera.dolly((real)dy); }

void Viewport::on_key(int key, int action, int mods)
{
	if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
	bool flip = (mods & GLFW_MOD_CONTROL) != 0;   // ctrl gives the opposite side

	switch (key) {
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, 1); break;
		case GLFW_KEY_HOME:
			frame_all(); break;

		// Axis views. Blender is z-up and this world is y-up, so front looks
		// down -z, top looks down -y, and right looks down -x.
		case GLFW_KEY_KP_1:
			camera.look_along(vec3(0, 0, flip ? 1 : -1));
			camera.orthographic = true; break;
		case GLFW_KEY_KP_3:
			camera.look_along(vec3(flip ? 1 : -1, 0, 0));
			camera.orthographic = true; break;
		case GLFW_KEY_KP_7:
			camera.look_along(vec3(0, flip ? 1 : -1, 0));
			camera.orthographic = true; break;

		// Orbit in steps, for when the mouse is not the right tool.
		case GLFW_KEY_KP_4: camera.orbit_by(-kStep, 0); break;
		case GLFW_KEY_KP_6: camera.orbit_by( kStep, 0); break;
		case GLFW_KEY_KP_8: camera.orbit_by(0,  kStep); break;
		case GLFW_KEY_KP_2: camera.orbit_by(0, -kStep); break;
		case GLFW_KEY_KP_9:
			camera.yaw += (real)3.14159265358979323846;
			camera.pitch = -camera.pitch; break;

		// Blender puts these in the shading popover; there is no popover here.
		case GLFW_KEY_R:
			set_rendered(!rendered); break;

		case GLFW_KEY_M:
			material_color = !material_color; break;

		case GLFW_KEY_KP_5:
			if (!rendered) camera.orthographic = !camera.orthographic;
			break;
		case GLFW_KEY_KP_ADD:      camera.dolly( 1); break;
		case GLFW_KEY_KP_SUBTRACT: camera.dolly(-1); break;
		default: break;
	}
}

void Viewport::set_render(RenderFn fn) { render_fn = std::move(fn); }

void Viewport::set_rendered(bool on) {
	if (on && !render_fn) return;
	rendered = on;
	if (rendered) {
		// The tracer has no orthographic camera, so rendered mode drops back
		// to the perspective the view already frames.
		camera.orthographic = false;
		render_w = render_h = 0;   // forces a start on the next frame
	} else {
		stop_render();
	}
}

void Viewport::stop_render() {
	cancel = true;
	if (worker.joinable()) worker.join();
	cancel = false;
}

void Viewport::start_render() {
	stop_render();
	if (!render_fn || render_w <= 0 || render_h <= 0) return;

	render_fb = std::make_unique<Framebuffer>(render_w, render_h);
	snapshot.assign((size_t)render_w * render_h * 3, 0.0f);

	// The camera is copied, not captured: the worker must not see the view
	// change under it while it traces.
	::camera cam = to_render_camera(rendered_view, (real)render_w / render_h);
	Framebuffer* fb = render_fb.get();
	worker = std::thread([this, cam, fb]() { render_fn(cam, *fb, cancel); });
}

// A render is started one frame after the view and the window both stop
// changing, so a drag cancels rather than queueing a render per frame.
void Viewport::update_render() {
	int w, h;
	glfwGetFramebufferSize(window, &w, &h);

	if (!same_view(camera, rendered_view) || w != render_w || h != render_h) {
		stop_render();
		rendered_view = camera;
		render_w = w;
		render_h = h;
		pending  = true;
		return;
	}
	if (pending) { pending = false; start_render(); }
}

void Viewport::draw_rendered()
{
	update_render();

	glDisable(GL_DEPTH_TEST);
	glClearColor(0.f, 0.f, 0.f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (render_fb) {
		{
			// Held for the copy only: drawing under it would stall every
			// render thread until the compositor accepted the frame.
			std::lock_guard<std::mutex> lock(render_fb->mtx);
			std::memcpy(snapshot.data(), render_fb->raw_data(),
						snapshot.size() * sizeof(float));
		}
		glBindTexture(GL_TEXTURE_2D, render_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, render_w, render_h, 0,
					 GL_RGB, GL_FLOAT, snapshot.data());

		glUseProgram(quad_program);
		glBindVertexArray(quad_vao);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, render_tex);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	glEnable(GL_DEPTH_TEST);
	glfwSwapBuffers(window);
}

void Viewport::draw()
{
	int fb_w, fb_h;
	glfwGetFramebufferSize(window, &fb_w, &fb_h);
	if (fb_h <= 0) return;                 // minimised: nothing to project onto
	glViewport(0, 0, fb_w, fb_h);

	if (rendered) { draw_rendered(); return; }

	mat4 view = camera.view();
	mat4 vp   = camera.view_proj((real)fb_w / (real)fb_h);
	float view_f[16], vp_f[16];
	mat4_to_float(view, view_f);
	mat4_to_float(vp,   vp_f);

	glClearColor(0.22f, 0.22f, 0.22f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(shader_program);
	glUniformMatrix4fv(glGetUniformLocation(shader_program, "uView"),
					   1, GL_FALSE, view_f);
	glUniformMatrix4fv(glGetUniformLocation(shader_program, "uViewProj"),
					   1, GL_FALSE, vp_f);
	glUniform3f(glGetUniformLocation(shader_program, "uFlatColor"),
				0.73f, 0.73f, 0.73f);
	glUniform1f(glGetUniformLocation(shader_program, "uUseMaterialColor"),
				material_color ? 1.0f : 0.0f);
	glBindVertexArray(vao);

	glDrawArrays(GL_TRIANGLES, 0, (GLsizei)opaque_count);

	// Media boundaries would otherwise be solid shells hiding what is inside.
	if (total_count > opaque_count) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDepthMask(GL_FALSE);
		glDrawArrays(GL_TRIANGLES, (GLint)opaque_count,
					 (GLsizei)(total_count - opaque_count));
		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
	}

	glfwSwapBuffers(window);
}
