#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "viewer/preview_window.h"
#include "viewer/gl_util.h"
#include "render/tonemap.h"
#include <iostream>
#include <string>
#include <vector>

PreviewWindow::PreviewWindow(int w, int h) : width(w), height(h)
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(width, height, "Render Preview", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize GLAD\n";
		std::exit(-1);
	}

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	float verts[] = { -1.f,-1.f,  3.f,-1.f,  -1.f,3.f };
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), nullptr);

	const char* vs = R"(
		#version 330 core
		layout(location = 0) in vec2 aPos;
		out vec2 uv;
		void main() {
			uv = (aPos + 1.0) * 0.5;
			gl_Position = vec4(aPos, 0.0, 1.0);
		}
	)";

	std::string fs = std::string(
		"#version 330 core\n"
		"in vec2 uv;\n"
		"out vec4 FragColor;\n"
		"uniform sampler2D screenTex;\n")
		+ tonemap_glsl() +
		"void main() {\n"
		"    FragColor = vec4(tonemap_display(texture(screenTex, uv).rgb), 1.0);\n"
		"}\n";

	shader_program = gl_create_program(vs, fs.c_str());
	glUseProgram(shader_program);
	glUniform1i(glGetUniformLocation(shader_program, "screenTex"), 0);

	glfwSwapInterval(1);   // else the redraw loop spins on a core
}

PreviewWindow::~PreviewWindow() {
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
	glDeleteProgram(shader_program);
	glDeleteTextures(1, &texture);
	glfwDestroyWindow(window);
	glfwTerminate();
}

bool PreviewWindow::should_close() const { return glfwWindowShouldClose(window); }

// This and update() are GLFW/GL calls: main thread only, never under a lock.
void PreviewWindow::wait_events(double timeout) { glfwWaitEventsTimeout(timeout); }

void PreviewWindow::update(const float* fb)
{
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, fb);
	glClear(GL_COLOR_BUFFER_BIT);
	glUseProgram(shader_program);
	glBindVertexArray(vao);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glfwSwapBuffers(window);
}