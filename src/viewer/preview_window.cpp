#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "viewer/preview_window.h"
#include <iostream>
#include <vector>

static unsigned int compile_shader(unsigned int type, const char* src) {
	unsigned int id = glCreateShader(type);
	glShaderSource(id, 1, &src, nullptr);
	glCompileShader(id);
	int ok;
	glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[512];
		glGetShaderInfoLog(id, 512, nullptr, log);
		std::cerr << "Shader compile error:\n" << log << "\n";
	}
	return id;
}

static unsigned int create_program(const char* vs, const char* fs) {
	unsigned int prog = glCreateProgram();
	unsigned int v = compile_shader(GL_VERTEX_SHADER, vs);
	unsigned int f = compile_shader(GL_FRAGMENT_SHADER, fs);
	glAttachShader(prog, v);
	glAttachShader(prog, f);
	glLinkProgram(prog);
	int ok;
	glGetProgramiv(prog, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[512];
		glGetProgramInfoLog(prog, 512, nullptr, log);
		std::cerr << "Program link error:\n" << log << "\n";
	}
	glDeleteShader(v);
	glDeleteShader(f);
	return prog;
}

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

	const char* fs = R"(
		#version 330 core
		in vec2 uv;
		out vec4 FragColor;
		uniform sampler2D screenTex;
		void main() {
			vec3 col = texture(screenTex, uv).rgb;
			col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
			FragColor = vec4(col, 1.0);
		}
	)";

	shader_program = create_program(vs, fs);
	glUseProgram(shader_program);
	glUniform1i(glGetUniformLocation(shader_program, "screenTex"), 0);
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
void PreviewWindow::poll_events()        { glfwPollEvents(); }

void PreviewWindow::update(const float* fb, float /*invSamples*/)
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