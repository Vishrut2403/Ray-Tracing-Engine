#include <glad/glad.h>
#include "viewer/gl_util.h"
#include <iostream>

unsigned int gl_compile_shader(unsigned int type, const char* src) {
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

unsigned int gl_create_program(const char* vertex_src, const char* fragment_src) {
	unsigned int prog = glCreateProgram();
	unsigned int v = gl_compile_shader(GL_VERTEX_SHADER, vertex_src);
	unsigned int f = gl_compile_shader(GL_FRAGMENT_SHADER, fragment_src);
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
