#pragma once

// Shader plumbing shared by the render preview and the viewport.
unsigned int gl_compile_shader(unsigned int type, const char* src);
unsigned int gl_create_program(const char* vertex_src, const char* fragment_src);
