#pragma once
#include <GLFW/glfw3.h>

class PreviewWindow {
public:
    PreviewWindow(int width, int height);
    ~PreviewWindow();

    bool should_close() const;
    void poll_events();
    void update(const float* framebuffer_data, float = 1.0f);

private:
    int width, height;
    unsigned int vao, vbo;
    unsigned int shader_program;
    GLFWwindow*  window;
    unsigned int texture;
};