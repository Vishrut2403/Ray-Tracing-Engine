#pragma once

#include <GLFW/glfw3.h>

class PreviewWindow {
public:
    PreviewWindow(int width, int height);
    ~PreviewWindow();

    bool should_close() const;
    void update(const float* framebuffer_data, float invSamples);
    void poll_events();

private:
    int width;
    int height;
    unsigned int vao;
    unsigned int vbo;
    unsigned int shader_program;
    int inv_samples_loc;

    GLFWwindow* window;
    unsigned int texture;
};