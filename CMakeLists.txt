cmake_minimum_required(VERSION 3.16)

project(RayTracer)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Collect all .cpp sources
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# Executable
add_executable(render ${SRC_FILES})

target_include_directories(render PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/external
)

# OpenMP (CPU parallelism)
find_package(OpenMP REQUIRED)
target_link_libraries(render PRIVATE OpenMP::OpenMP_CXX)

# -----------------------------
# OpenGL
# -----------------------------
find_package(OpenGL REQUIRED)
target_link_libraries(render PRIVATE OpenGL::GL)

# GLFW
find_package(glfw3 3.3 REQUIRED)
target_link_libraries(render PRIVATE glfw)

# GLAD (OpenGL loader)
add_library(glad
    ${CMAKE_SOURCE_DIR}/src/external/glad.c
)

target_include_directories(glad PUBLIC
    ${CMAKE_SOURCE_DIR}/src/external
)

target_link_libraries(render PRIVATE glad)

if(UNIX AND NOT APPLE)
    target_link_libraries(render PRIVATE dl)
endif()