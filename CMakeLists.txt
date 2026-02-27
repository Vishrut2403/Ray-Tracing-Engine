cmake_minimum_required(VERSION 3.16)

project(RayTracer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all .cpp files
file(GLOB_RECURSE SRC_FILES src/*.cpp)

add_executable(render ${SRC_FILES})

# Add all header directories
target_include_directories(render PRIVATE
    src
    src/core
    src/hittables
    src/materials
    src/textures
    src/acceleration
    src/external
    src/pdfs
)