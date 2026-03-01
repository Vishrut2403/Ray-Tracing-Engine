cmake_minimum_required(VERSION 3.16)

project(RayTracer)

# --- C++ Standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Export compile_commands.json for VS Code ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Collect all source files ---
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# --- Create executable ---
add_executable(render ${SRC_FILES})

# --- Single include root ---
target_include_directories(render PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# --- OpenMP ---
find_package(OpenMP REQUIRED)
target_link_libraries(render PRIVATE OpenMP::OpenMP_CXX)