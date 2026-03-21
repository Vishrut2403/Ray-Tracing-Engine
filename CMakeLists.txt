cmake_minimum_required(VERSION 3.18)
project(RayTracer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RTX 3060 = sm_86
set(CMAKE_CUDA_ARCHITECTURES 86)

# ── CPU sources (all .cpp) ──────────────────────────────────────────────────
file(GLOB_RECURSE CPU_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(FILTER CPU_SRC EXCLUDE REGEX ".*intellisense\\.cpp$")

# ── GPU sources (all .cu) ───────────────────────────────────────────────────
file(GLOB_RECURSE CUDA_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cu)

add_executable(render ${CPU_SRC} ${CUDA_SRC})

target_include_directories(render PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/external
    ${CMAKE_SOURCE_DIR}/src/cuda
)

# Allow .cu files to include project headers
set_target_properties(render PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ── OpenMP ──────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)
target_link_libraries(render PRIVATE OpenMP::OpenMP_CXX)

# ── OpenGL ──────────────────────────────────────────────────────────────────
find_package(OpenGL REQUIRED)
target_link_libraries(render PRIVATE OpenGL::GL)

find_package(glfw3 3.3 REQUIRED)
target_link_libraries(render PRIVATE glfw)

add_library(glad ${CMAKE_SOURCE_DIR}/src/external/glad.c)
target_include_directories(glad PUBLIC ${CMAKE_SOURCE_DIR}/src/external)
target_link_libraries(render PRIVATE glad)

if(UNIX AND NOT APPLE)
    target_link_libraries(render PRIVATE dl)
endif()

# ── CUDA runtime + cuRAND ───────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
target_link_libraries(render PRIVATE CUDA::cudart CUDA::curand)