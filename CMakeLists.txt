cmake_minimum_required(VERSION 3.18)
project(RayTracer LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_PARALLEL_LEVEL $ENV{NPROC})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RTX 3060 = sm_86
set(CMAKE_CUDA_ARCHITECTURES 86)

execute_process(
    COMMAND gcc -print-file-name=include-fixed
    OUTPUT_VARIABLE GCC_INCLUDE_FIXED
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND gcc -print-file-name=include
    OUTPUT_VARIABLE GCC_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# CPU sources (all .cpp)
file(GLOB_RECURSE CPU_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(APPEND CPU_SRC ${CMAKE_SOURCE_DIR}/src/external/miniz.c)
list(FILTER CPU_SRC EXCLUDE REGEX ".*intellisense\\.cpp$")

# GPU sources
file(GLOB_RECURSE CUDA_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cu)

# Tests bring their own main, and their .cu must not land in `render`.
set(TEST_SRC ${CMAKE_SOURCE_DIR}/src/tests/test_main.cpp
             ${CMAKE_SOURCE_DIR}/src/tests/test_gpu.cu
             ${CMAKE_SOURCE_DIR}/src/tests/test_medium.cu
             ${CMAKE_SOURCE_DIR}/src/tests/test_render.cpp)
list(FILTER CPU_SRC  EXCLUDE REGEX ".*/tests/.*")
list(FILTER CUDA_SRC EXCLUDE REGEX ".*/tests/.*")
set(CORE_SRC ${CPU_SRC})
list(FILTER CORE_SRC EXCLUDE REGEX ".*/main\\.cpp$")

add_executable(render ${CPU_SRC} ${CUDA_SRC})
add_executable(tests  ${CORE_SRC} ${CUDA_SRC} ${TEST_SRC})

find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenImageDenoise REQUIRED)

add_library(glad ${CMAKE_SOURCE_DIR}/src/external/glad.c)
target_include_directories(glad PUBLIC ${CMAKE_SOURCE_DIR}/src/external)

foreach(tgt render tests)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/external
        ${CMAKE_SOURCE_DIR}/src/cuda
    )
    target_include_directories(${tgt} BEFORE PRIVATE
        ${GCC_INCLUDE_FIXED}
        ${GCC_INCLUDE}
    )
    # Allow .cu files to include project headers
    set_target_properties(${tgt} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${tgt} PRIVATE
        OpenMP::OpenMP_CXX
        OpenGL::GL
        glfw
        glad
        CUDA::cudart
        CUDA::curand
        OpenImageDenoise
    )
    if(UNIX AND NOT APPLE)
        target_link_libraries(${tgt} PRIVATE dl)
    endif()
endforeach()
