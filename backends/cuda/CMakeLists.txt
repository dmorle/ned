add_subdirectory("type_dispatch_helpers")

cmake_minimum_required (VERSION 3.8)

include_directories("include")

add_library(
    libcuned STATIC
    
    "source/cugraph.cpp"
    "source/nodes/basic_ops.cu"
    "source/nodes/matops.cu"
    "source/nodes/activations.cu"

    "include/cuned/cugraph.h"
    "include/cuned/cunodes.h"
)

find_package(CUDAToolkit)

target_include_directories(libcuned PRIVATE "${CMAKE_SOURCE_DIR}/libned/include/")
target_link_libraries(libcuned PRIVATE libned)
target_link_libraries(libcuned PRIVATE CUDA::cudart)

set_property(TARGET libcuned PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libcuned PROPERTY CXX_STANDARD 20)
set_property(TARGET libcuned PROPERTY CUDA_STANDARD 17)

add_compile_definitions(TEST_DIR="${CMAKE_SOURCE_DIR}/backends/cuda/tests/")

add_executable(
    libcuned_tests
    "tests/main.cpp"
)

target_include_directories(libcuned_tests PRIVATE "${CMAKE_SOURCE_DIR}/libned/include/")
target_link_libraries(libcuned_tests PRIVATE libcuned)

set_property(TARGET libcuned_tests PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libcuned_tests PROPERTY CXX_STANDARD 20)
set_property(TARGET libcuned_tests PROPERTY CUDA_STANDARD 17)
