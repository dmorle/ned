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

target_include_directories(libcuned PUBLIC "${CMAKE_SOURCE_DIR}/libned/include/")
target_link_libraries(libcuned PUBLIC libned)
target_link_libraries(libcuned PRIVATE CUDA::cudart)

set_property(TARGET libcuned PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libcuned PROPERTY CXX_STANDARD 20)
set_property(TARGET libcuned PROPERTY CUDA_STANDARD 17)
