cmake_minimum_required (VERSION 3.8)

add_compile_definitions(SOURCE_DIR="${CMAKE_SOURCE_DIR}/backends/cuda/source/nodes/")

add_executable(
    basic_ops_helper
    "basic_ops.cpp"
)

set_property(TARGET basic_ops_helper PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET basic_ops_helper PROPERTY CXX_STANDARD 20)
