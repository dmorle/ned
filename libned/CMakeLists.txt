cmake_minimum_required (VERSION 3.8)

include_directories("include")

add_library(
    libned STATIC

    "source/lang/eval.cpp"
    "source/lang/obj.cpp"
    "source/lang/ast.cpp"
    "source/lang/lexer.cpp"
    
    "include/ned/lang/eval.h"
    "include/ned/lang/obj.h"
    "include/ned/lang/ast.h"
    "include/ned/lang/lexer.h"

    "include/ned/core/tensor.h"
    "include/ned/core/graph.h"
)

set_property(TARGET libned PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libned PROPERTY CXX_STANDARD 20)

add_executable(
    libned_tests
    "tests/main.cpp"
)

target_compile_definitions(libned_tests PUBLIC TESTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/")
target_link_libraries(libned_tests libned)

set_property(TARGET libned_tests PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libned_tests PROPERTY CXX_STANDARD 20)
