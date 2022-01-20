include_directories("include")

add_library(
    libned STATIC

    "source/errors.cpp"

    "source/util/libs.cpp"

    "source/core/config.cpp"
    "source/core/setup.cpp"
    "source/core/graph.cpp"

    "source/lang/ast.cpp"
    "source/lang/bytecode.cpp"
    "source/lang/interp.cpp"
    "source/lang/lexer.cpp"
    "source/lang/obj.cpp"
    "source/lang/compiler.cpp"

    "include/ned/errors.h"

    "include/ned/util/libs.h"

    "include/ned/core/config.h"
    "include/ned/core/setup.h"
    "include/ned/core/graph.h"
    
    "include/ned/lang/ast.h"
    "include/ned/lang/bytecode.h"
    "include/ned/lang/interp.h"
    "include/ned/lang/lexer.h"
    "include/ned/lang/obj.h"
    "include/ned/lang/compiler.h"
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
