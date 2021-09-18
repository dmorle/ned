cmake_minimum_required (VERSION 3.8)

# Not nessisary, only for intellisense
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/libned/include")
include_directories("C:/Program Files/Python38/include")

link_directories("C:/Program Files/Python38/libs")

add_compile_definitions(PYNED_ENV_PTH="${CMAKE_CURRENT_SOURCE_DIR}/pyned-env/Lib/site-packages")

set(PYNED_HEADERS
    "include/pyned/lang/methods.h"
    "include/pyned/lang/ast.h"
    "include/pyned/lang/obj.h"
    
    "include/pyned/core/graph.h"
)

add_library(
    libpynedc STATIC
    "source/core/module.cpp"
    "source/core/graph.cpp"
    ${PYNED_HEADERS}
)

set_property(TARGET libpynedc PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libpynedc PROPERTY CXX_STANDARD 20)

add_library(
    libpynedl STATIC
    "source/lang/module.cpp"
    "source/lang/methods.cpp"
    "source/lang/ast.cpp"
    "source/lang/obj.cpp"
    ${PYNED_HEADERS}
)

set_property(TARGET libpynedl PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libpynedl PROPERTY CXX_STANDARD 20)
target_link_libraries(libpynedl PUBLIC libpynedc)

target_link_libraries(libpynedc PRIVATE libned)
target_link_libraries(libpynedl PRIVATE libned)
if (${CMAKE_BUILD_TYPE} EQUAL "Debug")
    target_link_libraries(libpynedc PRIVATE "python38_d")
    target_link_libraries(libpynedl PUBLIC "python38_d")
else ()
    target_link_libraries(libpynedc PRIVATE "python38")
    target_link_libraries(libpynedl PUBLIC "python38")
endif ()

configure_file("setup.py" "setup.py")
file(COPY "pyned" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# just for testing

add_executable(
    pynedc
    "source/core/module.cpp"
)

set_property(TARGET pynedc PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET pynedc PROPERTY CXX_STANDARD 20)

target_link_libraries(pynedc PRIVATE libpynedc)

add_executable(
    pynedl
    "source/lang/module.cpp"
)

set_property(TARGET pynedl PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET pynedl PROPERTY CXX_STANDARD 20)

target_link_libraries(pynedl PRIVATE libpynedl)
