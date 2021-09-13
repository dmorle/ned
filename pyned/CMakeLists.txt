cmake_minimum_required (VERSION 3.8)

# Not nessisary, only for intellisense
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/libned/include")
include_directories("C:/Program Files/Python38/include")

link_directories("C:/Program Files/Python38/libs")
link_libraries(libned)
if (${CMAKE_BUILD_TYPE} EQUAL "Debug")
    link_libraries("python38_d")
else ()
    link_libraries("python38")
endif ()

set(PYNED_HEADERS
    "include/pyned/lang/methods.h"
    "include/pyned/lang/ast.h"
)

add_executable(
    pynedl
    "source/lang/module.cpp"
    "source/lang/methods.cpp"
    "source/lang/ast.cpp"
    ${PYNED_HEADERS}
)

set_property(TARGET pynedl PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET pynedl PROPERTY CXX_STANDARD 20)

add_executable(
    pynedc
    "source/core/module.cpp"
    ${PYNED_HEADERS}
)

set_property(TARGET pynedc PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET pynedc PROPERTY CXX_STANDARD 20)

configure_file("setup.py" "setup.py")
file(COPY "pyned" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
