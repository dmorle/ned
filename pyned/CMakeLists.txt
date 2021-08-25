cmake_minimum_required (VERSION 3.8)

# Not nessisary, only for intellisense
add_executable(
    pyned
    "source/main.cpp"

    "source/core/module.cpp"

    "source/lang/module.cpp"
    "source/lang/methods.cpp"
    "source/lang/ned_ast.cpp"

    "include/pyned/pyned.h"

    "include/pyned/lang/methods.h"
    "include/pyned/lang/ast.h"
)

target_link_directories(pyned PRIVATE "C:/Program Files/Python38/libs")
target_link_libraries(pyned libned)
target_link_libraries(pyned "python38_d")

target_include_directories(pyned PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(pyned PRIVATE "${CMAKE_SOURCE_DIR}/libned/include")
target_include_directories(pyned PRIVATE "C:/Program Files/Python38/include")

configure_file("setup.py" "setup.py")
file(COPY "pyned" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
