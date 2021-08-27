cmake_minimum_required (VERSION 3.8)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/libned/include")

add_library(
    cppned STATIC

)

target_link_libraries(cppned libned)

add_executable(
    cppned_tests
    "main.cpp"
)

target_link_libraries(cppned_tests cppned)
