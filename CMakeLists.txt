cmake_minimum_required (VERSION 3.12)

# project ("ned" LANGUAGES CXX CUDA)
project ("ned" LANGUAGES CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if (MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS_DEBUG "/MTd /Zi /Od /RTC1")
    else ()
        set(CMAKE_CXX_FLAGS_DEBUG "/MT /Zi /Od /OPT:NOREF /DEBUG:FULL /RTC1")
    endif ()
endif ()

# Include sub-projects.
add_subdirectory ("libned")
# add_subdirectory ("pyned")
# add_subdirectory ("cppned")
# add_subdirectory ("backends")
