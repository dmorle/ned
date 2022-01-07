cmake_minimum_required (VERSION 3.12)

project ("ned" LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Include sub-projects.
add_subdirectory ("libned")
add_subdirectory ("pyned")
# add_subdirectory ("cppned")
# add_subdirectory ("backends")
