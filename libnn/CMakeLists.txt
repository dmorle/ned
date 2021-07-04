cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

include_directories("include")

add_executable (
	libnn
	"src/main.cpp"

	"include/libnn/core/graph.h"
	"include/libnn/core/tensor.h"
)

set_property(TARGET libnn PROPERTY CXX_STANDARD 17)
