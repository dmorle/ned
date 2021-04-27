cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

include_directories("include")

add_library (
	libnn STATIC
	"src/ops/conv2d.cu"
	"src/ops/linear.cu"

	"include/libnn/core/tensor.h"
	"include/libnn/ops/op.h"
	"include/libnn/ops/linear.h"
	"include/libnn/ops/conv2d.h"
)

set_property(TARGET libnn PROPERTY CXX_STANDARD 17)
set_target_properties (libnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
