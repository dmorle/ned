add_library(
	nvmext_basic SHARED
	
	"include/basic.h"

	"source/basic.cpp"
)

set_property(TARGET nvmext_basic PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET nvmext_basic PROPERTY CXX_STANDARD 20)

target_include_directories(nvmext_basic PUBLIC "${CMAKE_SOURCE_DIR}/libned/include/")
target_include_directories(nvmext_basic PUBLIC "${CMAKE_SOURCE_DIR}/backends/nvm/include/")
target_include_directories(nvmext_basic PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(nvmext_basic PUBLIC libned)
target_link_libraries(nvmext_basic PUBLIC libnvm)

add_custom_command(TARGET nvmext_basic POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_BINARY_DIR}/nvmext_basic.dll" "${CMAKE_CURRENT_BINARY_DIR}/../")
