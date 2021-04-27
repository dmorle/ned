include_directories("include")
include_directories("${PROJECT_SOURCE_DIR}/libnn/include")

add_executable (
	nntest
	"src/main.cpp"
)

set_property(TARGET nntest PROPERTY CXX_STANDARD 20)
target_link_libraries(nntest libnn)
