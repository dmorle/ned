cmake_minimum_required (VERSION 3.8)

include_directories("include")

add_library(
    libnedptx STATIC
    "source/graphops.cpp"
    "source/graphgen.cpp"

    "include/nedptx/graphops.h"
    "include/nedptx/graphgen.h"
)

find_package(CUDAToolkit)

target_include_directories(libnedptx PUBLIC "${CMAKE_SOURCE_DIR}/libned/include/")
target_link_libraries(libnedptx PRIVATE libned)
target_link_libraries(libnedptx PRIVATE CUDA::cudart)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

target_include_directories(libnedptx PUBLIC ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

set(LLVM_BUILD_DIR "C:/llvm/build")

llvm_map_components_to_libnames(
    llvm_libs
    aggressiveinstcombine
    analysis
    asmparser
    asmprinter
    binaryformat
    bitreader
    bitstreamreader
    bitwriter
    cfguard
    cfiverify
    codegen
    core
    coroutines
    coverage
    debuginfocodeview
    debuginfod
    debuginfodwarf
    debuginfogsym
    debuginfomsf
    debuginfopdb
    demangle
    diff
    dlltooldriver
    dwarflinker
    dwp
    executionengine
    exegesis
    exegesisx86
    extensions
    filecheck
    frontendopenacc
    frontendopenmp
    fuzzercli
    fuzzmutate
    globalisel
    instcombine
    instrumentation
    interfacestub
    interpreter
    ipo
    irreader
    jitlink
    libdriver
    lineeditor
    linker
    lto
    mc
    mca
    mcdisassembler
    mcjit
    mcparser
    mirparser
    nvptxcodegen
    nvptxdesc
    nvptxinfo
    objcarcopts
    objcopy
    object
    objectyaml
    option
    orcjit
    orcshared
    orctargetprocess
    passes
    profiledata
    remarks
    runtimedyld
    scalaropts
    selectiondag
    support
    symbolize
    tablegen
    tablegenglobalisel
    target
    testingsupport
    textapi
    transformutils
    vectorize
    windowsdriver
    windowsmanifest
    x86asmparser
    x86codegen
    x86desc
    x86disassembler
    x86info
    x86targetmca
    xray
)

# manually specifying the full library path because cmake was being stupid
foreach(llvm_lib ${llvm_libs})
    message(STATUS "Linking library: ${llvm_lib}")
    target_link_libraries(libnedptx PRIVATE "${LLVM_BUILD_DIR}/${CMAKE_BUILD_TYPE}/lib/${llvm_lib}.lib")
endforeach()

target_include_directories(libnedptx PRIVATE "${CMAKE_SOURCE_DIR}/libned/include/")
target_include_directories(libnedptx PRIVATE "C:/llvm/llvm-project/llvm/include")

find_package(ZLIB REQUIRED)
target_link_libraries(libnedptx PRIVATE ZLIB::ZLIB)

add_compile_definitions(TEST_DIR="${CMAKE_SOURCE_DIR}/backends/llvm/tests/")

set_property(TARGET libnedptx PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libnedptx PROPERTY CXX_STANDARD 20)

add_custom_command(TARGET libnedptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${LLVM_BUILD_DIR}/${CMAKE_BUILD_TYPE}/bin/lld-link.exe" "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(
    libnedptx_tests
    "tests/main.cpp"
)

target_include_directories(libnedptx_tests PRIVATE "${CMAKE_SOURCE_DIR}/libned/include/")
target_link_libraries(libnedptx_tests PRIVATE libnedptx)

set_property(TARGET libnedptx_tests PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libnedptx_tests PROPERTY CXX_STANDARD 20)
