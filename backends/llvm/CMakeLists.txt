cmake_minimum_required (VERSION 3.8)

include_directories("include")

add_library(
    libnedvm STATIC
    "source/graphgen.cpp"
    "include/nedvm/common.h"
    "include/nedvm/graphgen.h"
)

target_include_directories(libnedvm PRIVATE "${CMAKE_SOURCE_DIR}/libned/include/")
target_link_libraries(libnedvm PRIVATE libned)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

target_include_directories(libnedvm PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(libnedvm PUBLIC "C:/llvm-build/llvm/llvm/include/")
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

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
    debuginfodwarf
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
    objcarcopts
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
    windowsmanifest
    x86asmparser
    x86codegen
    x86desc
    x86disassembler
    x86info
    xray
)

# manually specifying the full library path because cmake was being stupid
foreach(llvm_lib ${llvm_libs})
    message(STATUS "Linking library: ${llvm_lib}")
    target_link_libraries(libnedvm PRIVATE "C:/llvm-build/build/Release/lib/${llvm_lib}.lib")
endforeach()

add_compile_definitions(TEST_DIR="${CMAKE_SOURCE_DIR}/backends/llvm/tests/")

add_executable(
    libnedvm_tests
    "tests/main.cpp"
)

target_include_directories(libnedvm_tests PRIVATE "${CMAKE_SOURCE_DIR}/libned/include/")
target_link_libraries(libnedvm_tests PRIVATE libnedvm)

set_property(TARGET libnedvm_tests PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libnedvm_tests PROPERTY CXX_STANDARD 20)
