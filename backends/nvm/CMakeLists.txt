cmake_minimum_required (VERSION 3.12)

include_directories("include")

add_library(
    libnvm STATIC
    "source/common.cpp"
    "source/graphgen.cpp"
    "source/runtime.cpp"

    "include/nvm/common.h"
    "include/nvm/graphgen.h"
    "include/nvm/runtime.h"
)

target_include_directories(libnvm PUBLIC "${CMAKE_SOURCE_DIR}/libned/include/")
target_link_libraries(libnvm PUBLIC libned)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

target_include_directories(libnvm PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(libnvm PUBLIC "C:/llvm/llvm-project/llvm/include/")
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
    debuginfologicalview
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
    frontendhlsl
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
    irprinter
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
    targetparser
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
    target_link_libraries(libnvm PRIVATE "${LLVM_BUILD_DIR}/${CMAKE_BUILD_TYPE}/lib/${llvm_lib}.lib")
endforeach()

find_package(ZLIB REQUIRED)
target_link_libraries(libnvm PRIVATE ZLIB::ZLIB)

add_compile_definitions(TESTS_DIR="${CMAKE_SOURCE_DIR}/backends/nvm/tests/")

set_property(TARGET libnvm PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libnvm PROPERTY CXX_STANDARD 20)

add_custom_command(TARGET libnvm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${LLVM_BUILD_DIR}/${CMAKE_BUILD_TYPE}/bin/llc.exe" "${CMAKE_CURRENT_BINARY_DIR}")
add_custom_command(TARGET libnvm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${LLVM_BUILD_DIR}/${CMAKE_BUILD_TYPE}/bin/lld-link.exe" "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(
    libnvm_tests
    "tests/main.cpp"
)

target_include_directories(libnvm_tests PRIVATE "${CMAKE_SOURCE_DIR}/libned/include/")
target_link_libraries(libnvm_tests PRIVATE libnvm)

set_property(TARGET libnvm_tests PROPERTY LINKER_LANGUAGE CXX)
set_property(TARGET libnvm_tests PROPERTY CXX_STANDARD 20)

add_subdirectory("extensions")

set(nvmexts "basic")

foreach (extname ${nvmexts})
    add_dependencies(libnvm_tests "nvmext_${extname}")
endforeach ()

add_executable(
    playground
    "tests/playground.cpp"
)

# For ned/util/libs.h
target_include_directories(playground PRIVATE "${CMAKE_SOURCE_DIR}/libned/include/")
target_link_libraries(playground PRIVATE libned)
