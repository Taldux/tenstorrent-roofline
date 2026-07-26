cmake_minimum_required(VERSION 3.20)
project(tt_benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# link to my own build of tt-metalium in the hawaii cluster, you may need to build it 
# first and change this if you want to run anywhere else
set(TT_METAL_HOME "/csghome/sc323/tt-metal"        CACHE PATH "tt-metal source root")
set(TT_BUILD      "${TT_METAL_HOME}/build_Release" CACHE PATH "tt-metal build directory")
set(CPM           "${TT_METAL_HOME}/.cpmcache"      CACHE PATH "CPM cache directory")

set(DEPS "${TT_BUILD}/_deps")

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(TT_RPATH_DIRS
    "${TT_BUILD}/tt_metal"
    "${TT_BUILD}/tt_stl"
    "${TT_BUILD}/tt_metal/third_party/umd/lib"
)
set(CMAKE_INSTALL_RPATH ${TT_RPATH_DIRS})
set(CMAKE_BUILD_RPATH   ${TT_RPATH_DIRS})

# Some imported targets from the build release of tt-metal
add_library(tt_umd SHARED IMPORTED GLOBAL)
set_target_properties(tt_umd PROPERTIES
    IMPORTED_LOCATION
        "${TT_BUILD}/tt_metal/third_party/umd/lib/libtt-umd.so"
)

add_library(tt_stl SHARED IMPORTED GLOBAL)
set_target_properties(tt_stl PROPERTIES
    IMPORTED_LOCATION "${TT_BUILD}/tt_stl/libtt_stl.so"
)

add_library(spdlog STATIC IMPORTED GLOBAL)
set_target_properties(spdlog PROPERTIES
    IMPORTED_LOCATION "${DEPS}/spdlog-build/libspdlog.a"
)

add_library(yaml_cpp STATIC IMPORTED GLOBAL)
set_target_properties(yaml_cpp PROPERTIES
    IMPORTED_LOCATION "${DEPS}/yaml-cpp-build/libyaml-cpp.a"
)

add_library(flatbuffers STATIC IMPORTED GLOBAL)
set_target_properties(flatbuffers PROPERTIES
    IMPORTED_LOCATION "${DEPS}/flatbuffers-build/libflatbuffers.a"
)

# main tt-metalium library
add_library(tt_metal SHARED IMPORTED GLOBAL)
set_target_properties(tt_metal PROPERTIES
    IMPORTED_LOCATION    "${TT_BUILD}/tt_metal/libtt_metal.so"
    INTERFACE_LINK_LIBRARIES "tt_umd;tt_stl;spdlog;yaml_cpp;flatbuffers"
)

# Include paths for compiling against tt-metalium out-of-tree
target_include_directories(tt_metal INTERFACE
    # Core public API
    "${TT_METAL_HOME}"
    "${TT_METAL_HOME}/tt_metal"
    "${TT_METAL_HOME}/tt_metal/api"
    "${TT_METAL_HOME}/tt_metal/api/tt-metalium"
    "${TT_METAL_HOME}/tt_stl"
    # UMD device API
    "${TT_METAL_HOME}/tt_metal/third_party/umd/device/api"
    # Host/device shared headers
    "${TT_METAL_HOME}/tt_metal/hostdevcommon/api"
    # Kernel-side hardware headers (needed for host-side TensorAccessorArgs)
    "${TT_METAL_HOME}/tt_metal/hw/inc/api"
    # Generated / post-build headers
    "${TT_BUILD}/tt_metal"
    # CPM vendored dependency headers
    "${CPM}/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include"
    "${CPM}/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include"
    "${CPM}/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include"
    "${CPM}/flatbuffers/2c4062bffa52fa4157b1b4deeae73395df475fda/include"
    "${CPM}/range-v3/0_12_0_patched/include"
    "${CPM}/yaml-cpp/0_8_0_upstream_patched/include"
    "${CPM}/tt-logger/d2339ce68562cae34cd95f3fece7fd94eb0529b7/include"
    "${CPM}/enchantum/2fb7ab238e36c101b9848892ddb6382276b65837/enchantum/include"
)

# helper macro
macro(add_benchmark NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE tt_metal)
    set_target_properties(${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endmacro()

# benchmarks
add_benchmark(peak_bandwidth benchmarks/peak_bandwidth/peak_bandwidth.cpp)
add_benchmark(peak_flops     benchmarks/peak_flops/peak_flops.cpp)
add_benchmark(l1_bandwidth   benchmarks/l1_bandwidth/l1_bandwidth.cpp)