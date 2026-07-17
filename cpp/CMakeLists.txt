cmake_minimum_required(VERSION 3.20)
project(roblox_character_detector_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(RCD_BUILD_LIVE "Build the Windows live overlay tool" ${WIN32})

# ---------------------------------------------------------------- ONNX Runtime
# Point ONNXRUNTIME_ROOT (cmake -DONNXRUNTIME_ROOT=... or env var) at an
# extracted ONNX Runtime package that contains include/ and lib/.
#   * Windows GPU (CUDA/TensorRT): Microsoft.ML.OnnxRuntime.Gpu or the
#     onnxruntime-win-x64-gpu-*.zip release archive
#   * Windows DirectML (any GPU vendor): Microsoft.ML.OnnxRuntime.DirectML
#   * Linux: onnxruntime-linux-x64-*.tgz
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime package")
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include"
        "${ONNXRUNTIME_ROOT}/build/native/include"  # nuget layout
  PATH_SUFFIXES onnxruntime)
find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib"
        "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native")
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR
    "ONNX Runtime not found. Download a package (see cpp/README.md) and "
    "configure with -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()
get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIB}" DIRECTORY)
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

find_package(Threads REQUIRED)

# ------------------------------------------------------------------- rcd_core
add_library(rcd_core STATIC
  src/detector/yolo_detector.cpp
  src/render/draw.cpp)
target_include_directories(rcd_core
  PUBLIC src "${ONNXRUNTIME_INCLUDE_DIR}"
  PRIVATE third_party/font8x8)
target_link_libraries(rcd_core PUBLIC "${ONNXRUNTIME_LIB}" Threads::Threads)
if(WIN32)
  target_compile_definitions(rcd_core PRIVATE UNICODE _UNICODE NOMINMAX
    WIN32_LEAN_AND_MEAN)
  if(MINGW)
    # MinGW lacks DirectML.h; enable DML via a manual declaration when the
    # linked onnxruntime.dll is a DirectML build.
    target_compile_definitions(rcd_core PRIVATE RCD_DML_MANUAL_DECL)
  endif()
endif()
if(MSVC)
  target_compile_options(rcd_core PRIVATE /W4)
else()
  target_compile_options(rcd_core PRIVATE -Wall -Wextra)
endif()

# -------------------------------------------------------------------- rcd_cli
add_executable(rcd_cli src/cli/main_cli.cpp)
target_include_directories(rcd_cli SYSTEM PRIVATE third_party/stb)
target_link_libraries(rcd_cli PRIVATE rcd_core)
set_target_properties(rcd_cli PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

# ------------------------------------------------------------------- rcd_live
if(RCD_BUILD_LIVE AND WIN32)
  add_executable(rcd_live
    src/win/main_live.cpp
    src/win/dxgi_capture.cpp
    src/win/overlay_window.cpp)
  target_link_libraries(rcd_live PRIVATE rcd_core
    d3d11 dxgi d2d1 dwrite user32)
  target_compile_definitions(rcd_live PRIVATE UNICODE _UNICODE NOMINMAX
    WIN32_LEAN_AND_MEAN)
elseif(RCD_BUILD_LIVE)
  message(STATUS "rcd_live is Windows-only; building rcd_cli only")
endif()

# Put onnxruntime's DLLs next to the executables on Windows.
if(WIN32)
  file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_LIB_DIR}/*.dll"
       "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native/*.dll")
  foreach(tgt IN ITEMS rcd_cli rcd_live)
    if(TARGET ${tgt} AND ONNXRUNTIME_DLLS)
      add_custom_command(TARGET ${tgt} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_DLLS} "$<TARGET_FILE_DIR:${tgt}>")
    endif()
  endforeach()
endif()
