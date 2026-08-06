cmake_minimum_required(VERSION 3.20)

project(tzaar_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# ─── NVTX (NVIDIA Nsight Tools Extension) ──────────────
# 搜尋順序：
#   1. 專案內建 nvtx_include/（手動下載放置）
#   2. CUDA toolkit 中的 nvtx3
if(EXISTS "${CMAKE_SOURCE_DIR}/../nvtx_include/nvtx3/nvToolsExt.h")
  # 專案內建的 nvtx header
  set(NVTX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../nvtx_include")
  message(STATUS "NVTX found (local): ${NVTX_INCLUDE_DIR}")
elseif(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/include/nvtx3/nvToolsExt.h")
  # CUDA toolkit 中的 nvtx3
  set(NVTX_INCLUDE_DIR "$ENV{CUDA_PATH}/include")
  message(STATUS "NVTX found (CUDA): ${NVTX_INCLUDE_DIR}")
elseif(EXISTS "/usr/local/cuda/include/nvtx3/nvToolsExt.h")
  # Linux CUDA
  set(NVTX_INCLUDE_DIR "/usr/local/cuda/include")
  message(STATUS "NVTX found (CUDA Linux): ${NVTX_INCLUDE_DIR}")
endif()

if(NVTX_INCLUDE_DIR)
  include_directories(${NVTX_INCLUDE_DIR})
  add_compile_definitions(TZAAR_USE_NVTX)
  message(STATUS "NVTX profiling ENABLED")
else()
  message(STATUS "NVTX not found; profiling markers disabled.")
endif()

# ─── 源碼檔案 ─────────────────────────────────────────────
set(TZAAR_SOURCES
    src/module.cpp
    src/core/board.cpp
    src/core/game.cpp
    src/core/action.cpp
        src/state/phase_state.cpp
    src/features/cnn_builder.cpp
    src/mcts/search.cpp
    src/mcts/search_manager.cpp
    src/mcts/bench.cpp
)

set(TZAAR_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
pybind11_add_module(tzaar_cpp ${TZAAR_SOURCES})
target_include_directories(tzaar_cpp PRIVATE ${TZAAR_INCLUDE_DIRS})
target_compile_features(tzaar_cpp PRIVATE cxx_std_17)

if(MSVC)
  # NOMINMAX 防止 <windows.h> 的 min/max 巨集與 std::min/std::max 衝突
  target_compile_options(tzaar_cpp PRIVATE /W4 /EHsc /utf-8 /DNOMINMAX)
else()
  target_compile_options(tzaar_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

