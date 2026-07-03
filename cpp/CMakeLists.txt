cmake_minimum_required(VERSION 3.20)

project(tzaar_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# ─── 源碼檔案 ─────────────────────────────────────────────
set(TZAAR_SOURCES
    src/module.cpp
    src/core/board.cpp
    src/core/game.cpp
    src/core/action.cpp
    src/state/phase_state.cpp
    src/features/cnn_builder.cpp
    src/mcts/search.cpp
)

set(TZAAR_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
pybind11_add_module(tzaar_cpp ${TZAAR_SOURCES})
target_include_directories(tzaar_cpp PRIVATE ${TZAAR_INCLUDE_DIRS})
target_compile_features(tzaar_cpp PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(tzaar_cpp PRIVATE /W4 /EHsc /utf-8)
else()
  target_compile_options(tzaar_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

