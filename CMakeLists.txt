cmake_minimum_required(VERSION 3.16)
project(veilsight LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VEILSIGHT_BUILD_TESTS "Build tests" ON)

add_subdirectory(core)
add_subdirectory(apps)

if (VEILSIGHT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()