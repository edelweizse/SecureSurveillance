cmake_minimum_required(VERSION 3.16)

add_library(ss_core)

# Headers
file(GLOB_RECURSE SS_CORE_HEADERS
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)

# Sources
file(GLOB_RECURSE SS_CORE_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
)
target_sources(ss_core PRIVATE ${SS_CORE_SOURCES} ${SS_CORE_HEADERS})
target_include_directories(ss_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_SOURCE_DIR}/libs)

# Compile Options
target_compile_features(ss_core PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ss_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Dependencies
# OpenCV
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    target_link_libraries(ss_core PUBLIC ${OpenCV_LIBS})
    target_compile_definitions(ss_core PUBLIC SS_HAS_OPENCV=1)
else()
    target_compile_definitions(ss_core PUBLIC SS_HAS_OPENCV=0)
endif()

# GStreamer
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_check_modules(GST_VIDEO REQUIRED IMPORTED_TARGET gstreamer-video-1.0)

target_link_libraries(ss_core PRIVATE
        PkgConfig::GST
        PkgConfig::GST_APP
        PkgConfig::GST_VIDEO
)
target_compile_definitions(ss_core PUBLIC SS_HAS_GST=1)

# YAML
find_package(yaml-cpp REQUIRED)
target_link_libraries(ss_core PRIVATE yaml-cpp)

# Threads
find_package(Threads REQUIRED)
target_link_libraries(ss_core PRIVATE Threads::Threads)

# NCNN
find_package(ncnn REQUIRED)
target_link_libraries(ss_core PRIVATE ncnn)