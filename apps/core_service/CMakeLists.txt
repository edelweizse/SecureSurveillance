cmake_minimum_required(VERSION 3.16)

add_executable(ss_core_service main.cpp)

target_link_libraries(ss_core_service PRIVATE ss_core)