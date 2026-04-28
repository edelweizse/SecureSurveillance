cmake_minimum_required(VERSION 3.16)

add_executable(veilsight_core_service main.cpp)

target_link_libraries(veilsight_core_service PRIVATE veilsight_core)