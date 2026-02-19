cmake_minimum_required(VERSION 3.16)

add_executable(ss_config_tests
        config_tests.cpp
        ${CMAKE_SOURCE_DIR}/core/src/common/config.cpp
        ${CMAKE_SOURCE_DIR}/core/src/common/replicate.cpp
)

target_include_directories(ss_config_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
)

target_compile_features(ss_config_tests PRIVATE cxx_std_17)

find_package(yaml-cpp REQUIRED)
target_link_libraries(ss_config_tests PRIVATE yaml-cpp)

add_test(NAME ss_config_tests COMMAND ss_config_tests)
