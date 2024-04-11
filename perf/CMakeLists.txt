cmake_minimum_required(VERSION 3.20)

include(FetchContent)
FetchContent_Declare(
    benchmark
    URL ${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark-1.8.3.zip
)

project(PerfermanceTest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)

add_executable(
    PerfermanceLaserLine
    ${CMAKE_CURRENT_SOURCE_DIR}/perfLaserLine.cpp
)

target_link_libraries(
    PerfermanceLaserLine
    PRIVATE
    benchmark::benchmark
    slmaster
)

add_executable(
    PerfShiftGrayCodePattern
    ${CMAKE_CURRENT_SOURCE_DIR}/perfShiftGrayCodePattern.cpp
)

target_link_libraries(
    PerfShiftGrayCodePattern
    PRIVATE
    benchmark::benchmark
    slmaster
)