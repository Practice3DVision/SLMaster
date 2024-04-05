cmake_minimum_required(VERSION 3.0)

project(TestSLMaster)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest-1.12.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(
    TestHuarayCamera
    ${CMAKE_CURRENT_SOURCE_DIR}/testHuarayCamera.cpp
)

target_link_libraries(
    TestHuarayCamera
    PUBLIC
    camera
    GTest::gtest_main
)

add_executable(
    TestProjectorDlpcApi4710
    ${CMAKE_CURRENT_SOURCE_DIR}/testProjectorDlpcApi4710.cpp
)

target_link_libraries(
    TestProjectorDlpcApi4710
    PUBLIC
    projector
    GTest::gtest_main
)

add_executable(
    TestProjectorDlpcApi3010
    ${CMAKE_CURRENT_SOURCE_DIR}/testProjectorDlpcApi3010.cpp
)

target_link_libraries(
    TestProjectorDlpcApi3010
    PUBLIC
    projector
    GTest::gtest_main
)

add_executable(
    TestLasterLine
    ${CMAKE_CURRENT_SOURCE_DIR}/testLaserLine.cpp
)

target_link_libraries(
    TestLasterLine
    PUBLIC
    slmaster
    GTest::gtest_main
)

include(GoogleTest)