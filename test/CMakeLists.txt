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
    Camera
    GTest::gtest_main
)

add_executable(
    TestProjectorDlpcApi4710
    ${CMAKE_CURRENT_SOURCE_DIR}/testProjectorDlpcApi4710.cpp
)

target_link_libraries(
    TestProjectorDlpcApi4710
    PUBLIC
    Projector
    GTest::gtest_main
)

add_executable(
    testBinocularCamera
    ${CMAKE_CURRENT_SOURCE_DIR}/testBinocularCamera.cpp
)

target_link_libraries(
    testBinocularCamera
    PUBLIC
    slmaster
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(TestHuarayCamera)
gtest_discover_tests(TestProjectorDlpcApi4710)
