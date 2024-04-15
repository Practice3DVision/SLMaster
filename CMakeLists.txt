cmake_minimum_required(VERSION 3.20)

project(SLMaster)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} /O2 /Zi")

add_subdirectory(src)

option(BUILD_GUI "build the gui." ON)

if(BUILD_GUI)
    add_subdirectory(gui)
    message(NOTICE "[SLMaster]: build gui...")
endif()

option(BUILD_TEST "build the test case with google_test." ON)

if(BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
    message(NOTICE "[SLMaster]: build test case...")
endif()

option(BUILD_PERF "build the perfermance case with google benchmark." ON)

if(BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/perf)
    message(NOTICE "[SLMaster]: build perfermance case...")
endif()