cmake_minimum_required(VERSION 3.20)

project(SLMaster)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Release)
#Release模式必须使用以下属性
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} /O2 /Zi")

add_subdirectory(FluentUI)
add_subdirectory(QuickQanava)
add_subdirectory(src)
add_subdirectory(gui)

option(BUILD_TEST "build the test case with google_test." ON)

if(BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
    message(NOTICE "build test case...")
endif()

option(BUILD_PERF "build the perfermance case with google benchmark." ON)

if(BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/perf)
    message(NOTICE "build perfermance case...")
endif()