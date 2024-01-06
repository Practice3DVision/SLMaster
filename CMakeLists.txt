cmake_minimum_required(VERSION 3.0)

project(SLMaster)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Release)
#Release模式必须使用以下属性
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} /DEBUG /O2 /ZI")

add_subdirectory(FluentUI)
add_subdirectory(device)
add_subdirectory(src)
add_subdirectory(gui)

option(BUILD_TEST "build the test case with google_test." OFF)

if(BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
    message("build test case...")
endif()
