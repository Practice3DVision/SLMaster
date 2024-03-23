cmake_minimum_required(VERSION 3.20)

project(cameras)

find_package(OpenCV COMPONENTS core cudev)
if(OpenCV_cudev_FOUND)
    add_definitions(-DOPENCV_WITH_CUDA_MODULE)
endif()

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(BUILD_SLMASTER_SHARED)
    add_library(cameras SHARED)
    target_compile_definitions(cameras PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(cameras PRIVATE -DDLL_EXPORTS)
    set_target_properties(cameras PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    add_library(cameras STATIC)
endif()

target_sources(cameras PRIVATE ${SOURCES})
target_include_directories(cameras PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cameras PUBLIC algorithm camera projector)