cmake_minimum_required(VERSION 3.0)

project(camera)

set(CAMERA_ROOT_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/common/camera.h
                       ${CMAKE_CURRENT_SOURCE_DIR}/common/cameraFactory.h
                       ${CMAKE_CURRENT_SOURCE_DIR}/common/typeDef.h)
set(CAMERA_ROOT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common/cameraFactory.cpp ${CMAKE_CURRENT_SOURCE_DIR}/common/safeQueue.hpp)

find_package(OpenCV REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/module/huaray_camera)

if(BUILD_SLMASTER_SHARED)
    add_library(camera SHARED)
    target_compile_definitions(camera PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(camera PRIVATE -DDLL_EXPORTS)
else()
    add_library(camera)
endif()

target_sources(camera PUBLIC ${CAMERA_ROOT_HEADER} ${CAMERA_ROOT_SOURCES})

target_include_directories(camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common ${CMAKE_CURRENT_SOURCE_DIR}/module/huaray_camera/include)

target_link_libraries(
    camera
    PUBLIC
    huarayCamera
    ${OpenCV_LIBRARIES}
)
