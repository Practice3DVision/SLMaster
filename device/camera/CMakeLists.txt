cmake_minimum_required(VERSION 3.0)

project(Camera)

set(CAMERA_ROOT_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/common/camera.h
                       ${CMAKE_CURRENT_SOURCE_DIR}/common/cameraFactory.h
                       ${CMAKE_CURRENT_SOURCE_DIR}/common/typeDef.h)
set(CAMERA_ROOT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common/cameraFactory.cpp ${CMAKE_CURRENT_SOURCE_DIR}/common/safeQueue.hpp)

find_package(OpenCV REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/module/huaray_camera)

if(BUILD_DEVICE_SHARED)
    add_library(Camera SHARED)
    target_compile_definitions(Camera PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(Camera PRIVATE -DDLL_EXPORTS)
else()
    add_library(Camera)
endif()

target_sources(Camera PUBLIC ${CAMERA_ROOT_HEADER} ${CAMERA_ROOT_SOURCES})

target_include_directories(Camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common ${CMAKE_CURRENT_SOURCE_DIR}/module/huaray_camera/include)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    huarayCamera
    ${OpenCV_LIBRARIES}
)
