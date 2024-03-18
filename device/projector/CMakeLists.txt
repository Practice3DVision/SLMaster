cmake_minimum_required(VERSION 3.0)

project(Projector)

set(PROJECTOR_ROOT_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/common/projector.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/common/projectorFactory.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/common/typeDef.h)
set(PROJECTOR_ROOT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common/projectorFactory.cpp)

find_package(OpenCV REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/module/projector_dlpc_api)

if(BUILD_DEVICE_SHARED)
    add_library(Projector SHARED)
    target_compile_definitions(Projector PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(Projector PRIVATE -DDLL_EXPORTS)
else()
    add_library(Projector)
endif()

target_sources(Projector PUBLIC ${PROJECTOR_ROOT_HEADER} ${PROJECTOR_ROOT_SOURCES})

target_include_directories(Projector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common ${CMAKE_CURRENT_SOURCE_DIR}/module/projector_dlpc_api/include)

target_link_libraries(
    Projector
    PUBLIC
    projectorDlpcApi
)
