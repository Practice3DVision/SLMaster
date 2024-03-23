cmake_minimum_required(VERSION 3.0)

project(projector)

set(PROJECTOR_ROOT_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/common/projector.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/common/projectorFactory.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/common/typeDef.h)
set(PROJECTOR_ROOT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common/projectorFactory.cpp)

find_package(OpenCV REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/module/projector_dlpc_api)

if(BUILD_SLMASTER_SHARED)
    add_library(projector SHARED)
    target_compile_definitions(projector PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(projector PRIVATE -DDLL_EXPORTS)
else()
    add_library(projector)
endif()

target_sources(projector PUBLIC ${PROJECTOR_ROOT_HEADER} ${PROJECTOR_ROOT_SOURCES})

target_include_directories(projector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common ${CMAKE_CURRENT_SOURCE_DIR}/module/projector_dlpc_api/include)

target_link_libraries(
    projector
    PUBLIC
    projectorDlpcApi
)
