cmake_minimum_required(VERSION 3.0)

project(huarayCamera)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
source_group("HEADERS" FILES ${HEADERS})
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
source_group("SOURCES" FILES ${SOURCES})

if(BUILD_CAMERA_SHARED)
    add_library(huarayCamera SHARED)
    target_compile_definitions(huarayCamera PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(huarayCamera PRIVATE -DDLL_EXPORTS)
else()
    add_library(huarayCamera)
endif()

target_sources(huarayCamera PRIVATE ${HEADERS} ${SOURCES})

target_link_directories(huarayCamera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)

target_include_directories(
    huarayCamera
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    huarayCamera
    PUBLIC
    ${OpenCV_LIBRARIES}
    MVSDKmd
)
