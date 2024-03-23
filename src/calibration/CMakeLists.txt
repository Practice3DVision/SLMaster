cmake_minimum_required(VERSION 3.20)

project(calibration)

find_package(Eigen3 REQUIRED)
find_package(OpenCV COMPONENTS core REQUIRED)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(BUILD_SLMASTER_SHARED)
    add_library(calibration SHARED)
    target_compile_definitions(calibration PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(calibration PRIVATE -DDLL_EXPORTS)
    set_target_properties(calibration PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    add_library(calibration STATIC)
endif()

target_sources(calibration PRIVATE ${SOURCES})
target_include_directories(calibration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(calibration PUBLIC ${OpenCV_LIBRARIES})