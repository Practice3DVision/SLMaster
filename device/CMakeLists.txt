cmake_minimum_required(VERSION 3.20)

#don't recommand.
option(BUILD_DEVICE_SHARED "build device shared lib..." OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/camera)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/projector)
