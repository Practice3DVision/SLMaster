project(slmaster)

option(BUILD_SLMASTER_SHARED "build shared library." OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/algorithm)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/calibration)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/device)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cameras)

if(BUILD_SLMASTER_SHARED)
    add_library(slmaster SHARED)
    target_compile_definitions(slmaster PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(slmaster PRIVATE -DDLL_EXPORTS)
    set_target_properties(slmaster PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    add_library(slmaster STATIC)
endif()

target_sources(slmaster PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/slmaster.cpp)
target_include_directories(slmaster PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(slmaster PUBLIC cameras algorithm calibration camera projector)