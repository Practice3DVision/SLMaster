project(slmaster)

find_package(OpenCV COMPONENTS core cudawraping structured_light cudastructuredlight)
find_package(PCL REQUIRED)
if(OpenCV_cudastructuredlight_FOUND)
    message("OpenCV With cudastructuredlight module.")
    add_definitions(-DWITH_CUDASTRUCTUREDLIGHT_MODULE)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(SLMaster_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

option(BUILD_SLMASTER_SHARED "build shared library." OFF)

if(BUILD_SLMASTER_SHARED)
    add_library(slmaster SHARED)
    target_compile_definitions(slmaster PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(slmaster PRIVATE -DDLL_EXPORTS)
    set_target_properties(slmaster PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    add_library(slmaster STATIC)
endif()

FILE(GLOB_RECURSE SLMASTER_SOURCES ${SLMaster_INCLUDE_DIRS}/*.h ${SLMaster_INCLUDE_DIRS}/*.hpp ${SLMaster_INCLUDE_DIRS}/*.cpp)
source_group("SLMASTER_SOURCES" FILES ${SLMASTER_SOURCES})

target_sources(slmaster PRIVATE ${SLMASTER_SOURCES})
target_link_libraries(slmaster PUBLIC ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} Camera Projector)
target_include_directories(slmaster PUBLIC ${SLMaster_INCLUDE_DIRS})

if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(slmaster PUBLIC -pthread)
endif()
