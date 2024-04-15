cmake_minimum_required(VERSION 3.20)

project(projectorDlpcApi)

find_package(OpenCV REQUIRED)

find_path(CyUsbSerial_DIR CyUSBSerial.h ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cyusbserial)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

if(BUILD_DEVICE_SHARED)
    add_library(projectorDlpcApi SHARED)
    target_compile_definitions(projectorDlpcApi PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(projectorDlpcApi PRIVATE -DDLL_EXPORTS)
else()
    add_library(projectorDlpcApi)
endif()

target_sources(projectorDlpcApi PRIVATE ${HEADERS} ${SOURCES})

target_link_directories(projectorDlpcApi PUBLIC ${CyUsbSerial_DIR})
        
target_include_directories(projectorDlpcApi
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CyUsbSerial_DIR}
)

target_link_libraries(
    projectorDlpcApi
    PUBLIC
    cyusbserial
    setupapi
    ${OpenCV_LIBRARIES}
)
