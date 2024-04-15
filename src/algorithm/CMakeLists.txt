cmake_minimum_required(VERSION 3.20)

project(algorithm)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV COMPONENTS core cudev)

FILE(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpuStructuredLight/*.h ${CMAKE_CURRENT_SOURCE_DIR}/cpuStructuredLight/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/cpuStructuredLight/*.cpp)

if(OpenCV_cudev_FOUND)
    message(NOTICE "[SLMaster]: OpenCV With CUDA module.")
    enable_language(CUDA)
    FILE(GLOB_RECURSE GPU_ALGORITHMS ${CMAKE_CURRENT_SOURCE_DIR}/gpuStructuredLight/*.h ${CMAKE_CURRENT_SOURCE_DIR}/gpuStructuredLight/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/gpuStructuredLight/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/gpuStructuredLight/*.cu)
    list(APPEND SOURCES ${GPU_ALGORITHMS})
    source_group("SOURCES" FILES ${SOURCES})
    add_library(algorithm ${SOURCES})
    target_include_directories(algorithm PUBLIC {CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(algorithm PUBLIC ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${CUDA_LIBRARIES})
    target_compile_definitions(algorithm PUBLIC -DOPENCV_WITH_CUDA_MODULE)
    set_target_properties(algorithm PROPERTIES CUDA_STANDARD 11)
else()
    message(WARNING "[SLMaster]: OpenCV dosn't With CUDA module! Just complier CPU Version.")
    
    source_group("SOURCES" FILES ${SOURCES})

    if(BUILD_SLMASTER_SHARED)
    add_library(algorithm SHARED ${SOURCES})
    target_compile_definitions(algorithm PUBLIC -DBUILD_SHARED_LIBS)
    target_compile_definitions(algorithm PRIVATE -DDLL_EXPORTS)
    set_target_properties(algorithm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    else()
        add_library(algorithm STATIC ${SOURCES})
    endif()

    target_include_directories(algorithm PUBLIC {CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
    target_link_libraries(algorithm PUBLIC ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
endif()
