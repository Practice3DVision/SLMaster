##################################
#   Find Camera
##################################
#   This sets the following variables:
# Camera_FOUND             -True if Camera Was found
# Camera_INCLUDE_DIRS      -Directories containing the Camera include files
# Camera_LIBRARY           -Libraries needed to use Camera

find_path(
    Camera_INCLUDE_DIRS
    cameraFactory.h
    ${Camera_DIR}/include
)

find_library(
    Camerad
    Camerad.lib
    ${Camera_DIR}/lib
)

find_library(
    Camera
    Camera.lib
    ${Camera_DIR}/lib
)

set(Camera_INCLUDE_DIRS ${Camera_INCLUDE_DIR})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Camera_LIBRARIES ${Camerad})
else()
    set(Camera_LIBRARIES ${Camera})
endif()

find_path(
    Huaray_Camera_INCLUDE_DIRS
    huarayCamera.h
    ${Camera_DIR}/include
)

if(${Huaray_Camera_INCLUDE_DIRS} STREQUAL ${Camera_DIR}/include) 
    message("Camera with huaray...")

    find_library(
        MVSDKmd
        MVSDKmd.lib
        ${Camera_DIR}/lib
    )
    find_library(
        huarayCamera
        huarayCamera.lib
        ${Camera_DIR}/lib
    )
    find_library(
        huarayCamerad
        huarayCamerad.lib
        ${Camera_DIR}/lib
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND Camera_LIBRARIES ${MVSDKmd} ${huarayCamerad})
    else()
        list(APPEND Camera_LIBRARIES ${MVSDKmd} ${huarayCamera})
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Camera
    DEFAULT_MSG
    Camera_INCLUDE_DIRS
    Camera_LIBRARIES
)

mark_as_advanced(
    Camera_LIBRARIES
    Camera_INCLUDE_DIRS
)
