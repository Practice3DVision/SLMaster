##################################
#   Find Projector
##################################
#   This sets the following variables:
# Projector_FOUND             -True if Projector Was found
# Projector_INCLUDE_DIRS      -Directories containing the Projector include files
# Projector_LIBRARY           -Libraries needed to use Projector

find_path(
    Projector_INCLUDE_DIRS
    projectorFactory.h
    ${Projector_DIR}/include
)

find_library(
    Projectord
    Projectord.lib
    ${Projector_DIR}/lib
)

find_library(
    Projector
    Projector.lib
    ${Projector_DIR}/lib
)

find_package(OpenCV REQUIRED)

set(Projector_INCLUDE_DIRS ${Projector_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Projector_LIBRARIES ${Projectord} ${OpenCV_LIBRARIES} setupapi)
else()
    set(Projector_LIBRARIES ${Projector} ${OpenCV_LIBRARIES} setupapi)
endif()

find_path(
    Projector_DlpcApi_INCLUDE_DIRS
    projectorDlpc34xx.h
    ${Projector_DIR}/include
)

if(${Projector_DlpcApi_INCLUDE_DIRS} STREQUAL ${Projector_DIR}/include) 
    message("Projector with dlpc34xx...")

    find_library(
        cyusbserial
        cyusbserial.lib
        ${Projector_DIR}/lib
    )
    find_library(
        projectorDlpcApi
        projectorDlpcApi.lib
        ${Projector_DIR}/lib
    )
    find_library(
        projectorDlpcApid
        projectorDlpcApid.lib
        ${Projector_DIR}/lib
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND Projector_LIBRARIES ${projectorDlpcApid} ${cyusbserial})
    else()
        list(APPEND Projector_LIBRARIES ${projectorDlpcApi} ${cyusbserial})
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Projector
    DEFAULT_MSG
    Projector_INCLUDE_DIRS
    Projector_LIBRARIES
)

mark_as_advanced(
    Projector_LIBRARIES
    Projector_INCLUDE_DIRS
)
