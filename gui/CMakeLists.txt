project(SLMasterGui)

find_package(PCL REQUIRED)
find_package(VTK REQUIRED)
find_package(QT NAMES Qt5 Qt6 REQUIRED COMPONENTS Core Gui Quick Qml QuickControls2)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Quick Qml QuickControls2)
find_package(OpenCV COMPONENTS core structured_light cudastructuredlight)
if(OpenCV_cudastructuredlight_FOUND)
    message("OpenCV With cudastructuredlight module.")
    add_definitions(-DWITH_CUDASTRUCTUREDLIGHT_MODULE)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/qml/*.h ${CMAKE_CURRENT_SOURCE_DIR}/qml/*.cpp)
file(GLOB_RECURSE QML ${CMAKE_CURRENT_SOURCE_DIR}/qml/*.qml)
file(GLOB_RECURSE QRC ${CMAKE_CURRENT_SOURCE_DIR}/qml/*.qrc ${CMAKE_CURRENT_SOURCE_DIR}/qml/qmldir* ${CMAKE_CURRENT_SOURCE_DIR}/qml/*.rc)

set(SLMasterGuiDEPLIBS slmaster Qt${QT_VERSION_MAJOR}::Quick Qt${QT_VERSION_MAJOR}::Qml Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::QuickControls2 QuickQanava)
if(CUDA_FOUND)
    message("cuda found...")
    list(APPEND SLMasterGuiDEPLIBS ${CUDA_LIBRARIES})
endif()

if (FLUENTUI_BUILD_STATIC_LIB AND (QT_VERSION VERSION_GREATER_EQUAL "6.2"))
    list(APPEND SLMasterGuiDEPLIBS fluentui)
else()
    list(APPEND SLMasterGuiDEPLIBS fluentuiplugin)
endif()

add_executable(SLMasterGui)
target_sources(SLMasterGui PRIVATE ${SOURCES} ${QML} ${QRC})
target_include_directories(SLMasterGui PRIVATE QuickQanava ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/calibrationMaster ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes Qt${QT_VERSION_MAJOR}::QuickControls2)
target_link_libraries(SLMasterGui PRIVATE ${SLMasterGuiDEPLIBS} ${VTK_LIBRARIES} ${PCL_LIBRARIES})

vtk_module_autoinit(
  TARGETS SLMasterGui
  MODULES ${VTK_LIBRARIES}
)
