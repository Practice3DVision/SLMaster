project(SLMasterGui)

add_subdirectory(thirdParty/FluentUI)
add_subdirectory(thirdParty/QuickQanava)

find_package(VTK REQUIRED)
find_package(QT NAMES Qt5 Qt6 REQUIRED COMPONENTS Core Gui Quick Qml QuickControls2)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Quick Qml QuickControls2)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/qml/*.h ${CMAKE_CURRENT_SOURCE_DIR}/qml/*.cpp)
file(GLOB_RECURSE QML ${CMAKE_CURRENT_SOURCE_DIR}/qml/*.qml)
file(GLOB_RECURSE QRC ${CMAKE_CURRENT_SOURCE_DIR}/qml/*.qrc ${CMAKE_CURRENT_SOURCE_DIR}/qml/qmldir* ${CMAKE_CURRENT_SOURCE_DIR}/qml/*.rc)

set(SLMasterGuiDEPLIBS slmaster ${VTK_LIBRARIES} Qt${QT_VERSION_MAJOR}::Quick Qt${QT_VERSION_MAJOR}::Qml Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::QuickControls2 QuickQanava)

if (FLUENTUI_BUILD_STATIC_LIB AND (QT_VERSION VERSION_GREATER_EQUAL "6.2"))
    list(APPEND SLMasterGuiDEPLIBS fluentui)
else()
    list(APPEND SLMasterGuiDEPLIBS fluentuiplugin)
endif()

add_executable(SLMasterGui WIN32)
target_sources(SLMasterGui PRIVATE ${SOURCES} ${QML} ${QRC})
target_include_directories(SLMasterGui PRIVATE QuickQanava ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/calibrationMaster ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes Qt${QT_VERSION_MAJOR}::QuickControls2)
target_link_libraries(SLMasterGui PRIVATE ${SLMasterGuiDEPLIBS})

vtk_module_autoinit(
  TARGETS SLMasterGui
  MODULES ${VTK_LIBRARIES}
)