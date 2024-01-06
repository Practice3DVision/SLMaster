import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15
import QtQuick.Window 2.15
import Qt.labs.platform 1.1

import FluentUI 1.0
import SLMasterGui 1.0

import "qrc:///ui/global"

FluContentPage{
    id: root
    title: ""
    launchMode: FluPageType.SingleInstance

    anchors.topMargin: 40
    anchors.top: parent.top
    anchors.bottom: parent.bottom
    anchors.left: parent.left
    anchors.right: parent.right
    leftPadding: 0
    rightPadding: 0
    bottomPadding: 0

    VTKRenderWindow {
        id: vtk_render_window
        anchors.fill: parent
        z: 99999999999

        VTKRenderItem {
            id: vtk_render_item
            anchors.fill: vtk_render_window
            renderWindow: vtk_render_window
        }

        Component.onCompleted: {
            GlobalSignals.render_items[2] = vtk_render_item;

            VTKProcessEngine.bindMeasurementRenderItem(vtk_render_item);
            VTKProcessEngine.setCurRenderItem(vtk_render_item);
            VTKProcessEngine.setBackgroundColor(FluTheme.windowBackgroundColor);
        }
    }

    Connections {
        target: FluTheme
        function onDarkModeChanged() {
            VTKProcessEngine.setBackgroundColor(FluTheme.dark ? Qt.rgba(32 / 255, 32 / 255 , 32 / 255, 1) : Qt.rgba(237 / 255, 237 / 255, 237 / 255, 1));
        }
    }
}
