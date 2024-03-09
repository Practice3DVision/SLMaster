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

    MouseArea {
        id: mouse_area
        anchors.fill: parent
        propagateComposedEvents: true
        acceptedButtons: Qt.AllButtons
        Keys.enabled: true
        focus: true

        property bool has_pressed: false
        property var select_area_pos: [0, 0, 0, 0]

        VTKRenderWindow {
            id: vtk_render_window
            anchors.fill: parent
            z: 99999999999

            VTKRenderItem {
                id: vtk_render_item
                anchors.fill: vtk_render_window
                renderWindow: vtk_render_window
            }

            MouseArea {
                anchors.fill: parent
                propagateComposedEvents: true
                acceptedButtons: Qt.AllButtons
                Keys.enabled: true
                hoverEnabled: true
                focus: true

                Keys.onPressed: {
                    if(event.key === Qt.Key_Escape) {
                        VTKProcessEngine.cancelClip();
                        canvas.width = -1;
                        canvas.height = -1;
                        canvas.requestPaint();
                    }

                    event.accepted = false;
                }

                onPressed: (mouse)=> {
                    if(mouse.button === Qt.LeftButton) {
                        if(operation_area.isAreaSelect) {
                            mouse_area.select_area_pos[0] = mouse.x;
                            mouse_area.select_area_pos[1] = mouse.y;
                            mouse_area.has_pressed = true;
                        }
                    }

                    if(mouse.button === Qt.RightButton) {
                        mouse_area.has_pressed = false;
                        finishSelectRec();
                    }

                    mouse.accepted = false;
                }

                onPositionChanged: (mouse)=> {
                    if(operation_area.isAreaSelect && mouse_area.has_pressed) {
                        mouse_area.select_area_pos[2] = mouse.x;
                        mouse_area.select_area_pos[3] = mouse.y;
                        drawSelectRec();
                    }

                    mouse.accepted = false;
                }
            }

            Component.onCompleted: {
                GlobalSignals.render_items[1] = vtk_render_item;

                VTKProcessEngine.bindPostProcessRenderItem(vtk_render_item);
                VTKProcessEngine.setCurRenderItem(vtk_render_item);
                VTKProcessEngine.setBackgroundColor(FluTheme.windowBackgroundColor);
            }
        }
    }

    Canvas{
        id:canvas
        onPaint: {
            var ctx = getContext("2d");
            //ctx.fillStyle = FluTheme.primaryColor
            ctx.strokeStyle = FluTheme.dark ? Qt.rgba(230, 240, 230, 1) : Qt.rgba(20, 20, 20, 1);
            ctx.lineWidth = 3
            ctx.lineJoin = "round"
            //ctx.fillRect(0, 0, width, height)
            //ctx.clearRect(ctx.lineWidth,ctx.lineWidth,width - ctx.lineWidth,height - ctx.lineWidth)
            ctx.strokeRect(0, 0, width, height)
        }
    }

    function drawSelectRec() {
        canvas.x = mouse_area.select_area_pos[0];
        canvas.y = mouse_area.select_area_pos[1];
        canvas.width = Math.abs(mouse_area.select_area_pos[2] - mouse_area.select_area_pos[0]);
        canvas.height = Math.abs(mouse_area.select_area_pos[3] - mouse_area.select_area_pos[1]);
        canvas.requestPaint();

        showInfo(Lang.please_select, 3000);
    }

    function finishSelectRec() {
        if(operation_area.isAreaSelect) {
            canvas.width = -1;
            canvas.height = -1;
            canvas.requestPaint();

            select_operation_menu.x = mouse_area.select_area_pos[2];
            select_operation_menu.y = mouse_area.select_area_pos[3];
            select_operation_menu.open();

            showInfo(Lang.select_finished, 3000);
        }
    }

    FluMenu {
        id: select_operation_menu
        FluMenuItem {
            text: Lang.reserved
            iconSource: FluentIcons.CheckboxComposite

            onClicked: {
                VTKProcessEngine.clip(true);
                select_operation_menu.close();
            }
        }

        FluMenuItem {
            text: Lang.remove
            iconSource: FluentIcons.Broom

            onClicked: {
                VTKProcessEngine.clip(false);
                select_operation_menu.close();
            }
        }

        FluMenuItem {
            text: Lang.cancel
            iconSource: FluentIcons.Clear

            onClicked: {
                VTKProcessEngine.cancelClip();
                select_operation_menu.close();
            }
        }
    }

    FluArea {
        id: operation_area
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.rightMargin: 20
        anchors.topMargin: 20
        width: 48
        height: operation_layout.height

        property bool isColorized: false
        property bool isAreaSelect: false
        property bool isEnablePointInfo: false

        ColumnLayout {
            //anchors.fill: parent
            id: operation_layout
            width: parent.width
            anchors.left: parent.left
            anchors.top: parent.top

            FluIconButton {
                id: colorized_button
                Layout.alignment: Qt.AlignHCenter
                iconSource: operation_area.isColorized ? FluentIcons.ColorOff : FluentIcons.Color
                iconSize: 28

                onClicked: {
                    operation_area.isColorized = !operation_area.isColorized;
                    VTKProcessEngine.enableColorBar(operation_area.isColorized);
                    operation_area.isColorized ? VTKProcessEngine.jetDepthColorMap() : VTKProcessEngine.cancelColorizeCloud();
                    colorized_button.color = operation_area.isColorized ? FluTheme.primaryColor : colorized_button.normalColor;
                }
            }

            FluIconButton {
                id: area_select_button
                Layout.alignment: Qt.AlignHCenter
                iconSource: operation_area.isAreaSelect ? FluentIcons.Cancel : FluentIcons.ClearSelection
                iconSize: 28

                onClicked: {
                    operation_area.isAreaSelect = !operation_area.isAreaSelect;
                    VTKProcessEngine.enableAreaSelected(operation_area.isAreaSelect);
                    operation_area.isAreaSelect ? showInfo(Lang.please_select_start_point, 3000) : showInfo(Lang.cancel, 3000);
                    area_select_button.color = operation_area.isAreaSelect ? FluTheme.primaryColor : area_select_button.normalColor;
                    //operation_area.isColorized ? VTKProcessEngine.jetDepthColorMap() : VTKProcessEngine.cancelColorizeCloud();
                }
            }

            FluIconButton {
                id: is_enable_point_info_button
                Layout.alignment: Qt.AlignHCenter
                iconSource: FluentIcons.Info
                iconSize: 28

                onClicked: {
                    operation_area.isEnablePointInfo = !operation_area.isEnablePointInfo;
                    VTKProcessEngine.enablePointInfo(operation_area.isEnablePointInfo);
                    operation_area.isEnablePointInfo ? showInfo(Lang.please_select_point_to_see_info, 3000) : showInfo(Lang.cancel, 3000);
                    is_enable_point_info_button.color = operation_area.isEnablePointInfo ? FluTheme.primaryColor : is_enable_point_info_button.normalColor;
                    //operation_area.isColorized ? VTKProcessEngine.jetDepthColorMap() : VTKProcessEngine.cancelColorizeCloud();
                }
            }

            FluIconButton {
                Layout.alignment: Qt.AlignHCenter
                iconSource: FluentIcons.Save
                iconSize: 28

                onClicked: {
                    saveFolderDialog.open();
                }
            }
        }
    }

    FileDialog {
        id: saveFolderDialog
        fileMode: FileDialog.SaveFile
        nameFilters: ["Ply files (*.ply)", "Pcd files (*.pcd)"]

        onAccepted: {
            VTKProcessEngine.saveCloud(currentFile.toString());
        }
    }

    ColumnLayout {
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.rightMargin: parent.width * 0.1
        anchors.bottomMargin: parent.height * 0.2

        FluText {
            Layout.fillWidth: true
            text: ("cloud size: %1").arg(VTKProcessEngine.postProcessPointSize)
            opacity: 0.3
        }

        FluText {
            Layout.fillWidth: true
            text: ("fps: %1").arg(vtk_render_item.fps)
            opacity: 0.3
        }
    }
}
