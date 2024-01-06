import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15
import QtQuick.Window 2.15
import Qt.labs.platform 1.1

import FluentUI 1.0
import SLMasterGui 1.0

import "qrc:///ui/global"

FluContentPage {
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

    property int cur_method: CameraEngine.getNumberAttribute("Pattern")
    property real cur_light_strength: CameraEngine.getNumberAttribute("Light Strength")
    property int cur_exposure_time: CameraEngine.getNumberAttribute("Exposure Time")
    property int filter_threshod: CameraEngine.getNumberAttribute("Contrast Threshold")
    property bool enable_gpu: CameraEngine.getBooleanAttribute("Gpu Accelerate")

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
                        displayBody.cancleSelect();
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
                GlobalSignals.render_items[0] = vtk_render_item;

                VTKProcessEngine.bindScanRenderItem(vtk_render_item);
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

    Connections {
        target: FluTheme
        function onDarkModeChanged() {
            VTKProcessEngine.setBackgroundColor(FluTheme.dark ? Qt.rgba(32 / 255, 32 / 255 , 32 / 255, 1) : Qt.rgba(237 / 255, 237 / 255, 237 / 255, 1));
        }
    }

    ColumnLayout {
        id: layout
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 20
        anchors.topMargin: 20
        width: texture_paint_item.width + 10

        FluExpander {
            id: texture_expander
            headerText: Lang.texture
            Layout.fillWidth: true
            contentHeight: texture_paint_item.height + 10

            Item {
                Flickable{
                    anchors.fill: parent
                    contentWidth: texture_paint_item.width
                    contentHeight: texture_paint_item.height
                    ImagePaintItem {
                        id: texture_paint_item
                        width: root.width / 6
                        height: width * 2 / 3
                        anchors.left: parent.left
                        anchors.top: parent.top
                        anchors.leftMargin: 5
                        anchors.topMargin: 5
                        color: FluTheme.dark ? Window.active ?  Qt.rgba(38/255,44/255,54/255,1) : Qt.rgba(39/255,39/255,39/255,1) : Qt.rgba(251/255,251/255,253/255,1)

                        Component.onCompleted: {
                            CameraEngine.bindScanTexturePaintItem(texture_paint_item);
                        }
                    }
                }
            }
        }

        FluExpander {
            id: setting_expander
            headerText: Lang.settings
            Layout.fillWidth: true
            contentHeight: settings_area.height + 10

            Item {
                Flickable{
                    anchors.fill: parent
                    contentWidth: settings_area.width
                    contentHeight: settings_area.height
                    FluArea {
                        id: settings_area
                        width: root.width / 6
                        height: 400
                        anchors.left: parent.left
                        anchors.top: parent.top
                        anchors.leftMargin: 5
                        anchors.topMargin: 5

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 4

                            FluText {
                                text: Lang.pattern_type
                            }

                            FluComboBox {
                                Layout.fillWidth: true
                                model: [Lang.sinus_comple_gray_code, Lang.multi_frequency_heterodyne, Lang.multi_view_stereo_geometry]
                                currentIndex: root.cur_method

                                onCurrentIndexChanged: {
                                    root.cur_method = currentIndex;
                                    CameraEngine.setNumberAttribute("Pattern", root.cur_method);
                                }

                                Component.onCompleted: {
                                    CameraEngine.setPatternType(root.cur_method);
                                }
                            }

                            FluToggleSwitch {
                                text: Lang.enable_gpu
                                checked: root.enable_gpu

                                onCheckedChanged: {
                                    root.enable_gpu = checked;
                                    CameraEngine.setBooleanAttribute("Gpu Accelerate", root.enable_gpu);
                                }
                            }

                            FluText {
                                text: Lang.light_strength
                            }

                            RowLayout {
                                Layout.fillWidth: true

                                FluSlider {
                                    id: light_strengh_slider
                                    Layout.fillWidth: true
                                    from: 10
                                    to: 100
                                    value: root.cur_light_strength * 100
                                    enabled: CameraEngine.isConnected

                                    onValueChanged: {
                                        root.cur_light_strength = value / 100;
                                        light_strength_spinbox.value = root.cur_light_strength * 100;
                                        CameraEngine.setNumberAttribute("Light Strength", root.cur_light_strength);
                                    }
                                }

                                FluSpinBox {
                                    id: light_strength_spinbox
                                    Layout.preferredWidth: parent.width / 3
                                    editable: true
                                    up.indicator: undefined
                                    down.indicator: undefined
                                    from: 10
                                    to: 100
                                    stepSize: 100
                                    value: root.cur_light_strength * 100
                                    enabled: CameraEngine.isConnected

                                    property int decimals: 2
                                    property real realValue: value / 100
                                    textFromValue: function(value, locale) {
                                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                                    }

                                    valueFromText: function(text, locale) {
                                        return Number.fromLocaleString(locale, text) * 100
                                    }

                                    onValueChanged: {
                                        root.cur_light_strength = value / 100;
                                        light_strengh_slider.value = value;
                                        CameraEngine.setNumberAttribute("Light Strength", root.cur_light_strength);
                                    }
                                }
                            }

                            FluText {
                                text: Lang.exposure_time
                            }

                            RowLayout {
                                Layout.fillWidth: true

                                FluSlider {
                                    id: exposure_time_slider
                                    Layout.fillWidth: true
                                    from: 100
                                    to: 100000
                                    value: root.cur_exposure_time
                                    enabled: CameraEngine.isConnected

                                    onValueChanged: {
                                        root.cur_exposure_time = value;
                                        exposure_time_spinbox.value = root.cur_exposure_time;
                                        CameraEngine.setNumberAttribute("Exposure Time", root.cur_exposure_time);
                                    }
                                }

                                FluSpinBox {
                                    id: exposure_time_spinbox
                                    Layout.preferredWidth: parent.width / 3
                                    editable: true
                                    up.indicator: undefined
                                    down.indicator: undefined
                                    from: 100
                                    to: 10000000
                                    value: root.cur_exposure_time
                                    enabled: CameraEngine.isConnected

                                    onValueChanged: {
                                        root.cur_exposure_time = value;
                                        exposure_time_slider.value = root.cur_exposure_time;
                                        CameraEngine.setNumberAttribute("Exposure Time", root.cur_exposure_time);
                                    }
                                }
                            }

                            FluText {
                                text: Lang.filter_threshod
                            }

                            RowLayout {
                                Layout.fillWidth: true

                                FluSlider {
                                    id: filter_threshod_slider
                                    Layout.fillWidth: true
                                    from: 0
                                    to: 255
                                    value: root.filter_threshod

                                    onValueChanged: {
                                        root.filter_threshod = value;
                                        filter_threshod_spinbox.value = root.filter_threshod;
                                        CameraEngine.setNumberAttribute("Contrast Threshold", root.filter_threshod);
                                    }
                                }

                                FluSpinBox {
                                    id: filter_threshod_spinbox
                                    Layout.preferredWidth: parent.width / 3
                                    editable: true
                                    up.indicator: undefined
                                    down.indicator: undefined
                                    from: 0
                                    to: 255
                                    value: root.filter_threshod

                                    onValueChanged: {
                                        root.filter_threshod = value;
                                        filter_threshod_slider.value = root.filter_threshod;
                                        CameraEngine.setNumberAttribute("Contrast Threshold", root.filter_threshod);
                                    }
                                }
                            }
                        }
                    }
                }
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

        property bool isPlay: false
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
                id: play_once_button
                Layout.alignment: Qt.AlignHCenter
                iconSource: FluentIcons.Play
                iconSize: 28

                onClicked: {
                    CameraEngine.startScan();
                }
            }

            FluIconButton {
                id: play_pause_button
                Layout.alignment: Qt.AlignHCenter
                iconSource: operation_area.isPlay ? FluentIcons.Stop : FluentIcons.Replay
                iconSize: 28

                onClicked: {
                    operation_area.isPlay = !operation_area.isPlay;

                    if(operation_area.isPlay) {
                        CameraEngine.continuesScan();
                    }
                    else {
                        CameraEngine.pauseScan();
                    }

                    play_pause_button.color = operation_area.isPlay ? FluTheme.primaryColor : play_pause_button.normalColor;
                }
            }

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
                iconSource: FluentIcons.Info
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

    Connections {
        target: VTKProcessEngine

        function onPointInfoChanged(x, y, z) {
            showSuccess("Point Loc: " + x.toString() + ", " + y.toString() + ", " + z.toString(), 3000);
        }
    }

    ColumnLayout {
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.rightMargin: parent.width * 0.1
        anchors.bottomMargin: parent.height * 0.2

        FluText {
            Layout.fillWidth: true
            text: ("cloud size: %1").arg(VTKProcessEngine.pointSize)
            opacity: 0.3
        }

        FluText {
            Layout.fillWidth: true
            text: ("fps: %1").arg(vtk_render_item.fps)
            opacity: 0.3
        }
    }
    /*
    //TODO@LiuYunhuang: 当前只通过相机JSON进行更新
    Connections{
        target: JSONListModel

        function onJsonUpdated() {
            root.cur_method = JSONListModel.model.get(GlobalSignals.camera_properties["Pattern"]).data;
            root.cur_light_strength = JSONListModel.model.get(GlobalSignals.camera_properties["Light Strength"]).data;
            root.cur_exposure_time = JSONListModel.model.get(GlobalSignals.camera_properties["Exposure Time"]).data;
            root.filter_threshod = JSONListModel.model.get(GlobalSignals.camera_properties["Contrast Threshold"]).data;
            root.enable_gpu = JSONListModel.model.get(GlobalSignals.camera_properties["Gpu Accelerate"]).data;
        }
    }
    */
}
