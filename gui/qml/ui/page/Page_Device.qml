import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15
import QtQuick.Window 2.15
import Qt.labs.platform 1.1

import FluentUI 1.0
import SLMasterGui 1.0

import "qrc:///ui/global"

FluContentPage{
    id:root
    title:""
    launchMode: FluPageType.SingleInstance

    anchors.topMargin: 50
    anchors.fill: parent

    property real light_strength: CameraEngine.getNumberAttribute("Light Strength")
    property int pre_exposure_time: CameraEngine.getNumberAttribute("Pre Exposure Time")
    property int exposure_time: CameraEngine.getNumberAttribute("Exposure Time")
    property int aft_exposure_time: CameraEngine.getNumberAttribute("Aft Exposure Time")
    property int camera_type: AppType.BinocularSLCamera
    property int pixel_depth: !CameraEngine.getBooleanAttribute("Is One Bit");
    property int stripe_direction: CameraEngine.getBooleanAttribute("Is Vertical")
    property int stripe_type: CameraEngine.getNumberAttribute("Pattern")
    property int defocus_encoding: pixel_depth ? AppType.Disable : AppType.OptimalPlusWithModulation
    property int img_width: Number(CameraEngine.getStringAttribute("DLP Width"))
    property int img_height: Number(CameraEngine.getStringAttribute("DLP Height"))
    property int clip_width: img_width
    property int clip_height: img_height
    property int cycles: CameraEngine.getNumberAttribute("Cycles")
    property int shiftTime: CameraEngine.getNumberAttribute("Phase Shift Times")
    property int connect_state : AppType.Disconnect
    property bool isResume: false
    property bool enableBurningStripe: false
    property bool isBurnTenLine: false
    property bool isKeepAdd: false

    function updateParams() {
        root.light_strength = CameraEngine.getNumberAttribute("Light Strength")
        root.pre_exposure_time = CameraEngine.getNumberAttribute("Pre Exposure Time")
        root.exposure_time = CameraEngine.getNumberAttribute("Exposure Time")
        root.aft_exposure_time = CameraEngine.getNumberAttribute("Aft Exposure Time")
        root.pixel_depth = !CameraEngine.getBooleanAttribute("Is One Bit");
        root.stripe_direction = CameraEngine.getBooleanAttribute("Is Vertical")
        root.stripe_type = CameraEngine.getNumberAttribute("Pattern")
        root.defocus_encoding = pixel_depth ? AppType.Disable : AppType.OptimalPlusWithModulation
        root.img_width = Number(CameraEngine.getStringAttribute("DLP Width"))
        root.img_height = Number(CameraEngine.getStringAttribute("DLP Height"))
        root.cycles = CameraEngine.getNumberAttribute("Cycles")
        root.shiftTime = CameraEngine.getNumberAttribute("Phase Shift Times")
    }

    Connections {
        target: GlobalSignals
        function onCameraParamsUpdate() {
            root.updateParams();
        }
    }

    RowLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        ColumnLayout {
            Layout.preferredWidth: 360
            Layout.preferredHeight: parent.height
            spacing: 10

            FluLoader {
                id: loader
                Layout.fillHeight: true
                Layout.fillWidth: true

                focus: true

                Component.onCompleted: {
                    sourceComponent = com_device;
                }
            }

            FluText {
                text: Lang.device_function_select
                font: FluTextStyle.Subtitle
                horizontalAlignment: Text.AlignLeft
            }

            FluArea {
                Layout.fillWidth: true
                Layout.preferredHeight: 80

                GridLayout {
                    FluIconButton {
                        text: Lang.device
                        iconSource: FluentIcons.Camera

                        onClicked: {
                            loader.sourceComponent = com_device;
                        }
                    }

                    FluIconButton {
                        text: Lang.stripe_encoding
                        iconSource: FluentIcons.ImageExport

                        onClicked: {
                            loader.sourceComponent = com_stripe;
                        }
                    }
                }
            }

            Component {
                id: com_device
                ColumnLayout {
                    anchors.fill: parent

                    FluText {
                        text: Lang.camera
                        font: FluTextStyle.Subtitle
                        horizontalAlignment: Text.AlignHCenter
                    }

                    FluArea {
                        opacity: 0.8
                        Layout.preferredWidth: parent.width
                        Layout.fillHeight: true

                        FluScrollablePage {
                            anchors.fill: parent
                            anchors.margins: 10

                            ColumnLayout {
                                anchors.fill: parent

                                RowLayout {
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 80

                                    FluImage {
                                        Layout.preferredHeight: 60
                                        Layout.preferredWidth: 200
                                        Layout.alignment: Qt.AlignHCenter
                                        source: "qrc:/res/image/icons8-camera-94.png"
                                        fillMode: Image.PreserveAspectFit
                                        horizontalAlignment: Image.AlignHCenter
                                    }

                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true

                                        FluLoader {
                                            id: loader
                                            Layout.preferredWidth: 36
                                            Layout.preferredHeight: 36
                                            Layout.alignment: Qt.AlignHCenter
                                            sourceComponent: CameraEngine.isOnLine ? online_icon : wating_ring
                                        }

                                        FluText {
                                            Layout.alignment: Qt.AlignHCenter
                                            text: CameraEngine.isOnLine ? Lang.online : Lang.offline
                                            font: FluTextStyle.BodyStrong
                                        }
                                    }
                                }

                                RowLayout{
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true

                                    FluComboBox{
                                        id: camera_type_select
                                        Layout.fillWidth: true
                                        editable: false
                                        model: [Lang.monocular_sl_camera, Lang.binocular_sl_camera, Lang.triple_sl_camera]
                                        currentIndex: root.camera_type
                                        onCurrentIndexChanged: {
                                            if(root.camera_type !== currentIndex) {
                                                root.camera_type = currentIndex;
                                                CameraEngine.selectCamera(root.camera_type);
                                                GlobalSignals.cameraParamsUpdate();
                                            }
                                        }

                                        Component.onCompleted: {
                                            //CameraEngine.selectCamera(root.camera_type);
                                            //root.updateParams();
                                            GlobalSignals.cameraParamsUpdate();
                                        }
                                    }

                                    FluButton {
                                        id: connect_btn
                                        Layout.fillWidth: true
                                        Layout.preferredHeight: camera_type_select.height
                                        text: CameraEngine.isConnected ? Lang.disconnect : Lang.connect

                                        onClicked: {
                                            if(CameraEngine.isConnected) {
                                                CameraEngine.disConnectCamera() ? showSuccess(Lang.disconnect_sucess, 3000) : showError(Lang.disconnect_failed, 3000);
                                            }
                                            else {
                                                CameraEngine.connectCamera() ? showSuccess(Lang.connect_sucess, 3000) : showError(Lang.connect_failed, 3000);
                                            }
                                        }
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
                                        value: root.light_strength * 100
                                        enabled: CameraEngine.isConnected

                                        onValueChanged: {
                                            if(Math.abs(root.light_strength - value / 100) > 0.01) {
                                                root.light_strength = value / 100;
                                                light_strength_spinbox.value = root.light_strength * 100;
                                                CameraEngine.setNumberAttribute("Light Strength", root.light_strength);
                                            }
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
                                        value: root.light_strength * 100
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
                                            if(Math.abs(root.light_strength - value / 100) > 0.01) {
                                                root.light_strength = value / 100;
                                                light_strengh_slider.value = value;
                                                CameraEngine.setNumberAttribute("Light Strength", root.light_strength);
                                            }
                                        }
                                    }
                                }

                                FluText {
                                    text: Lang.pre_exposure_time
                                }

                                RowLayout {
                                    Layout.fillWidth: true

                                    FluSlider {
                                        id: pre_exposure_time_slider
                                        Layout.fillWidth: true
                                        from: 100
                                        to: 100000000
                                        value: root.pre_exposure_time
                                        enabled: CameraEngine.isConnected

                                        onValueChanged: {
                                            root.pre_exposure_time = value;
                                            pre_exposure_time_spinbox.value = root.pre_exposure_time;
                                            CameraEngine.setNumberAttribute("Pre Exposure Time", root.pre_exposure_time);
                                        }
                                    }

                                    FluSpinBox {
                                        id: pre_exposure_time_spinbox
                                        Layout.preferredWidth: parent.width / 3
                                        editable: true
                                        up.indicator: undefined
                                        down.indicator: undefined
                                        from: 100
                                        to: 100000000
                                        value: root.pre_exposure_time
                                        enabled: CameraEngine.isConnected

                                        onValueChanged: {
                                            root.pre_exposure_time = value;
                                            pre_exposure_time_slider.value = root.pre_exposure_time;
                                            CameraEngine.setNumberAttribute("Pre Exposure Time", root.pre_exposure_time);
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
                                        to: 100000000
                                        value: root.exposure_time
                                        enabled: CameraEngine.isConnected

                                        onValueChanged: {
                                            root.exposure_time = value;
                                            exposure_time_spinbox.value = root.exposure_time;
                                            CameraEngine.setNumberAttribute("Exposure Time", root.exposure_time);
                                        }
                                    }

                                    FluSpinBox {
                                        id: exposure_time_spinbox
                                        Layout.preferredWidth: parent.width / 3
                                        editable: true
                                        up.indicator: undefined
                                        down.indicator: undefined
                                        from: 100
                                        to: 100000000
                                        value: root.exposure_time
                                        enabled: CameraEngine.isConnected

                                        onValueChanged: {
                                            root.exposure_time = value;
                                            exposure_time_slider.value = root.exposure_time;
                                            CameraEngine.setNumberAttribute("Exposure Time", root.exposure_time);
                                        }
                                    }
                                }

                                FluText {
                                    text: Lang.aft_exposure_time
                                }

                                RowLayout {
                                    Layout.fillWidth: true

                                    FluSlider {
                                        id: aft_exposure_time_slider
                                        Layout.fillWidth: true
                                        from: 100
                                        to: 100000000
                                        value: root.aft_exposure_time
                                        enabled: CameraEngine.isConnected

                                        onValueChanged: {
                                            root.aft_exposure_time = value;
                                            aft_exposure_time_spinbox.value = root.aft_exposure_time;
                                            CameraEngine.setNumberAttribute("Aft Exposure Time", root.aft_exposure_time);
                                        }
                                    }

                                    FluSpinBox {
                                        id: aft_exposure_time_spinbox
                                        Layout.preferredWidth: parent.width / 3
                                        editable: true
                                        up.indicator: undefined
                                        down.indicator: undefined
                                        from: 100
                                        to: 100000000
                                        value: root.aft_exposure_time
                                        enabled: CameraEngine.isConnected

                                        onValueChanged: {
                                            root.aft_exposure_time = value;
                                            aft_exposure_time_slider.value = root.aft_exposure_time;
                                            CameraEngine.setNumberAttribute("Aft Exposure Time", root.aft_exposure_time);
                                        }
                                    }
                                }

                                GridLayout {
                                    Layout.topMargin: 8
                                    Layout.fillWidth: true
                                    rows: 3
                                    columns: 3
                                    rowSpacing: 8

                                    FluButton {
                                        Layout.fillWidth: true
                                        text: Lang.project_once
                                        enabled: CameraEngine.isConnected

                                        onClicked: {
                                            CameraEngine.projectOnce();
                                            root.enableBurningStripe = false;
                                        }
                                    }

                                    FluButton {
                                        Layout.fillWidth: true
                                        text: Lang.project_continue
                                        enabled: CameraEngine.isConnected

                                        onClicked: {
                                            CameraEngine.projectContinues();
                                        }
                                    }

                                    FluButton {
                                        Layout.fillWidth: true
                                        text: Lang.pause_project
                                        enabled: CameraEngine.isConnected

                                        onClicked: {
                                            CameraEngine.pauseProject(root.isResume);
                                            root.isResume = !root.isResume;
                                        }
                                    }

                                    FluButton {
                                        Layout.fillWidth: true
                                        text: Lang.project_step
                                        enabled: CameraEngine.isConnected

                                        onClicked: {
                                            CameraEngine.stepProject();
                                        }
                                    }

                                    FluButton {
                                        Layout.fillWidth: true
                                        text: Lang.stop_project
                                        enabled: CameraEngine.isConnected

                                        onClicked: {
                                            CameraEngine.stopProject();
                                        }
                                    }

                                    FluButton {
                                        Layout.fillWidth: true
                                        text: Lang.project_ten_line
                                        enabled: CameraEngine.isConnected

                                        onClicked: {
                                            CameraEngine.tenLine();
                                            root.isBurnTenLine = true;
                                            showLoading(Lang.burn_ing, false);
                                        }
                                    }
                                }

                                FluButton {
                                    Layout.topMargin: 5
                                    Layout.fillWidth: true
                                    text: Lang.burn
                                    enabled: CameraEngine.isConnected && root.enableBurningStripe

                                    onClicked: {
                                        CameraEngine.setBooleanAttribute("Is One Bit", root.pixel_depth === AppType.OneBit);
                                        CameraEngine.setBooleanAttribute("Is Vertical", root.stripe_direction === AppType.Vertical);
                                        CameraEngine.setNumberAttribute("Pattern", root.stripe_type);
                                        CameraEngine.setNumberAttribute("Cycles", root.cycles);
                                        CameraEngine.setNumberAttribute("Phase Shift Times", root.shiftTime);

                                        CameraEngine.burnStripe();
                                        showLoading(Lang.burn_ing, false);
                                    }
                                }

                                Connections {
                                    target: CameraEngine
                                    function onIsBurnWorkFinishChanged() {
                                        hideLoading();

                                        if(root.isBurnTenLine) {
                                            CameraEngine.projectContinues();
                                            root.isBurnTenLine = false;
                                        }
                                    }
                                }
                            }

                            Component {
                                id: wating_ring
                                FluProgressRing {

                                }
                            }

                            Component {
                                id: online_icon
                                FluImage {
                                    source: "qrc:/res/image/icons8-online-48.png"
                                }
                            }
                        }
                    }
                }
            }

            Component {
                id: com_stripe
                ColumnLayout {
                    anchors.fill: parent

                    FluText {
                        text: Lang.stripe_encoding
                        font: FluTextStyle.Subtitle
                        horizontalAlignment: Text.AlignHCenter
                    }

                    FluArea {
                        opacity: 0.8
                        Layout.preferredWidth: parent.width
                        Layout.fillHeight: true

                        FluScrollablePage {
                            anchors.fill: parent
                            anchors.margins: 10
                            ColumnLayout {
                                Layout.fillHeight: true
                                Layout.fillWidth: true

                                FluText {
                                    text: Lang.pixel_depth
                                    font: FluTextStyle.BodyStrong
                                }

                                RowLayout {
                                    Layout.fillWidth: true

                                    FluRadioButton {
                                        id: one_bit_btn
                                        text: Lang.one_bit
                                        checked: root.pixel_depth === AppType.OneBit

                                        onClicked: {
                                            root.pixel_depth = AppType.OneBit;
                                            one_bit_btn.checked = true;
                                            eight_bit_btn.checked = false;

                                            defocus_method_cbx.currentIndex = AppType.Binary;
                                        }
                                    }

                                    FluRadioButton {
                                        id: eight_bit_btn
                                        text: Lang.eight_bit
                                        checked: root.pixel_depth === AppType.EightBit

                                        onClicked: {
                                            root.pixel_depth = AppType.EightBit;
                                            eight_bit_btn.checked = true;
                                            one_bit_btn.checked = false;

                                            defocus_method_cbx.currentIndex = AppType.Disable;
                                        }
                                    }
                                }

                                FluText {
                                    text: Lang.direction
                                    font: FluTextStyle.BodyStrong
                                }

                                RowLayout {
                                    Layout.fillWidth: true

                                    FluRadioButton {
                                        id: honrizon_btn
                                        text: Lang.honrizon
                                        checked: root.stripe_direction === AppType.Horizion

                                        onClicked: {
                                            root.stripe_direction = AppType.Horizion;
                                            honrizon_btn.checked = true;
                                            vertical_btn.checked = false;
                                        }
                                    }

                                    FluRadioButton {
                                        id: vertical_btn
                                        text: Lang.vertical
                                        checked: root.stripe_direction === AppType.Vertical

                                        onClicked: {
                                            root.stripe_direction = AppType.Vertical;
                                            vertical_btn.checked = true;
                                            honrizon_btn.checked = false;
                                        }
                                    }
                                }

                                FluText {
                                    text: Lang.stripe_type
                                    font: FluTextStyle.BodyStrong
                                }

                                FluComboBox {
                                    Layout.fillWidth: true
                                    model: [Lang.sine_complementary_gray_code, Lang.multi_frequency_heterodyne, Lang.multi_view_stereo_geometry, Lang.sine_shift_gray_code]
                                    currentIndex: root.stripe_type

                                    onCurrentIndexChanged: {
                                        root.stripe_type = currentIndex;
                                    }
                                }

                                FluText {
                                    text: Lang.defocus_encoding
                                    font: FluTextStyle.BodyStrong
                                }

                                FluComboBox {
                                    id: defocus_method_cbx
                                    Layout.fillWidth: true
                                    model: [Lang.disable, Lang.binary, Lang.error_diffusion_method, Lang.opwm]
                                    currentIndex: root.defocus_encoding
                                    enabled: root.pixel_depth == AppType.OneBit

                                    onCurrentIndexChanged: {
                                        root.defocus_encoding = currentIndex;

                                        if(root.defocus_encoding == AppType.Disable && enabled) {
                                            showWarning(Lang.one_bit_disable_defocus_warning, 3000);
                                            currentIndex = AppType.Binary;
                                        }
                                    }
                                }

                                FluLoader {
                                    id: stripe_type_component_loader
                                    Component.onCompleted:  {
                                        stripe_type_component_loader.sourceComponent = sine_complementary_gray_code_spinbox;
                                    }

                                    Component {
                                        id: sine_complementary_gray_code_spinbox
                                        GridLayout {
                                            Layout.fillWidth: true
                                            rows: 2
                                            columns: 2
                                            columnSpacing: 46
                                            FluText {
                                                Layout.fillWidth: true
                                                text: Lang.img_width
                                                font: FluTextStyle.BodyStrong
                                            }

                                            FluText {
                                                Layout.fillWidth: true
                                                text: Lang.img_height
                                                font: FluTextStyle.BodyStrong
                                            }

                                            FluSpinBox {
                                                id: img_width_spbox
                                                editable: true
                                                value: root.img_width
                                                from: 0
                                                to: 9999999

                                                onValueChanged: {
                                                    root.img_width = value;
                                                }
                                            }

                                            FluSpinBox {
                                                id: img_height_spbox
                                                editable: true
                                                value: root.img_height
                                                from: 0
                                                to: 9999999

                                                onValueChanged: {
                                                    root.img_height = value;
                                                }
                                            }

                                            FluText {
                                                Layout.fillWidth: true
                                                text: Lang.clip_width
                                                font: FluTextStyle.BodyStrong
                                            }

                                            FluText {
                                                Layout.fillWidth: true
                                                text: Lang.clip_height
                                                font: FluTextStyle.BodyStrong
                                            }

                                            FluSpinBox {
                                                id: clip_width_spbox
                                                editable: true
                                                value: root.clip_width
                                                from: 0
                                                to: 9999999

                                                onValueChanged: {
                                                    root.clip_width = value;
                                                }
                                            }

                                            FluSpinBox {
                                                id: clip_height_spbox
                                                editable: true
                                                value: root.clip_height
                                                from: 0
                                                to: 9999999

                                                onValueChanged: {
                                                    root.clip_height = value;
                                                }
                                            }

                                            FluText {
                                                Layout.fillWidth: true
                                                text: Lang.cycles
                                                font: FluTextStyle.BodyStrong
                                            }

                                            FluText {
                                                Layout.fillWidth: true
                                                text: Lang.shift_time
                                                font: FluTextStyle.BodyStrong
                                            }

                                            FluSpinBox {
                                                id: cycles_spbox
                                                editable: true
                                                value: root.cycles
                                                from: 0
                                                to: 9999999

                                                onValueChanged: {
                                                    root.cycles = value;
                                                }
                                            }

                                            FluSpinBox {
                                                id: shift_time_spbox
                                                editable: true
                                                value: root.shiftTime
                                                from: 0
                                                to: 9999999

                                                onValueChanged: {
                                                    root.shiftTime = value;
                                                }
                                            }
                                        }
                                    }
                                }

                                RowLayout {
                                    Layout.fillWidth: true

                                    FluRadioButton {
                                        id: clear_pre_stripes
                                        text: Lang.clear_pre_stripes
                                        checked: !root.isKeepAdd

                                        onClicked: {
                                            root.isKeepAdd = false;
                                            clear_pre_stripes.checked = true;
                                            keep_add_stripes.checked = false;
                                        }
                                    }

                                    FluRadioButton {
                                        id: keep_add_stripes
                                        text: Lang.keep_add_stripes
                                        checked: root.isKeepAdd

                                        onClicked: {
                                            root.isKeepAdd = true;
                                            clear_pre_stripes.checked = false;
                                            keep_add_stripes.checked = true;
                                        }
                                    }
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    Layout.topMargin: 10
                                    FluButton {
                                        Layout.preferredWidth: parent.width / 2
                                        text: Lang.encode

                                        onClicked: {
                                            var num_of_stripes = CameraEngine.createStripe(root.pixel_depth, root.stripe_direction, root.stripe_type, root.defocus_encoding, root.img_width, root.img_height, root.clip_width, root.clip_height, root.cycles, root.shiftTime, root.isKeepAdd);
                                            stripe_index_indicator.pageCurrent = 1;
                                            CameraEngine.displayStripe(1);
                                            root.enableBurningStripe = true;
                                        }
                                    }

                                    FluButton {
                                        Layout.fillWidth: true
                                        text: Lang.save

                                        onClicked: {
                                            folder_dialog.open();
                                        }

                                        FolderDialog {
                                            id: folder_dialog
                                            title: Lang.please_choose_save_folder

                                            onAccepted: {
                                                CameraEngine.saveStripe(currentFolder.toString());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10
            FluText {
                text: Lang.stripe_img
                font: FluTextStyle.Subtitle
            }

            FluArea {
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true

                FluShadow { }

                ImagePaintItem {
                    id: stripePaintItem
                    anchors.fill: parent
                    color: FluTheme.dark ? Window.active ?  Qt.rgba(38/255,44/255,54/255,1) : Qt.rgba(39/255,39/255,39/255,1) : Qt.rgba(251/255,251/255,253/255,1)

                    Component.onCompleted: {
                        CameraEngine.bindStripePaintItem(stripePaintItem);
                    }
                }
            }

            FluPagination{
                id: stripe_index_indicator
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignLeft
                previousText: Lang.previous_text
                nextText: Lang.next_text
                pageCurrent: 1
                itemCount: 10
                pageButtonCount: 10

                onPageCurrentChanged: {
                    CameraEngine.displayStripe(pageCurrent);
                }
            }

            Connections {
                target: CameraEngine

                function onStripeImgsChanged(num) {
                    stripe_index_indicator.itemCount = num * stripe_index_indicator.pageButtonCount;
                }
            }
        }
    }
}
