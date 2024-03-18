import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Qt.labs.platform 1.1

import FluentUI 1.0
import SLMasterGui 1.0

import "qrc:///ui/global"

FluWindow {
    id:window
    title: Lang.calibration
    width: 500
    height: 600
    fixSize: true
    launchMode: FluWindowType.SingleTask

    property int target_type: AppType.Blob
    property int row_feature_num: 13
    property int col_feature_num: 10
    property int calibrate_type: AppType.Intrinsic
    property real feature_distance: 15
    property real honrizon_pitch: 32
    property real vertical_pitch: 60
    property real inner_circle_inner_radius: 2.5
    property real inner_circle_externer_radius: 5
    property real externer_circle_inner_radius: 7.5
    property real externer_circle_externer_radius: 10
    property bool export_as_left_color_calibration: false

    FluArea {
        anchors.fill: parent

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 20
            spacing: 10

            GridLayout {
                Layout.fillWidth: true
                rows: 2
                columns: 3

                FluText {
                    Layout.fillWidth: true
                    font: FluTextStyle.BodyStrong
                    text: Lang.target_type
                }

                FluComboBox {
                    Layout.preferredWidth: parent.width / 3
                    model: [Lang.chess_board, Lang.blob, Lang.concentric_circle]
                    currentIndex: window.target_type

                    onCurrentIndexChanged: {
                        window.target_type = currentIndex;
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true

                FluRadioButton {
                    id: calibrate_intrinsic_radio
                    Layout.fillWidth: true
                    text: Lang.calibrate_intrincis
                    checked: window.calibrate_type === AppType.Intrinsic

                    onClicked: {
                        window.calibrate_type = AppType.Intrinsic;
                        calibrate_intrinsic_radio.checked = true;
                        calibrate_extrinsic_radio.checked = false;
                    }
                }

                FluRadioButton {
                    id: calibrate_extrinsic_radio
                    Layout.fillWidth: true
                    text: Lang.calibrate_extrincis
                    checked: window.calibrate_type === AppType.Extrinsic

                    onClicked: {
                        window.calibrate_type = AppType.Extrinsic;
                        calibrate_extrinsic_radio.checked = true;
                        calibrate_intrinsic_radio.checked = false;
                    }
                }
            }

            GridLayout {
                Layout.fillWidth: true
                rows: 2
                columns: 2

                FluText {
                    Layout.fillWidth: true
                    font: FluTextStyle.BodyStrong
                    text: Lang.honrizon_pitch
                }

                FluText {
                    Layout.fillWidth: true
                    font: FluTextStyle.BodyStrong
                    text: Lang.vertical_pitch
                }

                FluSpinBox {
                    Layout.fillWidth: true
                    from: 0
                    to: 999999999
                    value: window.honrizon_pitch * stepSize
                    stepSize: 100
                    editable: true
                    property int decimals: 2
                    property real realValue: value / 100
                    textFromValue: function(value, locale) {
                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                    }

                    valueFromText: function(text, locale) {
                        return Number.fromLocaleString(locale, text) * 100
                    }

                    onValueChanged: {
                        window.honrizon_pitch = value / 100;
                    }
                }

                FluSpinBox {
                    Layout.fillWidth: true
                    from: 0
                    to: 999999999
                    value: window.vertical_pitch * stepSize
                    stepSize: 100
                    editable: true
                    property int decimals: 2
                    property real realValue: value / 100
                    textFromValue: function(value, locale) {
                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                    }

                    valueFromText: function(text, locale) {
                        return Number.fromLocaleString(locale, text) * 100
                    }

                    onValueChanged: {
                        window.vertical_pitch = value / 100;
                    }
                }
            }

            GridLayout {
                Layout.fillWidth: true
                rows: 2
                columns: 3
                rowSpacing: 10
                columnSpacing: 10

                FluText {
                    Layout.fillWidth: true
                    font: FluTextStyle.BodyStrong
                    text: Lang.row_feature_points_num
                }

                FluText {
                    Layout.fillWidth: true
                    font: FluTextStyle.BodyStrong
                    text: Lang.col_feature_points_num
                }

                FluText {
                    Layout.fillWidth: true
                    font: FluTextStyle.BodyStrong
                    text: Lang.features_distance
                }

                FluSpinBox {
                    from: 0
                    to: 9999999
                    editable: true
                    value: window.row_feature_num

                    onValueChanged: {
                        window.row_feature_num = value;
                    }
                }

                FluSpinBox {
                    from: 0
                    to: 9999999
                    editable: true
                    value: window.col_feature_num

                    onValueChanged: {
                        window.col_feature_num = value;
                    }
                }

                FluSpinBox {
                    from: 0
                    to: 999999999
                    value: window.feature_distance * stepSize
                    stepSize: 100
                    editable: true
                    property int decimals: 2
                    property real realValue: value / 100
                    textFromValue: function(value, locale) {
                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                    }

                    valueFromText: function(text, locale) {
                        return Number.fromLocaleString(locale, text) * 100
                    }

                    onValueChanged: {
                        window.feature_distance = value / 100;
                    }
                }
            }

            GridLayout {
                Layout.fillWidth: true
                rows: 4
                columns: 2
                rowSpacing: 10
                columnSpacing: 20
                visible: window.target_type === AppType.ConcentricCircle

                FluText {
                    Layout.preferredWidth: parent.width / 2
                    font: FluTextStyle.BodyStrong
                    text: Lang.inner_circle_inner_radius
                }

                FluText {
                    Layout.fillWidth: true
                    font: FluTextStyle.BodyStrong
                    text: Lang.inner_circle_externer_radius
                }

                FluSpinBox {
                    Layout.preferredWidth: parent.width / 2
                    from: 0
                    to: 999999999
                    value: window.inner_circle_inner_radius * stepSize
                    stepSize: 100
                    editable: true
                    property int decimals: 2
                    property real realValue: value / 100
                    textFromValue: function(value, locale) {
                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                    }

                    valueFromText: function(text, locale) {
                        return Number.fromLocaleString(locale, text) * 100
                    }

                    onValueChanged: {
                        window.inner_circle_inner_radius = value / 100;
                    }
                }

                FluSpinBox {
                    Layout.fillWidth: true
                    from: 0
                    to: 999999999
                    value: window.inner_circle_externer_radius * stepSize
                    stepSize: 100
                    editable: true
                    property int decimals: 2
                    property real realValue: value / 100
                    textFromValue: function(value, locale) {
                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                    }

                    valueFromText: function(text, locale) {
                        return Number.fromLocaleString(locale, text) * 100
                    }

                    onValueChanged: {
                        window.inner_circle_externer_radius = value / 100;
                    }
                }

                FluText {
                    Layout.preferredWidth: parent.width / 2
                    font: FluTextStyle.BodyStrong
                    text: Lang.externer_circle_inner_radius
                }

                FluText {
                    Layout.fillWidth: true
                    font: FluTextStyle.BodyStrong
                    text: Lang.externer_circle_externer_radius
                }

                FluSpinBox {
                    Layout.preferredWidth: parent.width / 2
                    from: 0
                    to: 999999999
                    value: window.externer_circle_inner_radius * stepSize
                    stepSize: 100
                    editable: true
                    property int decimals: 2
                    property real realValue: value / 100
                    textFromValue: function(value, locale) {
                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                    }

                    valueFromText: function(text, locale) {
                        return Number.fromLocaleString(locale, text) * 100
                    }

                    onValueChanged: {
                        window.externer_circle_inner_radius = value / 100;
                    }
                }

                FluSpinBox {
                    Layout.fillWidth: true
                    from: 0
                    to: 999999999
                    value: window.externer_circle_externer_radius * stepSize
                    stepSize: 100
                    editable: true
                    property int decimals: 2
                    property real realValue: value / 100
                    textFromValue: function(value, locale) {
                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                    }

                    valueFromText: function(text, locale) {
                        return Number.fromLocaleString(locale, text) * 100
                    }

                    onValueChanged: {
                        window.externer_circle_externer_radius = value / 100;
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true

                FluButton {
                    Layout.preferredWidth: parent.width / 2
                    Layout.preferredHeight: 50
                    text: Lang.read_params

                    onClicked: {
                        fileDialog.open();
                    }
                }

                FileDialog {
                    id: fileDialog

                    onAccepted: {
                        CalibrateEngine.readLocalCaliFile(currentFile.toString());
                    }
                }

                FluButton {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 50
                    text: Lang.set_calibration_params

                    onClicked: {
                        if(window.target_type === AppType.ConcentricCircle) {
                            CalibrateEngine.setConcentricCirclesRadius(window.inner_circle_inner_radius, window.inner_circle_externer_radius, window.externer_circle_inner_radius, window.externer_circle_externer_radius);
                        }

                        CalibrateEngine.setProjectorCaliParams(window.target_type, window.row_feature_num, window.col_feature_num, window.feature_distance, window.honrizon_pitch, window.vertical_pitch, window.calibrate_type);
                        window.close();
                    }
                }
            }
        }
    }

    Component.onCompleted: {
        CalibrateEngine.setProjectorCaliParams(window.target_type, window.row_feature_num, window.col_feature_num, window.feature_distance, window.honrizon_pitch, window.vertical_pitch, window.calibrate_type);
    }
}
