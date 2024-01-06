import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import FluentUI 1.0
import SLMasterGui 1.0

import "qrc:///ui/global"

FluWindow {
    id:window
    title: Lang.calibration
    width: 500
    height: 460
    fixSize: true
    launchMode: FluWindowType.SingleTask

    property int target_type: AppType.Blob
    property int row_feature_num: 13
    property int col_feature_num: 10
    property real feature_distance: 15
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

            FluText {
                Layout.fillWidth: true
                font: FluTextStyle.BodyStrong
                text: Lang.target_type
            }

            FluComboBox {
                Layout.preferredWidth: parent.width / 3
                model: [Lang.chess_board, Lang.blob, Lang.concentric_circle]

                onCurrentIndexChanged: {
                    window.target_type = currentIndex;
                }
            }

            FluRadioButton {
                text:Lang.export_as_left_color_calibration
                checked: window.export_as_left_color_calibration

                onCheckedChanged: {
                    window.export_as_left_color_calibration = checked;
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
                    Layout.fillWidth: true
                    Layout.preferredHeight: 50
                    text: Lang.single_calibration

                    onClicked: {
                        if(window.target_type === AppType.ConcentricCircle) {
                            CalibrateEngine.setConcentricCirclesRadius(window.inner_circle_inner_radius, window.inner_circle_externer_radius, window.externer_circle_inner_radius, window.externer_circle_externer_radius);
                        }

                        CalibrateEngine.singleCalibrate(window.target_type, window.row_feature_num, window.col_feature_num, window.feature_distance);
                        window.close();
                    }
                }

                FluButton {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 50
                    text: Lang.stereo_calibration

                    onClicked: {
                        if(window.target_type === AppType.ConcentricCircle) {
                            CalibrateEngine.setConcentricCirclesRadius(window.inner_circle_inner_radius, window.inner_circle_externer_radius, window.externer_circle_inner_radius, window.externer_circle_externer_radius);
                        }

                        CalibrateEngine.stereoCalibrate(window.target_type, window.row_feature_num, window.col_feature_num, window.feature_distance, window.export_as_left_color_calibration);
                        window.close();
                    }
                }
            }
        }
    }
}
