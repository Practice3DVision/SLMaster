import QtQuick              2.15
import QtQuick.Controls     2.15
import QtQuick.Layouts      1.15
import QtQuick.Dialogs      1.3
import Qt.labs.platform     1.1

import FluentUI 1.0
import SLMasterGui 1.0

import QuickQanava 2.0 as Qan
import "qrc:/../../../../../QuickQanava" as Qan
import "qrc:///ui/global"

Qan.NodeItem {
    id: cloudInputNodeItem
    Layout.preferredWidth: 280
    Layout.preferredHeight: 300
    width: Layout.preferredWidth
    height: Layout.preferredHeight
    connectable: Qan.NodeItem.Connectable

    Item {
        anchors.fill: parent
        FluRectangle {
            anchors.fill: parent
            radius: [16, 16, 4, 4]
            color: FluTheme.dark ? Qt.rgba(223/255,173/255,135/255,1) : Qt.rgba(77/255,139/255,110/255,1)
            Label {
                id: title_label
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.right: parent.right
                height: 30
                padding: -2
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: Lang.greedyProjectionTriangulation
                font.bold: true
                color: FluTheme.fontPrimaryColor
                background: FluRectangle {
                    radius: [16, 16, 0, 0]
                    color: FluTheme.primaryColor
                }
            }

            ColumnLayout {
                anchors.top: title_label.bottom
                anchors.left: parent.left
                anchors.right:parent.right
                anchors.bottom: parent.bottom
                anchors.margins: 2

                GridLayout {
                    Layout.fillWidth: true
                    rows: 3
                    columns: 2

                    FluText {
                        text: Lang.kSearch
                    }

                    FluText{
                        text: Lang.multiplier
                    }

                    FluSpinBox {
                        Layout.fillWidth: parent.width / 2
                        from: -999999999
                        to: 999999999
                        value: node.kSearch * stepSize
                        stepSize: 100
                        editable: true
                        property int decimals: 2
                        property real realValue: value / 100
                        up.indicator: undefined
                        down.indicator: undefined
                        textFromValue: function(value, locale) {
                            return Number(value / 100).toLocaleString(locale, 'f', decimals)
                        }

                        valueFromText: function(text, locale) {
                            return Number.fromLocaleString(locale, text) * 100
                        }

                        onValueChanged: {
                            node.kSearch = value / 100;
                        }
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.multiplier * stepSize
                        stepSize: 100
                        editable: true
                        property int decimals: 2
                        property real realValue: value / 100
                        up.indicator: undefined
                        down.indicator: undefined
                        textFromValue: function(value, locale) {
                            return Number(value / 100).toLocaleString(locale, 'f', decimals)
                        }

                        valueFromText: function(text, locale) {
                            return Number.fromLocaleString(locale, text) * 100
                        }

                        onValueChanged: {
                            node.multiplier = value / 100;
                        }
                    }

                    FluText{
                        text: Lang.maxNearestNumber
                    }

                    FluText{
                        text: Lang.searchRadius
                    }

                    FluSpinBox {
                        Layout.fillWidth: parent.width / 2
                        from: -999999999
                        to: 999999999
                        value: node.maxNearestNumber * stepSize
                        stepSize: 100
                        editable: true
                        property int decimals: 2
                        property real realValue: value / 100
                        up.indicator: undefined
                        down.indicator: undefined
                        textFromValue: function(value, locale) {
                            return Number(value / 100).toLocaleString(locale, 'f', decimals)
                        }

                        valueFromText: function(text, locale) {
                            return Number.fromLocaleString(locale, text) * 100
                        }

                        onValueChanged: {
                            node.maxNearestNumber = value / 100;
                        }
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.searchRadius * stepSize
                        stepSize: 100
                        editable: true
                        property int decimals: 2
                        property real realValue: value / 100
                        up.indicator: undefined
                        down.indicator: undefined
                        textFromValue: function(value, locale) {
                            return Number(value / 100).toLocaleString(locale, 'f', decimals)
                        }

                        valueFromText: function(text, locale) {
                            return Number.fromLocaleString(locale, text) * 100
                        }

                        onValueChanged: {
                            node.searchRadius = value / 100;
                        }
                    }

                    FluText{
                        text: Lang.minimumAngle
                    }

                    FluText{
                        text: Lang.maximumAngle
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.minimumAngle * stepSize
                        stepSize: 100
                        editable: true
                        property int decimals: 2
                        property real realValue: value / 100
                        up.indicator: undefined
                        down.indicator: undefined
                        textFromValue: function(value, locale) {
                            return Number(value / 100).toLocaleString(locale, 'f', decimals)
                        }

                        valueFromText: function(text, locale) {
                            return Number.fromLocaleString(locale, text) * 100
                        }

                        onValueChanged: {
                            node.minimumAngle = value / 100;
                        }
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.maximumAngle * stepSize
                        stepSize: 100
                        editable: true
                        property int decimals: 2
                        property real realValue: value / 100
                        up.indicator: undefined
                        down.indicator: undefined
                        textFromValue: function(value, locale) {
                            return Number(value / 100).toLocaleString(locale, 'f', decimals)
                        }

                        valueFromText: function(text, locale) {
                            return Number.fromLocaleString(locale, text) * 100
                        }

                        onValueChanged: {
                            node.maximumAngle = value / 100;
                        }
                    }
                }

                FluText{
                    text: Lang.maximumSurfaceAngle
                }

                FluSpinBox {
                    Layout.fillWidth: true
                    from: -999999999
                    to: 999999999
                    value: node.maximumSurfaceAngle * stepSize
                    stepSize: 100
                    editable: true
                    property int decimals: 2
                    property real realValue: value / 100
                    up.indicator: undefined
                    down.indicator: undefined
                    textFromValue: function(value, locale) {
                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                    }

                    valueFromText: function(text, locale) {
                        return Number.fromLocaleString(locale, text) * 100
                    }

                    onValueChanged: {
                        node.maximumSurfaceAngle = value / 100;
                    }
                }

                FluCheckBox {
                    text: Lang.normalConsistency
                    checked: node.normalConsistency

                    onClicked: {
                        node.normalConsistency = !node.normalConsistency;
                    }
                }
            }
        }

        Qan.BottomRightResizer {
            target: cloudInputNodeItem
            handler: Rectangle {
                color: "red"
                width: 8
                height: 8
                radius: 4
            }
        }
    }
}
