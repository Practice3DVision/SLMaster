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
                text: Lang.poisson
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
                        text: Lang.minDepth
                    }

                    FluText{
                        text: Lang.maxDepth
                    }

                    FluSpinBox {
                        Layout.fillWidth: parent.width / 2
                        from: -999999999
                        to: 999999999
                        value: node.minDepth * stepSize
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
                            node.minDepth = value / 100;
                        }
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.maxDepth * stepSize
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
                            node.maxDepth = value / 100;
                        }
                    }

                    FluText{
                        text: Lang.scale
                    }

                    FluText{
                        text: Lang.solverDivide
                    }

                    FluSpinBox {
                        Layout.fillWidth: parent.width / 2
                        from: -999999999
                        to: 999999999
                        value: node.scale * stepSize
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
                            node.scale = value / 100;
                        }
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.solverDivide * stepSize
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
                            node.solverDivide = value / 100;
                        }
                    }

                    FluText{
                        text: Lang.isoDivide
                    }

                    FluText{
                        text: Lang.minSamplePoints
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.isoDivide * stepSize
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
                            node.isoDivide = value / 100;
                        }
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.minSamplePoints * stepSize
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
                            node.minSamplePoints = value / 100;
                        }
                    }
                }

                FluText{
                    text: Lang.degree
                }

                FluSpinBox {
                    Layout.fillWidth: true
                    from: -999999999
                    to: 999999999
                    value: node.degree * stepSize
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
                        node.degree = value / 100;
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    FluCheckBox {
                        Layout.preferredWidth: parent.width / 2
                        text: Lang.confidence
                        checked: node.confidence

                        onClicked: {
                            node.confidence = !node.confidence;
                        }
                    }

                    FluCheckBox {
                        Layout.fillWidth: true
                        text: Lang.manifold
                        checked: node.manifold

                        onClicked: {
                            node.manifold = !node.manifold;
                        }
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
