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
    Layout.preferredWidth: 250
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
                text: Lang.passThroughFilterNode
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

                FluCheckBox {
                    Layout.fillWidth: true
                    text: Lang.enableX
                    font.bold: true
                    checked: node.filterX

                    onClicked: {
                        node.filterX = !node.filterX;
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    visible: node.filterX
                    rows: 2
                    columns: 2

                    FluText {
                        text: Lang.minX
                    }

                    FluText{
                        text: Lang.maxX
                    }

                    FluSpinBox {
                        Layout.preferredWidth: parent.width / 2
                        from: -999999999
                        to: 999999999
                        value: node.minX * stepSize
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
                            node.minX = value / 100;
                        }
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.maxX * stepSize
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
                            node.maxX = value / 100;
                        }
                    }
                }

                FluCheckBox {
                    Layout.fillWidth: true
                    text: Lang.enableY
                    font.bold: true
                    checked: node.filterY

                    onClicked: {
                        node.filterY = !node.filterY;
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    visible: node.filterY
                    rows: 2
                    columns: 2

                    FluText {
                        text: Lang.minY
                    }

                    FluText{
                        text: Lang.maxY
                    }

                    FluSpinBox {
                        Layout.preferredWidth: parent.width / 2
                        from: -999999999
                        to: 999999999
                        value: node.minY * stepSize
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
                            node.minY = value / 100;
                        }
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.maxY * stepSize
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
                            node.maxY = value / 100;
                        }
                    }
                }


                FluCheckBox {
                    Layout.fillWidth: true
                    text: Lang.enableZ
                    font.bold: true
                    checked: node.filterZ

                    onClicked: {
                        node.filterZ = !node.filterZ;
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    visible: node.filterZ
                    rows: 2
                    columns: 2

                    FluText {
                        text: Lang.minZ
                    }

                    FluText{
                        text: Lang.maxZ
                    }

                    FluSpinBox {
                        Layout.preferredWidth: parent.width / 2
                        from: -999999999
                        to: 999999999
                        value: node.minZ * stepSize
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
                            node.minZ = value / 100;
                        }
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.maxZ * stepSize
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
                            node.maxZ = value / 100;
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
