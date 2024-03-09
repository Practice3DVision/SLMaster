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
    Layout.preferredWidth: 220
    Layout.preferredHeight: 200
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
                text: Lang.staticRemovel
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
                    rows: 4
                    columns: 1

                    FluText {
                        text: Lang.meanK
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.k * stepSize
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
                            node.k = value / 100;
                        }
                    }

                    FluText{
                        text: Lang.stdThresh
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.stdThreshold * stepSize
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
                            node.stdThreshold = value / 100;
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
