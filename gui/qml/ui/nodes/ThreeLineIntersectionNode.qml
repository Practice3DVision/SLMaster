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
    Layout.preferredWidth: 200
    Layout.preferredHeight: 250
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
                text: Lang.three_line_intersection_node
                font.bold: true
                color: FluTheme.fontPrimaryColor
                background: FluRectangle {
                    radius: [16, 16, 0, 0]
                    color: FluTheme.primaryColor
                }
            }
        }

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 2
            anchors.topMargin: 30

            FluText{
                text: Lang.length
            }

            FluSpinBox {
                Layout.fillWidth: true
                from: -999999999
                to: 999999999
                value: node.length
                editable: true
                up.indicator: undefined
                down.indicator: undefined

                onValueChanged: {
                    node.length = value;
                }
            }

            FluText{
                text: Lang.width
            }

            FluSpinBox {
                Layout.fillWidth: true
                from: -999999999
                to: 999999999
                value: node.width
                editable: true
                up.indicator: undefined
                down.indicator: undefined

                onValueChanged: {
                    node.width = value;
                }
            }

            FluText{
                text: Lang.height
            }

            FluSpinBox {
                Layout.fillWidth: true
                from: -999999999
                to: 999999999
                value: node.height
                editable: true
                up.indicator: undefined
                down.indicator: undefined

                onValueChanged: {
                    node.height = value;
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
