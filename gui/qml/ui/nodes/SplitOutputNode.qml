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
    id: splitOutputNodeItem
    Layout.preferredWidth: 130
    Layout.preferredHeight: 100
    width: Layout.preferredWidth
    height: Layout.preferredHeight
    connectable: Qan.NodeItem.InConnectable

    Item {
        anchors.fill: parent

        FluRectangle {
            id: title
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
                text: Lang.split_output_node
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
                text: Lang.split_port_id
            }

            FluSpinBox {
                Layout.fillWidth: true
                from: 0
                to: 999999999
                value: node.portIndex
                editable: true
                up.indicator: undefined
                down.indicator: undefined

                onValueChanged: {
                    node.portIndex = value;
                }
            }
        }

        Qan.BottomRightResizer {
            target: splitOutputNodeItem
            handler: Rectangle {
                color: "red"
                width: 8
                height: 8
                radius: 4
            }
        }
    }
}
