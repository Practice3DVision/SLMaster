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
    Layout.preferredHeight: 120
    width: Layout.preferredWidth
    height: Layout.preferredHeight
    connectable: Qan.NodeItem.OutConnectable

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
                text: Lang.cloudInputNode
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

                Label {
                    Layout.fillWidth: true
                    horizontalAlignment: Text.AlignLeft
                    text: Lang.cloudInputMode
                    font.bold: true
                    color: FluTheme.fontPrimaryColor
                }

                RowLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    FluRadioButton {
                        id: useCameraBtn
                        text: Lang.fromCamera
                        checked: node.mode === CloudInputNode.CloudInputMode.FromCamera

                        onClicked: {
                            useFileBtn.checked = false;
                            node.mode = CloudInputNode.CloudInputMode.FromCamera;
                        }
                    }

                    FluRadioButton {
                        id: useFileBtn
                        text: Lang.fromFile

                        checked: node.mode === CloudInputNode.CloudInputMode.FromFile

                        onClicked: {
                            useCameraBtn.checked = false;
                            node.mode = CloudInputNode.CloudInputMode.FromFile;
                        }
                    }
                }

                RowLayout {
                    visible: useFileBtn.checked

                    FluButton {
                        Layout.preferredWidth: parent.width / 2
                        text: Lang.selectCloudFile

                        onClicked: {
                            selectCloudFileDialog.open();
                        }
                    }

                    FluText {
                        Layout.fillWidth: true
                        text: node.filePath
                        elide: Text.ElideLeft
                    }
                }

                FileDialog {
                    id: selectCloudFileDialog
                    fileMode: FileDialog.Open

                    onAccepted: {
                        node.filePath = file.toString();
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
