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
    anchors.top: parent.top
    anchors.bottom: parent.bottom
    anchors.left: parent.left
    anchors.right: parent.right

    Component{
        id:com_scan_mode_type
        Item{
            id: control
            width: 320
            height: 340
            FluShadow{
                radius:5
                anchors.fill: item_content
            }
            FluClip{
                id:item_content
                radius: [5,5,5,5]
                width: 300
                height: 320
                anchors.centerIn: parent.Center
                Rectangle{
                    anchors.fill: parent
                    radius: 5
                    color:FluTheme.itemHoverColor
                    visible: item_mouse.containsMouse
                }
                Rectangle{
                    anchors.fill: parent
                    radius: 5
                    color:Qt.rgba(0,0,0,0.0)
                    visible: !item_mouse.containsMouse
                }
                ColumnLayout{
                    width: parent.width

                    Image {
                        Layout.topMargin: 20
                        Layout.leftMargin: 20
                        Layout.preferredWidth: 50
                        Layout.preferredHeight: 50
                        source: model.icon
                    }
                    FluText{
                        text: model.title
                        font: FluTextStyle.Body
                        Layout.topMargin: 20
                        Layout.leftMargin: 20
                        Layout.rightMargin: 20
                        Layout.fillWidth: true
                    }
                    FluText{
                        text: model.desc
                        Layout.topMargin: 5
                        Layout.preferredWidth: 160
                        Layout.leftMargin: 20
                        Layout.rightMargin: 20
                        Layout.fillWidth: true
                        color: FluColors.Grey120
                        font.pixelSize: 12
                        wrapMode: Text.WrapAnywhere
                    }
                }
                FluIcon{
                    iconSource: FluentIcons.OpenInNewWindow
                    iconSize: 15
                    anchors{
                        bottom: parent.bottom
                        right: parent.right
                        rightMargin: 10
                        bottomMargin: 10
                    }
                }
                MouseArea{
                    id:item_mouse
                    anchors.fill: parent
                    hoverEnabled: true

                    onClicked: {
                        loader.setSource(url);
                        CameraEngine.setScanMode(scan_mode);
                        GlobalSignals.scan_mode = scan_mode;
                    }
                }
            }
        }
    }

    Connections {
        target: Lang

        function on__LocaleChanged() {
            model_header.element_titles = [Lang.static_scan_mode, Lang.dynamic_scan_mode, Lang.offline_scan_mode];
            model_header.element_tips = [Lang.static_scan_mode_tip, Lang.dynamic_scan_mode_tip, Lang.offline_scan_mode_tip];
        }
    }

    ListModel{
        id:model_header

        property var element_titles: [Lang.static_scan_mode, Lang.dynamic_scan_mode, Lang.offline_scan_mode]
        property var element_tips: [Lang.static_scan_mode_tip, Lang.dynamic_scan_mode_tip, Lang.offline_scan_mode_tip]

        Component.onCompleted: {
            append({icon:"qrc:/res/image/icons8-photo-96.png", title: model_header.element_titles[0], desc: model_header.element_tips[0], scan_mode: AppType.Static, url: "qrc:/ui/page/SubPage_StaticScanMode.qml"});
            append({icon:"qrc:/res/image/icons8-film-roll-96.png", title: model_header.element_titles[1], desc: model_header.element_tips[1], scan_mode: AppType.Dynamic, url:"qrc:/ui/page/SubPage_OfflineCameraCalibration.qml"});
            append({icon:"qrc:/res/image/icons8-previous-96.png", title: model_header.element_titles[2], desc: model_header.element_tips[2], scan_mode: AppType.Offline, url:"qrc:/ui/page/SubPage_OfflineScanMode.qml"});
        }

        onElement_tipsChanged: {
            setProperty(0, "title", model_header.element_titles[0]);
            setProperty(0, "desc", model_header.element_tips[0]);
            setProperty(1, "title", model_header.element_titles[1]);
            setProperty(1, "desc", model_header.element_tips[1]);
            setProperty(2, "title", model_header.element_titles[2]);
            setProperty(2, "desc", model_header.element_tips[2]);
        }
    }

    Component {
        id: com_init

        ColumnLayout {
            anchors.fill: parent
            anchors.leftMargin: 20
            anchors.topMargin: 20
            spacing: 20

            FluText{
                text: Lang.scan_mode_selcet
                font: FluTextStyle.Subtitle
            }

            ListView{
                id: list
                Layout.fillWidth: true
                Layout.fillHeight: true
                //Layout.alignment: Qt.AlignTop
                spacing: 60
                orientation: ListView.Horizontal
                //highlight: Rectangle { color: "lightsteelblue"; radius: 5 }
                model: model_header
                header: Item{height: 10;width: 10}
                footer: Item{height: 10;width: 10}
                clip: false
                delegate: com_scan_mode_type
            }
        }
    }

    FluLoader {
        id: loader
        anchors.fill: parent
        sourceComponent: com_init
    }

    Connections {
        target: loader.item

        function onBack() {
            loader.sourceComponent = com_init;
        }
    }
}
