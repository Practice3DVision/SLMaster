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

    signal back
    signal configureRotator

    FluIconButton {
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.topMargin: 5
        anchors.rightMargin: 5
        iconSource: FluentIcons.Back
        iconSize: 16

        onClicked: {
            root.back();
        }
    }

    Component{
        id:com_item
        Item{
            width: 320
            height: 240
            FluArea{
                radius: 8
                width: 300
                height: 220
                anchors.centerIn: parent
                Rectangle{
                    anchors.fill: parent
                    radius: 8
                    color:{
                        if(item_mouse.containsMouse){
                            return FluTheme.itemHoverColor
                        }
                        return FluTheme.itemNormalColor
                    }
                }
                Image{
                    id:item_icon
                    height: 80
                    width: 80
                    source: image
                    anchors{
                        left: parent.left
                        leftMargin: 20
                        verticalCenter: parent.verticalCenter
                    }
                }
                FluText{
                    id:item_title
                    text: title
                    font: FluTextStyle.BodyStrong
                    anchors{
                        left: item_icon.right
                        leftMargin: 20
                        top: item_icon.top
                    }
                }
                FluText{
                    id:item_desc
                    text: desc
                    color:FluColors.Grey120
                    wrapMode: Text.WrapAnywhere
                    elide: Text.ElideRight
                    font: FluTextStyle.Caption
                    maximumLineCount: 10
                    anchors{
                        left: item_title.left
                        right: parent.right
                        rightMargin: 20
                        top: item_title.bottom
                        topMargin: 5
                    }
                }

                Rectangle{
                    height: 12
                    width: 12
                    radius:  6
                    color: FluTheme.primaryColor
                    anchors{
                        right: parent.right
                        top: parent.top
                        rightMargin: 14
                        topMargin: 14
                    }
                }

                MouseArea{
                    id:item_mouse
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: {
                        console.debug(emit_signal_index)
                        if(emit_signal_index === 0) {
                            GlobalSignals.startScan();
                        }
                        else if(emit_signal_index === 1) {
                            configureRotator();
                        }
                    }
                }
            }
        }
    }

    ListModel {
        id: model

        Component.onCompleted: {
            updateModel();
        }
    }

    function updateModel() {
        model.clear()

        model.append({
            title: Lang.pure_scan_mode,
            desc: Lang.pure_scan_desc,
            image: "qrc:/res/image/icons8-clean-96.png",
            emit_signal_index: 0,
        })

        model.append({
            title: Lang.turret_scan_mode,
            desc: Lang.turret_scan_desc,
            image: "qrc:/res/image/icons8-process-96.png",
            emit_signal_index: 1,
        })
    }

    Connections {
        target: Lang
        function on__LocaleChanged() {
            updateModel();
        }
    }

    Connections {
        target: root

        function onConfigureRotator() {
            console.debug("configure rotator!");
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 20
        anchors.topMargin: 20
        spacing: 20

        FluText {
            text: Lang.configure_static_scan_algorithm
            font: FluTextStyle.Subtitle
            horizontalAlignment: Text.AlignLeft
        }

        GridView{
            Layout.fillWidth: true
            Layout.fillHeight: true
            cellHeight: 240
            cellWidth: 320
            model: model
            interactive: false
            delegate: com_item
        }
    }
}
