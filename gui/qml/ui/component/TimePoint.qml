import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15

import FluentUI 1.0

import "qrc:/ui/global"
import "../global"

Item {
    id: item
    implicitWidth: 120
    implicitHeight: 40

    property string functionName
    property string statusText
    property color normalColor: FluTheme.dark ? Qt.rgba(1,1,1,1) : Qt.rgba(50/255,50/255,50/255,1)
    property color borderNormalColor: button.checked ? FluTheme.primaryColor : FluTheme.dark ? Qt.rgba(161/255,161/255,161/255,1) : Qt.rgba(141/255,141/255,141/255,1)

    signal checkStateChanged

    function setChecked(state) {
        button.checked = state;
    }

    ColumnLayout {
        id: layout
        anchors.fill: parent

        FluText {
            id: function_text
            Layout.fillHeight: true
            Layout.alignment: Qt.AlignHCenter
            text: item.functionName
            color: item.normalColor
        }

        FluText {
            Layout.fillHeight: true
            Layout.alignment: Qt.AlignHCenter
            text: item.statusText
            color: borderNormalColor
            font.bold: button.checked
            font.pointSize: function_text.font.pointSize * 1.2
        }

        FluRadioButton {
            id: button
            Layout.fillHeight: true
            Layout.alignment: Qt.AlignHCenter
            text: ""
            checked: false
            size: checked ? 24 : 18
            verticalPadding: 1
            horizontalPadding: 1
            Layout.leftMargin: checked ? -3 : 0
            Layout.bottomMargin: checked ? 2 : 0

            onClicked: {
                checkStateChanged();
            }
        }
    }
}
