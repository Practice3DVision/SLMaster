import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import FluentUI 1.0
import "qrc:///example/qml/component"
import "../component"

FluScrollablePage{

    title:"ToggleSwitch"

    FluArea{
        Layout.fillWidth: true
        height: 68
        paddings: 10
        Layout.topMargin: 20
        Row{
            spacing: 30
            anchors.verticalCenter: parent.verticalCenter
            FluToggleSwitch{
                disabled: toggle_switch.checked
            }
            FluToggleSwitch{
                disabled: toggle_switch.checked
                text:"Right"
            }
            FluToggleSwitch{
                disabled: toggle_switch.checked
                text:"Left"
                textRight: false
            }
        }
        FluToggleSwitch{
            id:toggle_switch
            anchors{
                right: parent.right
                verticalCenter: parent.verticalCenter
            }
            text:"Disabled"
        }
    }
    CodeExpander{
        Layout.fillWidth: true
        Layout.topMargin: -1
        code:'FluToggleSwitch{
    text:"Text"
}'
    }


}
