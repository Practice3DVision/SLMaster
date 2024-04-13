import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic
import QtQuick.Window
import FluentUI

Button {
    property bool disabled: false
    property string contentDescription: ""
    property color normalColor: FluTheme.dark ? Qt.rgba(62/255,62/255,62/255,1) : Qt.rgba(254/255,254/255,254/255,1)
    property color hoverColor: FluTheme.dark ? Qt.rgba(68/255,68/255,68/255,1) : Qt.rgba(246/255,246/255,246/255,1)
    property color disableColor: FluTheme.dark ? Qt.rgba(59/255,59/255,59/255,1) : Qt.rgba(244/255,244/255,244/255,1)
    property color textColor: {
        if(FluTheme.dark){
            if(!enabled){
                return Qt.rgba(131/255,131/255,131/255,1)
            }
            if(pressed){
                return Qt.rgba(162/255,162/255,162/255,1)
            }
            return Qt.rgba(1,1,1,1)
        }else{
            if(!enabled){
                return Qt.rgba(160/255,160/255,160/255,1)
            }
            if(pressed){
                return Qt.rgba(96/255,96/255,96/255,1)
            }
            return Qt.rgba(0,0,0,1)
        }
    }
    property var window : Window.window
    default property alias contentData: menu.contentData
    Accessible.role: Accessible.Button
    Accessible.name: control.text
    Accessible.description: contentDescription
    Accessible.onPressAction: control.clicked()
    id: control
    rightPadding:35
    enabled: !disabled
    focusPolicy:Qt.TabFocus
    verticalPadding: 0
    horizontalPadding:12
    background: Rectangle{
        implicitWidth: 28
        implicitHeight: 28
        border.color: FluTheme.dark ? "#505050" : "#DFDFDF"
        border.width: 1
        radius: 4
        FluFocusRectangle{
            visible: control.activeFocus
            radius:8
        }
        color:{
            if(!enabled){
                return disableColor
            }
            return hovered ? hoverColor :normalColor
        }
        FluIcon{
            iconSource:FluentIcons.ChevronDown
            iconSize: 15
            anchors{
                right: parent.right
                rightMargin: 10
                verticalCenter: parent.verticalCenter
            }
            iconColor:title.color
        }
    }
    contentItem: FluText {
        id:title
        text: control.text
        verticalAlignment: Text.AlignVCenter
        color: control.textColor
    }
    onClicked: {
        if(menu.count !==0){
            var pos = control.mapToItem(null, 0, 0)
            var containerHeight = menu.count*36
            if(window.height>pos.y+control.height+containerHeight){
                menu.y = control.height
            }else if(pos.y>containerHeight){
                menu.y = -containerHeight
            }else{
                menu.y = window.height-(pos.y+containerHeight)
            }
            menu.open()
        }
    }
    FluMenu{
        id:menu
        modal:true
        width: control.width
    }
}
