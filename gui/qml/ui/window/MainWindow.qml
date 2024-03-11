import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Qt.labs.platform 1.1

import FluentUI 1.0
import SLMasterGui 1.0

import "qrc:///ui/global"
import "../global"
import "qrc:///ui/component"
import "../component"

FluWindow {
    id: window
    width: 1280
    height: 680
    minimumWidth: 520
    minimumHeight: 200
    title: "SLMasterGui"
    launchMode: FluWindowType.SingleTask
    fitsAppBarWindows: true

    Component.onCompleted: {
        JSONListModel.query = "$.camera.algorithm[*]";
        JSONListModel.source = "qrc:/res/config/binoocularCameraConfig.json";

        //初始化所有的vtk item
        timeline_nav.curPage = 3;
        timeline_nav.curPage = 5;
        timeline_nav.curPage = 0;
        timeline_nav.updateCheckedState();
    }

    appBar: FluAppBar {
        width: window.width
        height: 30
        darkText: Lang.dark_mode
        showDark: true
        darkClickListener: (button)=>handleDarkChanged(button)
        closeClickListener: ()=>{dialog_close.open()}
        z: 7
    }

    SystemTrayIcon {
        id: sys_tray_icon
        visible: true
        icon.source: AppIcon.application
        tooltip: "SLMasterGui"
        menu: Menu {
            MenuItem {
                text: Lang.exit
                onTriggered: {
                    FluApp.exit(0)
                }
            }
        }
        onActivated: (reason)=>{
            if(reason === SystemTrayIcon.Trigger) {
                window.show();
                window.raise();
                window.requestActivate();
            }
        }
    }

    Timer {
        id: timer_window_hide_delay
        interval: 150
        onTriggered: {
            window.hide()
        }
    }

    FluContentDialog {
        id: dialog_close
        title: Lang.exit
        message: Lang.exit_confirm
        negativeText: Lang.minimum
        buttonFlags: FluContentDialogType.NegativeButton | FluContentDialogType.NeutralButton | FluContentDialogType.PositiveButton
        onNegativeClicked: {
            sys_tray_icon.showMessage(Lang.frendly_tip, Lang.hide_tip)
            timer_window_hide_delay.restart()
        }
        positiveText: Lang.exit
        neutralText: Lang.cancel
        onPositiveClicked: {
            FluApp.exit(0)
        }
    }

    FluNavigationView {
        id: nav_view
        width: parent.width
        height: parent.height
        items: ItemsOriginal
        pageMode: FluNavigationViewType.Stack
        topPadding: FluTools.isMacos() ? 20 : 0
        displayMode: FluNavigationViewType.Minimal
        logo: AppIcon.application
        title: "SLMasterGui"

        Component.onCompleted: {
            ItemsOriginal.navigationView = nav_view;
            setCurrentIndex(0);
        }
    }

    FluIconButton{
        id:btn_setttings
        iconSource: FluentIcons.Settings
        iconSize: 15
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 120
        anchors.topMargin: 4
        implicitWidth: 30
        implicitHeight: 30
        clip: true
        onClicked: {
            FluApp.navigate("/SettingsPage")
        }
    }

    FluIconButton{
        id:btn_about
        iconSource: FluentIcons.Help
        iconSize: 15
        anchors.left: btn_setttings.right
        anchors.top: parent.top
        anchors.topMargin: 4
        implicitWidth: 30
        implicitHeight: 30
        clip: true
        onClicked: {
            FluApp.navigate("/AboutWindow")
        }
    }

    TimeLineNavigator {
        id: timeline_nav
        anchors.top: parent.top
        width: parent.width
        height: 40
        anchors.topMargin: 40

        onCurPageChanged: {
            if(curPage == 3 || curPage == 5) {
                window.fillBackground = false;
                window.fillBackgroundChanged();

                var indexItem = curPage == 3 ? 0 : 1;
                if(GlobalSignals.render_items[indexItem] !== undefined) {
                    VTKProcessEngine.setCurRenderItem(GlobalSignals.render_items[indexItem]);
                }
            }
            else {
                window.fillBackground = true;
                window.fillBackgroundChanged();
            }

            nav_view.setCurrentIndex(curPage);
        }

        Connections {
            target: GlobalSignals

            function onStartScan() {
                timeline_nav.curPage = 3;
                timeline_nav.updateCheckedState();
            }
        }

        Connections {
            target: VTKProcessEngine

            function onPostProcessOutput() {
                timeline_nav.curPage = 5;
                timeline_nav.updateCheckedState();
            }
        }
    }

    Component {
        id: com_reveal
        CircularReveal {
            id: reveal
            target: window.contentItem
            anchors.fill: parent
            onAnimationFinished: {
                loader_reveal.sourceComponent = undefined;
            }

            onImageChanged: {
                changeDark();
            }
        }
    }

    FluLoader {
        id: loader_reveal
        anchors.fill: parent
    }

    function distance(x1, y1, x2, y2) {
        return Math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    }

    function handleDarkChanged(button) {
        if(!FluTheme.enableAnimation || window.fitsAppBarWindows === false) {
            changeDark();
        }
        else {
            loader_reveal.sourceComponent = com_reveal;
            var target = window.contentItem;
            var pos = button.mapToItem(target, 0, 0);
            var mouseX = pos.x;
            var mouseY = pos.y;
            var radius = Math.max(distance(mouseX, mouseY, 0, 0), distance(mouseX, mouseY, target.width, 0),distance(mouseX,mouseY,0,target.height),distance(mouseX,mouseY,target.width,target.height));
            var reveal = loader_reveal.item;
            reveal.start(reveal.width * Screen.devicePixelRatio, reveal.height * Screen.devicePixelRatio, Qt.point(mouseX, mouseY), radius);
        }
    }

    function changeDark() {
        if(FluTheme.dark) {
            FluTheme.darkMode = FluThemeType.Light;
        }
        else {
            FluTheme.darkMode = FluThemeType.Dark;
        }
    }
}
