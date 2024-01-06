import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import FluentUI 1.0

Item {
    id: app

    Connections{
        target: FluTheme
        function onDarkModeChanged(){
            SettingsHelper.saveDarkMode(FluTheme.darkMode);
        }
    }

    Connections{
        target: FluApp
        function onVsyncChanged(){
            SettingsHelper.saveVsync(FluApp.vsync);
        }
        function onUseSystemAppBarChanged(){
            SettingsHelper.saveUseSystemAppBar(FluApp.useSystemAppBar);
        }
    }

    Component.onCompleted: {
        FluApp.init(app);
        FluApp.useSystemAppBar = false;
        FluApp.vsync = SettingsHelper.getVsync();
        FluTheme.darkMode = SettingsHelper.getDarkMode();
        FluTheme.enableAnimation = true;
        FluApp.routes = {
            "/": "qrc:/ui/window/MainWindow.qml",
            "/SettingsPage": "qrc:/ui/component/SettingsPage.qml",
            "/AboutWindow": "qrc:ui/window/AboutWindow.qml",
            "/CalibrationSettinsWindow": "qrc:/ui/window/CalibrationSettingsWindow.qml",
            "/ProjectorCaliParamsSettinsWindow": "qrc:/ui/window/ProjectorCaliParamsSettinsWindow.qml",
        }
        FluApp.initialRoute = "/";
        FluApp.run();
    }
}
