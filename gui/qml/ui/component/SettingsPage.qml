import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15

import FluentUI 1.0
import SLMasterGui 1.0

import "qrc:/ui/global"
import "../global"

FluWindow {
    id: window
    launchMode: FluWindowType.SingleInstance
    minimumWidth: 480
    minimumHeight: 680

    FluScrollablePage {
        anchors.fill: parent
        title: Lang.settings

        FluViewModel{
            objectName: "SettingsViewModel"
            scope: FluViewModelType.Application
            property int displayMode

            onInitData: {
                displayMode = FluNavigationViewType.Auto
            }
        }

        FluArea {
            Layout.fillWidth: true
            height: 80
            paddings: 10

            ColumnLayout {
                spacing: 10
                anchors {
                    left: parent.left
                    top: parent.top
                }

                FluText {
                    text: Lang.locale
                    font: FluTextStyle.BodyStrong
                    Layout.bottomMargin: 4
                }

                Flow {
                    spacing: 5
                    Repeater {
                        model: Lang.__localeList
                        delegate: FluRadioButton {
                            checked: Lang.__locale === modelData
                            text: modelData
                            clickListener: function() {
                                Lang.__locale = modelData;
                            }
                        }
                    }
                }
            }
        }

        FluArea{
            Layout.fillWidth: true
            Layout.topMargin: 20
            height: 128
            paddings: 10

            ColumnLayout{
                spacing: 5
                anchors{
                    top: parent.top
                    left: parent.left
                }
                FluText{
                    text:Lang.dark_mode
                    font: FluTextStyle.BodyStrong
                    Layout.bottomMargin: 4
                }
                Repeater{
                    model: [{title:Lang.system_mode,mode:FluThemeType.System},{title:Lang.light_mode,mode:FluThemeType.Light},{title:Lang.dark_mode,mode:FluThemeType.Dark}]
                    delegate:  FluRadioButton{
                        checked : FluTheme.darkMode === modelData.mode
                        text:modelData.title
                        clickListener:function(){
                            FluTheme.darkMode = modelData.mode
                            SettingsHelper.saveDarkMode(FluApp.darkMode)
                        }
                    }
                }
            }
        }

        FluArea{
                Layout.fillWidth: true
                Layout.topMargin: 20
                height: 50
                paddings: 10
                FluCheckBox{
                    text:Lang.v_sync
                    checked: FluApp.vsync
                    anchors.verticalCenter: parent.verticalCenter
                    onClicked: {
                        FluApp.vsync = !FluApp.vsync
                        SettingsHelper.saveVsync(FluApp.vsync)
                        dialog_restart.open()
                    }
                }
            }

            FluArea{
                Layout.fillWidth: true
                Layout.topMargin: 20
                height: 50
                paddings: 10
                FluCheckBox{
                    text: Lang.use_system_bar
                    checked: FluApp.useSystemAppBar
                    anchors.verticalCenter: parent.verticalCenter
                    onClicked: {
                        FluApp.useSystemAppBar = !FluApp.useSystemAppBar
                        SettingsHelper.saveUseSystemAppBar(FluApp.useSystemAppBar)
                        dialog_restart.open()
                    }
                }
            }

            FluArea{
                Layout.fillWidth: true
                Layout.topMargin: 20
                height: 50
                paddings: 10
                FluCheckBox{
                    text: Lang.fits_app_bar
                    checked: window.fitsAppBarWindows
                    anchors.verticalCenter: parent.verticalCenter
                    onClicked: {
                        window.fitsAppBarWindows = !window.fitsAppBarWindows
                    }
                }
            }

            FluArea{
                Layout.fillWidth: true
                Layout.topMargin: 20
                height: 50
                paddings: 10
                FluCheckBox{
                    text: Lang.software_render
                    checked: SettingsHelper.getRender() === "software"
                    anchors.verticalCenter: parent.verticalCenter
                    onClicked: {
                        if(SettingsHelper.getRender() === "software"){
                            SettingsHelper.saveRender("")
                        }else{
                            SettingsHelper.saveRender("software")
                        }
                        dialog_restart.open()
                    }
                }
            }

            FluContentDialog{
                id:dialog_restart
                title: Lang.frendly_tip
                message: Lang.check_restart
                buttonFlags: FluContentDialogType.NegativeButton | FluContentDialogType.PositiveButton
                negativeText: Lang.cancel
                positiveText: Lang.confirm
                onPositiveClicked:{
                    FluApp.exit(931)
                }
            }
    }
}
