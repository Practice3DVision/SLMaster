import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import FluentUI 1.0

import "qrc:///ui/global"

FluWindow {

    id:window
    title: Lang.about
    width: 600
    height: 600
    fixSize: true
    launchMode: FluWindowType.SingleTask

    FluScrollablePage {
        anchors.fill: parent
        ColumnLayout{
            anchors{
                top: parent.top
                left: parent.left
                right: parent.right
            }

            RowLayout{
                Layout.topMargin: 20
                Layout.leftMargin: 15
                spacing: 14
                FluText{
                    text:"SLMasterGui"
                    font: FluTextStyle.Title
                    MouseArea{
                        anchors.fill: parent
                        onClicked: {
                            FluApp.navigate("/")
                        }
                    }
                }
            }

            RowLayout{
                spacing: 14
                Layout.topMargin: 20
                Layout.leftMargin: 15
                FluText{
                    text: Lang.author
                }
                FluText{
                    text: Lang.yunhuangliu
                    Layout.alignment: Qt.AlignBottom
                }
            }

            RowLayout{
                spacing: 14
                Layout.leftMargin: 15
                FluText{
                    text: "GitHub："
                }
                FluTextButton{
                    id:text_hublink
                    topPadding:0
                    bottomPadding:0
                    text: "https://github.com/Yunhuang-Liu"
                    Layout.alignment: Qt.AlignBottom
                    onClicked: {
                        Qt.openUrlExternally(text_hublink.text)
                    }
                }
            }

            RowLayout{
                spacing: 14
                Layout.leftMargin: 15
                FluText{
                    text: "CSDN："
                }
                FluTextButton{
                    topPadding:0
                    bottomPadding:0
                    text: "https://blog.csdn.net/m0_56071788?spm=1010.2135.3001.5421"
                    Layout.alignment: Qt.AlignBottom
                    onClicked: {
                        Qt.openUrlExternally(text)
                    }
                }
            }

            FluText {
                Layout.topMargin: 20
                Layout.bottomMargin: 20
                id:text_info
                text: Lang.about_info
            }

            FluText{
                text: Lang.wechat
            }

            Image{
                Layout.preferredWidth: 200
                Layout.preferredHeight: 250
                Layout.alignment: Qt.AlignHCenter

                source: "qrc:/res/image/wechat.jpg"
                fillMode: Image.PreserveAspectFit
                horizontalAlignment: Image.AlignHCenter
            }
        }
    }
}
