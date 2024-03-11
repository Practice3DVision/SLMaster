import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import FluentUI 1.0
import "qrc:///example/qml/component"
import "../component"

FluPage{
    launchMode: FluPageType.SingleTop
    FluRemoteLoader{
        anchors.fill: parent
        source: "https://zhu-zichu.gitee.io/Qt5_T_RemoteLoader.qml"
    }
}
