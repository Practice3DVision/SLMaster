import QtQuick 2.15
import QtQuick.Controls 2.15
import FluentUI 1.0

QtObject {
    readonly property string key : FluTools.uuid()
    property int _idx
    property var _ext
    property var _parent
}
