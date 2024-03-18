import QtQuick              2.15
import QtQuick.Controls     2.15
import QtQuick.Layouts      1.15
import QtQuick.Dialogs      1.3
import Qt.labs.platform     1.1

import FluentUI 1.0
import SLMasterGui 1.0

import QuickQanava 2.0 as Qan
import "qrc:/../../../../../QuickQanava" as Qan
import "qrc:///ui/global"

Qan.NodeItem {
    id: cloudInputNodeItem
    Layout.preferredWidth: 200
    Layout.preferredHeight: 250
    width: Layout.preferredWidth
    height: Layout.preferredHeight
    connectable: Qan.NodeItem.Connectable

    Item {
        anchors.fill: parent
        FluRectangle {
            anchors.fill: parent
            radius: [16, 16, 4, 4]
            color: FluTheme.dark ? Qt.rgba(223/255,173/255,135/255,1) : Qt.rgba(77/255,139/255,110/255,1)
            Label {
                id: title_label
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.right: parent.right
                height: 30
                padding: -2
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: Lang.sac_segment
                font.bold: true
                color: FluTheme.fontPrimaryColor
                background: FluRectangle {
                    radius: [16, 16, 0, 0]
                    color: FluTheme.primaryColor
                }
            }

            ColumnLayout {
                anchors.top: title_label.bottom
                anchors.left: parent.left
                anchors.right:parent.right
                anchors.bottom: parent.bottom
                anchors.margins: 2

                GridLayout {
                    Layout.fillWidth: true
                    rows: 6
                    columns: 1
 
                    FluText {
                        text: Lang.method_type
                    }

                    FluComboBox {
                        Layout.fillWidth: true
                        model: [Lang.sac_RANSAC, Lang.sac_LMEDS, Lang.sac_MSAC, Lang.sac_RRANSAC, Lang.sac_RMSAC, Lang.sac_MLESAC, Lang.sac_PROSAC]
                        currentIndex: node.methodType
                        
                        onCurrentIndexChanged: {
                            node.methodType = currentIndex;
                        }
                    }

                    FluText {
                        text: Lang.model_type
                    }

                    FluComboBox {
                        Layout.fillWidth: true
                        model: [Lang.sacMODEL_PLANE, Lang.sacMODEL_LINE, Lang.sacMODEL_CIRCLE2D, Lang.sacMODEL_CIRCLE3D, Lang.sacMODEL_SPHERE, Lang.sacMODEL_CYLINDER, Lang.sacMODEL_CONE, Lang.sacMODEL_TORUS, Lang.sacMODEL_PARALLEL_LINE, Lang.sacMODEL_PERPENDICULAR_PLANE, Lang.sacMODEL_PARALLEL_LINES, Lang.sacMODEL_NORMAL_PLANE, Lang.sacMODEL_NORMAL_SPHERE, Lang.sacMODEL_REGISTRATION, Lang.sacMODEL_REGISTRATION_2D, Lang.sacMODEL_PARALLEL_PLANE, Lang.sacMODEL_NORMAL_PARALLEL_LINES, Lang.sacMODEL_STICK, Lang.sacMODEL_ELLIPSE3D]
                        currentIndex: node.modelType

                        onCurrentIndexChanged: {
                            node.modelType = currentIndex;
                        }
                    }

                    FluText{
                        text: Lang.distance_threshold
                    }

                    FluSpinBox {
                        Layout.fillWidth: true
                        from: -999999999
                        to: 999999999
                        value: node.distanceThreshold * stepSize
                        stepSize: 100
                        editable: true
                        property int decimals: 2
                        property real realValue: value / 100
                        up.indicator: undefined
                        down.indicator: undefined
                        textFromValue: function(value, locale) {
                            return Number(value / 100).toLocaleString(locale, 'f', decimals)
                        }

                        valueFromText: function(text, locale) {
                            return Number.fromLocaleString(locale, text) * 100
                        }

                        onValueChanged: {
                            node.distanceThreshold = value / 100;
                        }
                    }
                }
            }
        }

        Qan.BottomRightResizer {
            target: cloudInputNodeItem
            handler: Rectangle {
                color: "red"
                width: 8
                height: 8
                radius: 4
            }
        }
    }
}
