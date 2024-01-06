import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15

import FluentUI 1.0
import SLMasterGui 1.0

import "qrc:/ui/global"
import "../global"

Item {
    id: item
    signal functionChanged(var page_type)

    property int curPage: AppType.Device

    function updateCheckedState() {
        for(var i = 0; i < layout.data.length; ++i) {
            if(i !== item.curPage) {
                layout.data[i].setChecked(false);
            }
            else {
                layout.data[i].setChecked(true);
            }
        }
    }

    RowLayout {
        id: layout
        anchors.fill: parent
        spacing: 10
        z: 999

        TimePoint {
            id: device_point
            functionName: Lang.device
            statusText: Lang.device

            onCheckStateChanged: {
                item.curPage = AppType.Device;
                updateCheckedState();
            }
        }

        TimePoint {
            id: calibration_point
            functionName: Lang.calibration
            statusText: Lang.calibration

            onCheckStateChanged: {
                item.curPage = AppType.Calibration;
                updateCheckedState();
            }
        }

        TimePoint {
            id: scan_mode_point
            functionName: Lang.scan_mode
            statusText: Lang.scan_mode

            onCheckStateChanged: {
                item.curPage = AppType.ScanMode;
                updateCheckedState();
            }
        }

        TimePoint {
            id: scan_point
            functionName: Lang.scan
            statusText: Lang.scan

            onCheckStateChanged: {
                item.curPage = AppType.Scan;
                updateCheckedState();
            }
        }

        TimePoint {
            id: post_process_point
            functionName: Lang.post_process
            statusText: Lang.post_process

            onCheckStateChanged: {
                item.curPage = AppType.PostProcess;
                updateCheckedState();
            }
        }

        TimePoint {
            id: measurement_point
            functionName: Lang.measurement
            statusText: Lang.measurement

            onCheckStateChanged: {
                item.curPage = AppType.Measurement;
                updateCheckedState();
            }
        }

        Component.onCompleted: {
            device_point.setChecked(item.curPage === AppType.Device);
            calibration_point.setChecked(item.curPage === AppType.Calibration);
            scan_mode_point.setChecked(item.curPage === AppType.ScanMode);
            scan_point.setChecked(item.curPage === AppType.Scan);
            post_process_point.setChecked(item.curPage === AppType.PostProcess);
            measurement_point.setChecked(item.curPage === AppType.Measurement);
        }
    }

    Rectangle {
        anchors.top: layout.bottom
        width: parent.width
        anchors.topMargin: -3
        height: 6
        color: FluTheme.dark ? Qt.rgba(1,1,1,1) : Qt.rgba(50/255,50/255,50/255,1)
    }
}
