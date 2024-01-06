pragma Singleton

import QtQuick 2.15
import SLMasterGui 1.0

QtObject {
    signal startScan

    property var render_items: []

    property int scan_mode: 0

    property var camera_properties : {
        "Contrast Threshold": 0,
        "Cost Min Diff": 1,
        "Cost Max Diff": 2,
        "Max Cost": 3,
        "Min Disparity": 4,
        "Max Disparity": 5,
        "Light Strength": 6,
        "Exposure Time": 7,
        "Pre Exposure Time": 8,
        "Aft Exposure Time": 9,
        "Phase Shift Times": 10,
        "Cycles": 11,
        "Total Fringes": 12,
        "Pattern": 13,
        "Minimum Depth": 14,
        "Maximum Depth": 15,
        "Enable Depth Camera": 16,
        "Gpu Accelerate": 17,
        "Is Vertical": 18,
        "Noise Filter": 19,
    }
}
