import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15
import QtQuick.Window 2.15
import Qt.labs.platform 1.1

import FluentUI 1.0
import SLMasterGui 1.0

import "qrc:///ui/global"

FluContentPage{
    id:root
    title:""
    launchMode: FluPageType.SingleInstance

    property real cali_error: 0

    signal back

    function gernerateData(errorsData) {
        var date = {datasets: [{
                        label: Lang.error_point_distribution,
                        xAxisID: 'x-axis-1',
                        yAxisID: 'y-axis-1',
                        borderColor: '#ff5555',
                        backgroundColor: 'rgba(255,192,192,0.3)',
                        data: errorsData
                    }]}

        return date;
    }

    function initTreeData(){
        const dig = () => {
            const leftKey = Lang.projector_camera;
            const list = [
                {
                    title: leftKey,
                    leftKey,
                }
            ];
            return list;
        };
        return dig();
    }

    function updateImgPaths(paths, data_source_index) {
        var children_paths = [];
        for(var i = 0; i < paths.length; ++i) {
            const key = paths[i];
            children_paths.push({
                                title: key,
                                key,
                                });
        }

        var dataSource = tree_view.dataSource;

        dataSource[data_source_index].children = children_paths;
        tree_view.dataSource = dataSource;
    }

    RowLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        ColumnLayout {
            Layout.preferredWidth: 400
            Layout.preferredHeight: parent.height
            spacing: 10

            FluText {
                text: Lang.img_browser
                font: FluTextStyle.Subtitle
                horizontalAlignment: Text.AlignLeft
            }

            FluArea {
                Layout.fillWidth: true
                Layout.fillHeight: true
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10

                    FluArea {
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        FluTreeView {
                            id:tree_view
                            anchors.fill: parent
                            draggable: true
                            showLine: true

                            property string prjector_folder: ""

                            CameraModel {
                                id: projector_camera_model

                                Component.onCompleted: {
                                    CalibrateEngine.bindProjectorModel(projector_camera_model);
                                }
                            }

                            Component.onCompleted: {
                                dataSource = initTreeData();
                            }

                            onCurrentNodeChanged: {
                                if(!currentNode) {
                                    return;
                                }

                                if(currentNode.title !== Lang.projector_camera) {
                                    if(stripe_index_indicator.pageCurrent === 1) {
                                        CalibrateEngine.updateDisplayImg(currentNode.title);
                                    }
                                    else if(stripe_index_indicator.pageCurrent === 2) {
                                        var errors = CalibrateEngine.updateErrorDistribute(currentNode.title, true);
                                        var errorDistributes = [];
                                        for(var i = 0; i < errors.length; ++i) {
                                            errorDistributes[i] = {x: errors[i].x, y: errors[i].y};
                                        }

                                        viewer_component_loader.item.chartData = gernerateData(errorDistributes);
                                    }

                                    //stripe_index_indicator.pageCurrent = 1;
                                    //CalibrateEngine.updateDisplayImg(currentNode.title);
                                }
                            }

                            onItemPressed: (mouse)=>{
                                if(!currentNode) {
                                    return;
                                }

                                if(currentNode.title !== Lang.projector_camera) {
                                    if(mouse.button === Qt.RightButton) {
                                        menu.popup();
                                    }
                                }
                            }

                            FluMenu{
                                id:menu
                                Action {
                                    text: Lang.remove
                                    onTriggered: {
                                        CalibrateEngine.removeProjectImg(tree_view.currentNode.title)
                                        //updateImgPaths(projector_camera_model.imgPaths(), 0);
                                    }
                                }
                            }

                            Connections {
                                target:CalibrateEngine

                                function onProjectorModelChanged() {
                                    updateImgPaths(projector_camera_model.imgPaths(), 0);
                                }
                            }
                        }
                    }

                    GridLayout {
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        rows: 2
                        columns: 2

                        FolderDialog {
                            id: save_folder_dialog
                            title: Lang.please_choose_save_folder

                            onAccepted: {
                                CalibrateEngine.saveCaliInfo(currentFolder.toString());
                                showSuccess(Lang.save_sucess, 3000);
                            }
                        }

                        FluButton {
                            Layout.preferredWidth: parent.width / 2
                            text: Lang.capture

                            onClicked: {
                                if(CalibrateEngine.captureOnce()) {
                                    showSuccess(Lang.capture_sucess, 3000)
                                }
                                else {
                                    showError(Lang.capture_failed, 3000)
                                }
                            }
                        }

                        FluButton {
                            Layout.fillWidth: true
                            text: Lang.save

                            onClicked: {
                                save_folder_dialog.open();
                            }
                        }

                        FluButton {
                            Layout.preferredWidth: parent.width / 2
                            text: Lang.set_calibration_params

                            onClicked: {
                                FluApp.navigate("/ProjectorCaliParamsSettinsWindow");
                            }
                        }

                        FluButton {
                            Layout.fillWidth: true
                            text: Lang.calibration

                            onClicked: {
                                CalibrateEngine.calibrateProjector();
                            }
                        }
                    }
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 10

            RowLayout {
                Layout.fillWidth: true

                FluText {
                    Layout.alignment: Qt.AlignLeft
                    Layout.fillWidth: true
                    text: Lang.result_display
                    font: FluTextStyle.Subtitle
                }

                FluIconButton {
                    Layout.alignment: Qt.AlignRight
                    iconSource: FluentIcons.Back
                    iconSize: 16

                    onClicked: {
                        root.back();
                    }
                }
            }

            FluLoader {
                id: viewer_component_loader
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.topMargin: -5
                sourceComponent: paint_component
            }

            Component {
                id: chat_component
                FluChart{
                    id: chart
                    anchors.fill: parent
                    chartType: 'scatter'
                    chartData: gernerateData([])
                    chartOptions: {return {
                            maintainAspectRatio: false,
                            responsive: true,
                            hoverMode: 'nearest',
                            intersect: true,
                            title: {
                                display: true,
                                text: Lang.img_error_distribution + root.cali_error.toString()
                            },
                            scales: {
                                xAxes: [{
                                        position: 'bottom',
                                        gridLines: {
                                            zeroLineColor: 'rgba(0,0,0,1)'
                                        }
                                    }],
                                yAxes: [{
                                        type: 'linear', // only linear but allow scale type registration. This allows extensions to exist solely for log scale for instance
                                        display: true,
                                        position: 'left',
                                        id: 'y-axis-1',
                                        gridLines: {
                                            zeroLineColor: 'rgba(0,0,0,1)'
                                        }
                                    }]
                            }
                        }
                    }
                }
            }

            Component {
                id: paint_component

                GridLayout {
                    anchors.fill: parent
                    //anchors.margins: 20
                    rows: 2
                    columns: 2
                    rowSpacing: 10
                    columnSpacing: 20

                    FluArea {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        ImagePaintItem {
                            id: honrizon_paint_item
                            anchors.fill: parent
                            color: FluTheme.dark ? Window.active ?  Qt.rgba(38/255,44/255,54/255,1) : Qt.rgba(39/255,39/255,39/255,1) : Qt.rgba(251/255,251/255,253/255,1)
                        }

                        FluText {
                            Layout.alignment: Qt.AlignLeft
                            anchors.left: parent.left
                            anchors.top: parent.top
                            anchors.leftMargin: 8
                            anchors.topMargin: 8
                            opacity: 0.6
                            text: Lang.phase_h
                            font: FluTextStyle.BodyStrong
                        }
                    }

                    FluArea {
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        ImagePaintItem {
                            id: vertical_paint_item
                            anchors.fill: parent
                            color: FluTheme.dark ? Window.active ?  Qt.rgba(38/255,44/255,54/255,1) : Qt.rgba(39/255,39/255,39/255,1) : Qt.rgba(251/255,251/255,253/255,1)
                        }

                        FluText {
                            Layout.alignment: Qt.AlignLeft
                            anchors.left: parent.left
                            anchors.top: parent.top
                            anchors.leftMargin: 8
                            anchors.topMargin: 8
                            opacity: 0.6
                            text: Lang.phase_v
                            font: FluTextStyle.BodyStrong
                        }
                    }

                    FluArea {
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        ImagePaintItem {
                            id: color_paint_item
                            anchors.fill: parent
                            color: FluTheme.dark ? Window.active ?  Qt.rgba(38/255,44/255,54/255,1) : Qt.rgba(39/255,39/255,39/255,1) : Qt.rgba(251/255,251/255,253/255,1)
                        }

                        FluText {
                            Layout.alignment: Qt.AlignLeft
                            anchors.left: parent.left
                            anchors.top: parent.top
                            anchors.leftMargin: 8
                            anchors.topMargin: 8
                            opacity: 0.6
                            text: Lang.cam_img
                            font: FluTextStyle.BodyStrong
                        }
                    }

                    FluArea {
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        ImagePaintItem {
                            id: projector_paint_item
                            anchors.fill: parent
                            color: FluTheme.dark ? Window.active ?  Qt.rgba(38/255,44/255,54/255,1) : Qt.rgba(39/255,39/255,39/255,1) : Qt.rgba(251/255,251/255,253/255,1)
                        }

                        FluText {
                            Layout.alignment: Qt.AlignLeft
                            anchors.left: parent.left
                            anchors.top: parent.top
                            anchors.leftMargin: 8
                            anchors.topMargin: 8
                            opacity: 0.6
                            text: Lang.projector_img
                            font: FluTextStyle.BodyStrong
                        }
                    }

                    Component.onCompleted: {
                        CalibrateEngine.bindOnlineProjectorPaintItem(color_paint_item, honrizon_paint_item, vertical_paint_item, projector_paint_item);
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                FluPagination{
                    id: stripe_index_indicator
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignLeft
                    previousText: Lang.previous_text
                    nextText: Lang.next_text
                    pageCurrent: 1
                    itemCount: 20
                    pageButtonCount: 10

                    onPageCurrentChanged: {
                        if(pageCurrent == 1) {
                            viewer_component_loader.sourceComponent = paint_component;
                        }
                        else if(pageCurrent == 2) {
                            viewer_component_loader.sourceComponent = chat_component;
                        }

                        tree_view.currentNodeChanged();
                    }
                }

                FluProgressBar{
                    id: progress_bar
                    Layout.preferredWidth: 300
                    Layout.rightMargin: 40
                    Layout.alignment: Qt.AlignRight
                    indeterminate: false
                    progressVisible: true
                    from: 0
                    to: 100

                    property bool isOnLoading: false

                    Connections {
                        target: CalibrateEngine

                        function onProgressChanged(progress) {
                            progress_bar.value = progress;
                        }
                    }

                    onValueChanged: {
                        if(value != 0 || value != 100 && !isOnLoading) {
                            showLoading(Lang.calibration_ing, false);
                            isOnLoading = true;
                        }

                        if(value == 100 && isOnLoading) {
                            hideLoading();
                            isOnLoading = false;
                            tree_view.currentNodeChanged();
                        }
                    }
                }
            }
        }
    }

    Connections {
        target: CalibrateEngine
        function onProjectErrorReturn(error) {
            showSuccess(Lang.calibration_error + error.toString(), 10000)
            root.cali_error = error;
        }
    }

    Component.onCompleted: {
        CalibrateEngine.setCurCaliType(AppType.Projector);
        FluApp.navigate("/ProjectorCaliParamsSettinsWindow");
    }
}
