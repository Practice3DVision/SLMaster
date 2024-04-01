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
            const leftKey = Lang.left_camera;
            const rightKey = Lang.right_camera;
            const list = [
                {
                    title: leftKey,
                    leftKey,
                },
                {
                    title: rightKey,
                    rightKey,
                },
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

                            property string left_camera_folder: ""
                            property string right_camera_folder: ""

                            CameraModel {
                                id: left_camera_model

                                Component.onCompleted: {
                                    CalibrateEngine.bindLeftCameraModel(left_camera_model);
                                }
                            }

                            CameraModel {
                                id: right_camera_model

                                Component.onCompleted: {
                                    CalibrateEngine.bindRightCameraModel(right_camera_model);
                                }
                            }

                            Component.onCompleted: {
                                dataSource = initTreeData();
                            }

                            onCurrentNodeChanged: {
                                if(!currentNode) {
                                    return;
                                }

                                if(currentNode.title !== "Left Camera" && currentNode.title !== "Right Camera") {
                                    if(stripe_index_indicator.pageCurrent === 1) {
                                        var path = currentNode.parent.title === "Left Camera" ? left_camera_model.curFolderPath() + "/" + currentNode.title : right_camera_model.curFolderPath() + "/" + currentNode.title;
                                        CalibrateEngine.updateDisplayImg(path);
                                    }
                                    else if(stripe_index_indicator.pageCurrent === 2) {
                                        var errors = CalibrateEngine.updateErrorDistribute(currentNode.title, currentNode.parent.title === "Left Camera");
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

                                if(currentNode.title !== "Left Camera" && currentNode.title !== "Right Camera") {
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
                                        if(tree_view.currentNode.parent.title === "Left Camera") {
                                            CalibrateEngine.removeCameraImg(tree_view.currentNode.title, true);
                                            left_camera_model.erase(tree_view.currentNode.title);
                                            updateImgPaths(left_camera_model.imgPaths(), 0);
                                        }
                                        else {
                                            CalibrateEngine.removeCameraImg(tree_view.currentNode.title, false);
                                            right_camera_model.erase(tree_view.currentNode.title);
                                            updateImgPaths(right_camera_model.imgPaths(), 1);
                                        }
                                    }
                                }

                                Action {
                                    text: Lang.sequenceInverse
                                    onTriggered: {
                                        CalibrateEngine.invFeatureSequence(tree_view.currentNode.title, tree_view.currentNode.parent.title === "Left Camera");
                                    }
                                }

                                Action {
                                    text: Lang.horizonVerticalInv
                                    onTriggered: {
                                        CalibrateEngine.invFeatureHVSequence(tree_view.currentNode.title, tree_view.currentNode.parent.title === "Left Camera");
                                    }
                                }
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true

                        FluButton {
                            id: select_left_folder_btn
                            Layout.fillWidth: true
                            text: Lang.select_left_folder

                            onClicked: {
                                folder_dialog.camera_folder_type = AppType.Left;
                                folder_dialog.open();
                            }
                        }

                        FluButton {
                            id: select_right_folder_btn
                            Layout.fillWidth: true
                            text: Lang.select_right_folder

                            onClicked: {
                                folder_dialog.camera_folder_type = AppType.Right;
                                folder_dialog.open();
                            }
                        }

                        FolderDialog {
                            id: folder_dialog
                            title: camera_folder_type === AppType.Left ? Lang.please_select_left_folder : camera_folder_type === AppType.Right ? Lang.please_select_right_folder : Lang.please_select_color_folder

                            property int camera_folder_type: AppType.Left

                            onAccepted: {
                                if(camera_folder_type === AppType.Left) {
                                    left_camera_model.recurseImg(currentFolder.toString());
                                    updateImgPaths(left_camera_model.imgPaths(), 0);
                                    tree_view.left_camera_folder = currentFolder.toString();
                                }
                                else if(camera_folder_type === AppType.Right) {
                                    right_camera_model.recurseImg(currentFolder.toString());
                                    updateImgPaths(right_camera_model.imgPaths(), 1);
                                    tree_view.right_camera_folder = currentFolder.toString();
                                }
                            }
                        }
                    }


                    RowLayout {
                        Layout.fillWidth: true

                        FolderDialog {
                            id: save_folder_dialog
                            title: Lang.please_choose_save_folder

                            onAccepted: {
                                CalibrateEngine.saveCaliInfo(currentFolder.toString());
                                showSuccess(Lang.save_sucess, 3000);
                            }
                        }

                        FluButton {
                            Layout.preferredWidth: parent.width / 3
                            text: Lang.calibration

                            onClicked: {
                                FluApp.navigate("/CalibrationSettinsWindow");
                            }
                        }

                        FluButton {
                            Layout.preferredWidth: parent.width / 3
                            text: Lang.save

                            onClicked: {
                                save_folder_dialog.open();
                            }
                        }

                        FluButton {
                            Layout.fillWidth: true
                            text: Lang.display_rectify_img

                            onClicked: {
                                stripe_index_indicator.pageCurrent = 1;
                                CalibrateEngine.displayStereoRectifyMap();
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

            FluArea {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.topMargin: -5
                clip: true

                FluLoader {
                    id: viewer_component_loader
                    anchors.fill: parent
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
                    ImagePaintItem {
                        id: img_paint_item
                        anchors.fill: parent
                        color: FluTheme.dark ? Window.active ?  Qt.rgba(38/255,44/255,54/255,1) : Qt.rgba(39/255,39/255,39/255,1) : Qt.rgba(251/255,251/255,253/255,1)

                        Component.onCompleted: {
                            CalibrateEngine.bindOfflineCamPaintItem(img_paint_item);
                        }
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
        function onErrorReturn(error) {
            showSuccess(Lang.calibration_error + error.toString(), 10000);
            root.cali_error = error;
        }

        function onFindFeaturePointsError(path) {
            showError((Lang.find_features_error).arg(path), 10000);
            hideLoading();
        }
    }

    Component.onCompleted: {
        CalibrateEngine.setCurCaliType(AppType.Stereo);
    }
}
