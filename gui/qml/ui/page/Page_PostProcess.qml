import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15
import QtQuick.Window 2.15
import Qt.labs.platform 1.1

import FluentUI 1.0
import SLMasterGui 1.0
import QuickQanava 2.0 as Qan
import "qrc:/../../../../../QuickQanava" as Qan

import "qrc:///ui/global"

FluContentPage {
    id: root
    title: ""
    launchMode: FluPageType.SingleInstance

    anchors.topMargin: 40
    anchors.top: parent.top
    anchors.bottom: parent.bottom
    anchors.left: parent.left
    anchors.right: parent.right
    leftPadding: 0
    rightPadding: 0
    bottomPadding: 0

    property bool linkCamera: true
    property var sourceNode

    Connections {
        target: CameraEngine

        function onFrameCaptured() {
            if(linkCamera) {
                sourceNode.updateSource();
            }
        }
    }

    Rectangle {
        id: backgroundRec
        anchors.fill: parent
        color: FluTheme.windowBackgroundColor

        FluMenu {
            id: menu
            property var operate_node
            FluMenuItem {
                text: "delete"

                onClicked: {
                    if(topology.selected_element_type === 0) {
                        topology.removeNode(topology.selected_element)
                    }
                    else if(topology.selected_element_type === 1) {
                        topology.removeEdge(topology.selected_element)
                    }
                    else if(topology.selected_element_type === 2) {
                        topology.removeGroup(topology.selected_element)
                    }
                }
            }
        }

        Qan.GraphView {
          id: graphView
          anchors.fill: parent
          navigable   : true
          resizeHandlerColor: FluTheme.windowActiveBackgroundColor
          gridThickColor: FluTheme.darkMode ? Qt.rgba(0.3, 0.3, 0.3, 0.5) : Qt.rgba(0.1, 0.1, 0.1, 0.5)
            /*
          MouseArea {
              anchors.fill: parent
              propagateComposedEvents: true
              focus: true

              Keys.onPressed: {
                  console.debug("dsafasdfsdafasdfdfsed")
              }
          }
            */
            function centerItem(item) {
                if (!item)
                    return
                var windowCenter = Qt.point((backgroundRec.width - item.width) / 2.,
                    (backgroundRec.height - item.height) / 2.)
                var graphNodeCenter = backgroundRec.mapToItem(containerItem, windowCenter.x, windowCenter.y)
                item.x = graphNodeCenter.x
                item.y = graphNodeCenter.y
            }

          graph: FlowGraph {
              id: topology

              property var selected_element
              property var selected_element_type //0: node, 1: edge, 2:group

              connectorEnabled: true
              connectorEdgeColor: "lightblue"
              connectorColor: FluTheme.primaryColor
              selectionColor: FluTheme.primaryColor

              portDelegate: Component {
                  Qan.PortItem {
                      width: 16;
                      height: 16
                      Rectangle {
                          anchors.fill: parent
                          color: "green"
                          border.color: "yellow"
                          border.width: 2
                      }
                  }
              }

               Component.onCompleted: {
                   defaultEdgeStyle.lineType = Qan.EdgeStyle.Curved
                   defaultEdgeStyle.lineColor = Qt.rgba(224/255,78/255,97/255,1)
               }

               onNodeRightClicked: (node, pos) =>{
                   topology.selected_element = node;
                   topology.selected_element_type = 0;
                   var itemPos = backgroundRec.mapFromItem(node.item, pos)
                   menu.x = itemPos.x;
                   menu.y = itemPos.y;
                   menu.open();
               }

               onEdgeRightClicked: (edge, pos) =>{
                   topology.selected_element = edge;
                   topology.selected_element_type = 1;
                   var itemPos = backgroundRec.mapFromItem(edge.item, pos)
                   menu.x = itemPos.x;
                   menu.y = itemPos.y;
                   menu.open();
               }

               onGroupRightClicked: (group, pos) =>{
                   topology.selected_element = group;
                   topology.selected_element_type = 2;
                   var itemPos = backgroundRec.mapFromItem(group.item, pos)
                   menu.x = itemPos.x;
                   menu.y = itemPos.y;
                   menu.open();
               }
            }

          FluExpander {
              id: setting_expander
              anchors.left: parent.left
              anchors.top: parent.top
              anchors.leftMargin: 20
              anchors.topMargin: 20
              width: settings_area.width
              headerText: Lang.nodes
              contentHeight: settings_area.height + 10

              Item {
                  Flickable{
                      anchors.fill: parent
                      contentWidth: settings_area.width + 5
                      contentHeight: settings_area.height
                      FluArea {
                          id: settings_area
                          width: root.width / 6
                          height: column.height
                          anchors.left: parent.left
                          anchors.top: parent.top
                          anchors.leftMargin: 5
                          anchors.topMargin: 5

                          ColumnLayout {
                              id: column
                              FluText {
                                  text: Lang.inoutput
                              }

                              GridLayout {
                                //rows: 8
                                columns: 4

                                FluIconButton {
                                    iconSource: FluentIcons.Cloud
                                    iconSize: 28
                                    text: Lang.cloudInputNode

                                    onClicked: {
                                        sourceNode = topology.insertFlowNode(FlowNode.CloudInput);
                                        graphView.centerItem(sourceNode.item);
                                    }
                                }

                                FluIconButton {
                                    iconSource: FluentIcons.SignOut
                                    iconSize: 28
                                    text: Lang.cloudOutputNode

                                    onClicked: {
                                        var node = topology.insertFlowNode(FlowNode.CloudOutput);
                                        graphView.centerItem(node.item);
                                    }
                                }

                                FluIconButton {
                                    iconSource: FluentIcons.SurfaceHub
                                    iconSize: 28
                                    text: Lang.meshOutputNode

                                    onClicked: {
                                        var node = topology.insertFlowNode(FlowNode.MeshOutput);
                                        graphView.centerItem(node.item);
                                    }
                                }

                                FluIconButton {
                                    iconSource: FluentIcons.Group
                                    iconSize: 28
                                    text: Lang.group

                                    onClicked: {
                                        var gg = topology.insertGroup();
                                        graphView.centerItem(gg.item);
                                    }
                                }

                                FluIconButton {
                                    iconSource: FluentIcons.Split20
                                    iconSize: 28
                                    text: Lang.split_output_node

                                    onClicked: {
                                        var node = topology.insertFlowNode(FlowNode.SplitOutput);
                                        graphView.centerItem(node.item);
                                    }
                                }

                                FluIconButton {
                                    iconSource: FluentIcons.Movies
                                    iconSize: 28
                                    text: Lang.actor_output_node

                                    onClicked: {
                                        var node = topology.insertFlowNode(FlowNode.ActorOutput);
                                        graphView.centerItem(node.item);
                                    }
                                }
                              }

                              FluText {
                                  text: Lang.filters
                              }

                              GridLayout {
                                  FluIconButton {
                                      iconSource: FluentIcons.Filter
                                      iconSize: 28
                                      text: Lang.passThroughFilterNode

                                      onClicked: {
                                          var node = topology.insertFlowNode(FlowNode.PassThroughFilter);
                                          graphView.centerItem(node.item);
                                      }
                                  }

                                  FluIconButton {
                                      iconSource: FluentIcons.AreaChart
                                      iconSize: 28
                                      text: Lang.staticRemovel

                                      onClicked: {
                                          var node = topology.insertFlowNode(FlowNode.StatisticalOutlierRemoval);
                                          graphView.centerItem(node.item);
                                      }
                                  }
                              }

                              FluText {
                                  text: Lang.registration
                              }

                              FluText {
                                  text: Lang.sample_consensus
                              }

                              FluText {
                                  text: Lang.segmentation
                              }

                              GridLayout {
                                  FluIconButton {
                                      iconSource: FluentIcons.Edit
                                      iconSize: 28
                                      text: Lang.sac_segment

                                      onClicked: {
                                          var node = topology.insertFlowNode(FlowNode.SACSegment);
                                          graphView.centerItem(node.item);
                                      }
                                  }

                                  FluIconButton {
                                      iconSource: FluentIcons.HWPNewLine
                                      iconSize: 28
                                      text: Lang.intersection_line_node

                                      onClicked: {
                                          var node = topology.insertFlowNode(FlowNode.IntersectionLine);
                                          graphView.centerItem(node.item);
                                      }
                                  }

                                  FluIconButton {
                                      iconSource: FluentIcons.Walk
                                      iconSize: 28
                                      text: Lang.three_line_intersection_node

                                      onClicked: {
                                          var node = topology.insertFlowNode(FlowNode.ThreeLineIntersection);
                                          graphView.centerItem(node.item);
                                      }
                                  }
                              }

                              FluText {
                                  text: Lang.surface
                              }

                              GridLayout {
                                  FluIconButton {
                                      iconSource: FluentIcons.StatusTriangleInner
                                      iconSize: 28
                                      text: Lang.greedyProjectionTriangulation

                                      onClicked: {
                                          var node = topology.insertFlowNode(FlowNode.GreedyProjectionTriangulation);
                                          graphView.centerItem(node.item);
                                      }
                                  }

                                  FluIconButton {
                                      iconSource: FluentIcons.Emoji
                                      iconSize: 28
                                      text: Lang.poisson

                                      onClicked: {
                                          var node = topology.insertFlowNode(FlowNode.Poisson);
                                          graphView.centerItem(node.item);
                                      }
                                  }
                              }

                              FluText {
                                  text: Lang.features
                              }
                          }
                      }
                  }
              }
          }
        }

        Qan.GraphPreview {
            id: graphPreview
            source: graphView
            viewWindowColor: FluTheme.primaryColor
            anchors.right: graphView.right; anchors.bottom: graphView.bottom
            anchors.rightMargin: 8; anchors.bottomMargin: 8
            width: previewMenu.mediumPreview.width
            height: previewMenu.mediumPreview.height
            FluMenu {
                id: previewMenu
                readonly property size smallPreview: Qt.size(150, 85)
                readonly property size mediumPreview: Qt.size(250, 141)
                readonly property size largePreview: Qt.size(350, 198)
                FluMenuItem {
                    text: "Hide preview"
                    onTriggered: graphPreview.visible = false
                }
                FluMenuSeparator { }
                FluMenuItem {
                    text: qsTr('Small')
                    checkable: true
                    checked: graphPreview.width === previewMenu.smallPreview.width &&
                             graphPreview.height === previewMenu.smallPreview.height
                    onTriggered: {
                        graphPreview.width = previewMenu.smallPreview.width
                        graphPreview.height = previewMenu.smallPreview.height
                    }
                }
                FluMenuItem {
                    text: qsTr('Medium')
                    checkable: true
                    checked: graphPreview.width === previewMenu.mediumPreview.width &&
                             graphPreview.height === previewMenu.mediumPreview.height
                    onTriggered: {
                        graphPreview.width = previewMenu.mediumPreview.width
                        graphPreview.height = previewMenu.mediumPreview.height
                    }
                }
                FluMenuItem {
                    text: qsTr('Large')
                    checkable: true
                    checked: graphPreview.width === previewMenu.largePreview.width &&
                             graphPreview.height === previewMenu.largePreview.height
                    onTriggered: {
                        graphPreview.width = previewMenu.largePreview.width
                        graphPreview.height = previewMenu.largePreview.height
                    }
                }
            }
            MouseArea {
                anchors.fill: parent
                acceptedButtons: Qt.RightButton
                onClicked: previewMenu.open(mouse.x, mouse.y)
            }
        }
    }

    FluArea {
        id: operation_area
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.rightMargin: 20
        anchors.topMargin: 20
        width: 48
        height: operation_layout.height

        ColumnLayout {
            id: operation_layout
            width: parent.width
            anchors.left: parent.left
            anchors.top: parent.top

            FluIconButton {
                Layout.alignment: Qt.AlignHCenter
                iconSource: FluentIcons.Play
                iconSize: 28

                onClicked: {
                    sourceNode.updateSource();
                }
            }

            FluIconButton {
                Layout.alignment: Qt.AlignHCenter
                iconSource: FluentIcons.Save
                iconSize: 28

                onClicked: {
                    saveFolderDialog.open();
                }
            }

            FluIconButton {
                Layout.alignment: Qt.AlignHCenter
                iconSource: FluentIcons.OpenFile
                iconSize: 28

                onClicked: {
                    saveFolderDialog.open();
                }
            }

            FluIconButton {
                Layout.alignment: Qt.AlignHCenter
                iconSource: FluentIcons.Link
                iconSize: 28
                color: linkCamera ? FluTheme.primaryColor : FluTheme.itemNormalColor

                onClicked: {
                    linkCamera = !linkCamera;
                }
            }
        }
    }
}



