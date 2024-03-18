#include "meshOutputNode.h"

#include "VtkProcessEngine.h"

#include <pcl/PolygonMesh.h>

using namespace pcl;

MeshOutputNode::MeshOutputNode() : FlowNode{FlowNode::Type::CloudOutput} {
    //setOutput(QVariant::fromValue(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>)));
}

QQmlComponent*  MeshOutputNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent> delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/MeshOutputNode.qml");
    return delegate.get();
}

void MeshOutputNode::inNodeOutputChanged() {
    FlowNode::inNodeOutputChanged();
    
    for (const auto inNode : get_in_nodes()) {
        const auto inFlowNode = qobject_cast<FlowNode*>(inNode);
        if (inFlowNode == nullptr ||
            !inFlowNode->getOutput().isValid())
            continue;

        auto inOutput = inFlowNode->getOutput().value<PolygonMesh::Ptr>();
        if(inOutput->polygons.empty()) {
            qInfo() << "[Mesh Output Node]: In node output mesh is empty!";
        }

        VTKProcessEngine::instance()->updateProcessMesh(inOutput);
    }

    qInfo() << "[Mesh Output Node]: Completed!";
    //setOutput(QVariant::fromValue(outPutCloud));
}
