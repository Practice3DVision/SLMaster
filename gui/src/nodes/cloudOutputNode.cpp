#include "cloudOutputNode.h"

#include "VtkProcessEngine.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using namespace pcl;

CloudOutputNode::CloudOutputNode() : FlowNode{FlowNode::Type::CloudOutput} {
    //setOutput(QVariant::fromValue(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>)));
}

QQmlComponent*  CloudOutputNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent> delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/CloudOutputNode.qml");
    return delegate.get();
}

void CloudOutputNode::inNodeOutputChanged() {
    FlowNode::inNodeOutputChanged();

    for (const auto inNode : get_in_nodes()) {
        const auto inFlowNode = qobject_cast<FlowNode*>(inNode);
        if (inFlowNode == nullptr ||
            !inFlowNode->getOutput().isValid())
            continue;

        auto inOutput = inFlowNode->getOutput().value<PointCloud<PointXYZRGB>::Ptr>();
        if(inOutput->empty()) {
            qInfo() << "[Cloud Output Node]: cloud is empty!";
        }

        VTKProcessEngine::instance()->updateProcessCloud(inOutput);
    }

    qInfo() << "[Cloud Output Node]: Completed!";
    //setOutput(QVariant::fromValue(outPutCloud));
}
