#include "SplitOutputNode.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace pcl;

QQmlComponent *SplitOutputNode::delegate(QQmlEngine &engine) noexcept {
    static std::unique_ptr<QQmlComponent> delegate;
    if (!delegate)
        delegate = std::make_unique<QQmlComponent>(
            &engine, "qrc:/ui/nodes/SplitOutputNode.qml");
    return delegate.get();
}

void SplitOutputNode::inNodeOutputChanged() {
    FlowNode::inNodeOutputChanged();

    const auto inNode = get_in_nodes().at(0);
    const auto inFlowNode = qobject_cast<FlowNode*>(inNode);

    if (inFlowNode == nullptr || !inFlowNode->getOutput().isValid())
        return;

    auto inOutput = inFlowNode->getOutput().value<std::vector<PointCloud<PointXYZRGB>::Ptr>>();

    if (inOutput.empty()) {
        qInfo() << "[Split Output Node]: In node output cloud is empty!";
    }
    else {
        setOutput(QVariant::fromValue(inOutput[portIndex_]));
    }

    qInfo() << "[Split Output Node]: Completed!";
}
