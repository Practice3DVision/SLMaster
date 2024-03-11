#include "actorOutputNode.h"

#include "VtkProcessEngine.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using namespace pcl;

ActorOutputNode::ActorOutputNode() : FlowNode{FlowNode::Type::ActorOutput} { }

QQmlComponent* ActorOutputNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent> delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/ActorOutputNode.qml");
    return delegate.get();
}

void ActorOutputNode::inNodeOutputChanged() {
    FlowNode::inNodeOutputChanged();

    for (const auto inNode : get_in_nodes()) {
        const auto inFlowNode = qobject_cast<FlowNode*>(inNode);
        if (inFlowNode == nullptr ||
            !inFlowNode->getOutput().isValid())
            continue;

        auto inOutput = inFlowNode->getOutput().value<vtkSmartPointer< vtkActor>>();
        if(!inOutput) {
            qInfo() << "[Actor Output Node]: actor output node is nullptr!";
        }

        VTKProcessEngine::instance()->updateProcessActor(inOutput);
    }

    qInfo() << "[Actor Output Node]: Completed!";
}