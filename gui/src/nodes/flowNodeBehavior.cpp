#include "flowNodeBehavior.h"

#include "flowNode.h"

void FlowNodeBehaviour::inNodeInserted( qan::Node& inNode, qan::Edge& edge ) noexcept
{
    Q_UNUSED(edge);
    const auto inFlowNode = qobject_cast<FlowNode*>(&inNode);
    const auto flowNodeHost = qobject_cast<FlowNode*>(getHost());
    if ( inFlowNode != nullptr &&
        flowNodeHost != nullptr ) {
        //
        QObject::connect(inFlowNode,    &FlowNode::outputChanged,
                         flowNodeHost,  &FlowNode::inNodeOutputChanged);
    }
    flowNodeHost->inNodeOutputChanged();    // Force a call since with a new edge insertion, actual value might aready be initialized
}

void FlowNodeBehaviour::inNodeRemoved( qan::Node& inNode, qan::Edge& edge ) noexcept
{
    Q_UNUSED(inNode); Q_UNUSED(edge);
}
