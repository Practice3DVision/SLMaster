#include "flowNode.h"

QQmlComponent*  FlowNode::delegate(QQmlEngine& engine) noexcept {
    static std::unique_ptr<QQmlComponent>   qan_FlowNode_delegate;
    if ( !qan_FlowNode_delegate )
        qan_FlowNode_delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/FlowNode.qml");
    return qan_FlowNode_delegate.get();
}

void FlowNode::inNodeOutputChanged(){ }

void FlowNode::setOutput(QVariant output) noexcept {
    _output = output;
    emit outputChanged();
}
