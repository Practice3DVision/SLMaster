#include "flowGraph.h"

#include "cloudInputNode.h"
#include "cloudOutputNode.h"
#include "meshOutputNode.h"
#include "passThroughFilterNode.h"
#include "flowNodeBehavior.h"
#include "statisticalOutlierRemovalNode.h"
#include "greedyProjectionTriangulationNode.h"
#include "poissonNode.h"
#include "sacSegmentNode.h"
#include "splitOutputNode.h"
#include "intersectionLineNode.h"
#include "threeLineIntersectionNode.h"
#include "actorOutputNode.h"

qan::Node* FlowGraph::insertFlowNode(FlowNode::Type type) {
    qan::Node* flowNode = nullptr;
    switch ( type ) {
        case FlowNode::Type::CloudInput:
            flowNode = insertNode<CloudInputNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
            break;
        case FlowNode::Type::CloudOutput:
            flowNode = insertNode<CloudOutputNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "In", "In" );
            break;
        case FlowNode::Type::PassThroughFilter:
            flowNode = insertNode<PassThroughFilterNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN", "IN" );
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
            break;
        case FlowNode::Type::StatisticalOutlierRemoval:
            flowNode = insertNode<StatisticalOutlierRemovalNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN", "IN" );
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
            break;
        case FlowNode::Type::GreedyProjectionTriangulation:
            flowNode = insertNode<GreedyProjectionTriangulationNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN", "IN" );
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
            break;
        case FlowNode::Type::Poisson:
            flowNode = insertNode<PoissonNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN", "IN" );
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
            break;
        case FlowNode::Type::MeshOutput:
            flowNode = insertNode<MeshOutputNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN", "IN" );
            break;
        case FlowNode::Type::SACSegment:
            flowNode = insertNode<SACSegmentNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN", "IN" );
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT1", "OUT1" );
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT2", "OUT2" );
            break;
        case FlowNode::Type::SplitOutput:
            flowNode = insertNode<SplitOutputNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN", "IN" );
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
            break;
        case FlowNode::Type::IntersectionLine:
            flowNode = insertNode<IntersectionLineNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN1", "IN1" );
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN2", "IN2" );
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
            break;
        case FlowNode::Type::ThreeLineIntersection:
            flowNode = insertNode<ThreeLineIntersectionLineNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN1", "IN1" );
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN2", "IN2" );
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN3", "IN3" );
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
            break;
        case FlowNode::Type::ActorOutput:
            flowNode = insertNode<ActorOutputNode>(nullptr);
            insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN", "IN" );
            insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
            break;
    }
    if (flowNode)
        flowNode->installBehaviour(std::make_unique<FlowNodeBehaviour>());
    return flowNode;
}
