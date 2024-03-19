/**
 * @file flowGraph.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __FLOW_GRAPH_H_
#define __FLOW_GRAPH_H_

#include "flowNode.h"

class FlowGraph : public qan::Graph
{
    Q_OBJECT
  public:
    explicit FlowGraph( QQuickItem* parent = nullptr ) noexcept : qan::Graph(parent) { }
  public:
    Q_INVOKABLE qan::Node*  insertFlowNode(int type) { return insertFlowNode(static_cast<FlowNode::Type>(type)); }       // FlowNode::Type could not be used from QML, Qt 5.10 bug???
    qan::Node*              insertFlowNode(FlowNode::Type type);
};

QML_DECLARE_TYPE(FlowGraph)

#endif //!__FLOW_GRAPH_H_
