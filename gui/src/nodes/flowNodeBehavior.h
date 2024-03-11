#ifndef __FLOW_NODE_BEHAVIOR_H_
#define __FLOW_NODE_BEHAVIOR_H_

#include "QuickQanava.h"

class FlowNodeBehaviour : public qan::NodeBehaviour
{
    Q_OBJECT
  public:
    explicit FlowNodeBehaviour(QObject* parent = nullptr) : qan::NodeBehaviour{ "FlowNodeBehaviour", parent } { /* Nil */ }
  protected:
    virtual void    inNodeInserted( qan::Node& inNode, qan::Edge& edge ) noexcept override;
    virtual void    inNodeRemoved( qan::Node& inNode, qan::Edge& edge ) noexcept override;
};

#endif //!__FLOW_NODE_BEHAVIOR_H_
