/**
 * @file flowNodeBehavior.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

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
