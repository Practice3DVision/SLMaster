#ifndef __INTERSECTION_LINE_NODE_H_
#define __INTERSECTION_LINE_NODE_H_

#include <thread>

#include "flowNode.h"

#include "typeDef.h"

class IntersectionLineNode : public FlowNode
{
    Q_OBJECT
  public:
    IntersectionLineNode() : FlowNode{FlowNode::Type::Poisson} { }
    ~IntersectionLineNode();
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  private:
    std::thread workThread_;
  protected slots:
    void inNodeOutputChanged();
};

QML_DECLARE_TYPE(IntersectionLineNode)

#endif //!__INTERSECTION_LINE_NODE_H_