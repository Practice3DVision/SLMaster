#ifndef __THREE_LINE_INTERSECTION_LINE_NODE_H_
#define __THREE_LINE_INTERSECTION_LINE_NODE_H_

#include <thread>

#include "flowNode.h"

#include "typeDef.h"

class ThreeLineIntersectionLineNode : public FlowNode
{
    Q_OBJECT
    Q_PROPERTY_AUTO(int, width)
    Q_PROPERTY_AUTO(int, height)
    Q_PROPERTY_AUTO(int, length)
  public:
    ThreeLineIntersectionLineNode() : FlowNode{FlowNode::Type::Poisson}, width_(167), height_(50), length_(167) { }
    ~ThreeLineIntersectionLineNode();
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  private:
    std::thread workThread_;
  protected slots:
    void inNodeOutputChanged();
};

QML_DECLARE_TYPE(ThreeLineIntersectionLineNode)

#endif //!__THREE_LINE_INTERSECTION_LINE_NODE_H_