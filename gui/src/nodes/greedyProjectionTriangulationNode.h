#ifndef __GREEDY_PROJECTION_TRIANGULATION_NODE_H_
#define __GREEDY_PROJECTION_TRIANGULATION_NODE_H_

#include <thread>

#include "flowNode.h"

#include "typeDef.h"

class GreedyProjectionTriangulationNode : public FlowNode
{
    Q_OBJECT
    Q_PROPERTY_AUTO(double, kSearch)
    Q_PROPERTY_AUTO(double, multiplier)
    Q_PROPERTY_AUTO(double, maxNearestNumber)
    Q_PROPERTY_AUTO(double, searchRadius)
    Q_PROPERTY_AUTO(double, minimumAngle)
    Q_PROPERTY_AUTO(double, maximumAngle)
    Q_PROPERTY_AUTO(double, maximumSurfaceAngle)
    Q_PROPERTY_AUTO(bool, normalConsistency)
  public:
    GreedyProjectionTriangulationNode() : FlowNode{FlowNode::Type::GreedyProjectionTriangulation}, kSearch_(20), multiplier_(2.5), maxNearestNumber_(600), searchRadius_(0.2), minimumAngle_(10), maximumAngle_(120), maximumSurfaceAngle_(45), normalConsistency_(false) { }
    ~GreedyProjectionTriangulationNode();
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  private:
    std::thread workThread_;
  protected slots:
    void inNodeOutputChanged();
};

QML_DECLARE_TYPE(GreedyProjectionTriangulationNode)

#endif //!__GREEDY_PROJECTION_TRIANGULATION_NODE_H_
