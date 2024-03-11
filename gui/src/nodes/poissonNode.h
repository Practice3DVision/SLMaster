#ifndef __POISSON_NODE_H_
#define __POISSON_NODE_H_

#include <thread>

#include "flowNode.h"

#include "typeDef.h"

class PoissonNode : public FlowNode
{
    Q_OBJECT
    Q_PROPERTY_AUTO(double, kSearch)
    Q_PROPERTY_AUTO(double, minDepth)
    Q_PROPERTY_AUTO(double, maxDepth)
    Q_PROPERTY_AUTO(double, scale)
    Q_PROPERTY_AUTO(double, solverDivide)
    Q_PROPERTY_AUTO(double, isoDivide)
    Q_PROPERTY_AUTO(double, minSamplePoints)
    Q_PROPERTY_AUTO(double, degree)
    Q_PROPERTY_AUTO(bool, confidence)
    Q_PROPERTY_AUTO(bool, manifold)
  public:
    PoissonNode() : FlowNode{FlowNode::Type::Poisson}, kSearch_(20), minDepth_(1), maxDepth_(8), scale_(1.25), solverDivide_(8), isoDivide_(8), minSamplePoints_(3), degree_(2), confidence_(false), manifold_(false) { }
    ~PoissonNode();
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  private:
    std::thread workThread_;
  protected slots:
    void inNodeOutputChanged();
};

QML_DECLARE_TYPE(PoissonNode)

#endif //!__POISSON_NODE_H_
