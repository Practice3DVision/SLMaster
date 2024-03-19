/**
 * @file statisticalOutlierRemovalNode.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __STATISTICAL_OUTLIER_REMOVAL_NODE_H_
#define __STATISTICAL_OUTLIER_REMOVAL_NODE_H_

#include <thread>

#include "flowNode.h"

#include "typeDef.h"

class StatisticalOutlierRemovalNode : public FlowNode
{
    Q_OBJECT
    Q_PROPERTY_AUTO(double, k)
    Q_PROPERTY_AUTO(double, stdThreshold)
  public:
    StatisticalOutlierRemovalNode() : FlowNode{FlowNode::Type::StatisticalOutlierRemoval}, k_(1500), stdThreshold_(1) { }
    ~StatisticalOutlierRemovalNode();
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  private:
    std::thread workThread_;
  protected slots:
    void inNodeOutputChanged();
};

QML_DECLARE_TYPE(StatisticalOutlierRemovalNode)

#endif //!__STATISTICAL_OUTLIER_REMOVAL_NODE_H_
