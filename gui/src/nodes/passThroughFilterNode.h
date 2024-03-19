/**
 * @file passThroughFilterNode.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __PASS_THROUGH_FILTER_NODE_H_
#define __PASS_THROUGH_FILTER_NODE_H_

#include <thread>

#include "flowNode.h"

#include "typeDef.h"

class PassThroughFilterNode : public FlowNode
{
    Q_OBJECT
    Q_PROPERTY_AUTO(bool, filterX)
    Q_PROPERTY_AUTO(bool, filterY)
    Q_PROPERTY_AUTO(bool, filterZ)
    Q_PROPERTY_AUTO(double, minX)
    Q_PROPERTY_AUTO(double, maxX)
    Q_PROPERTY_AUTO(double, minY)
    Q_PROPERTY_AUTO(double, maxY)
    Q_PROPERTY_AUTO(double, minZ)
    Q_PROPERTY_AUTO(double, maxZ)
  public:
    PassThroughFilterNode() : FlowNode{FlowNode::Type::PassThroughFilter}, minX_(0), maxX_(1000), minY_(0), maxY_(1000), minZ_(0), maxZ_(1000), filterX_(true), filterY_(true), filterZ_(true) { }
    ~PassThroughFilterNode();
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  private:
    std::thread workThread_;
  protected slots:
    void inNodeOutputChanged();
};

QML_DECLARE_TYPE(PassThroughFilterNode)

#endif //!__PASS_THROUGH_FILTER_NODE_H_
