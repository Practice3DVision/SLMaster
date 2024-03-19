/**
 * @file sacSegmentNode.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __SAC_SEGMENT_NODE_H_
#define __SAC_SEGMENT_NODE_H_

#include <thread>

#include "flowNode.h"

#include "typeDef.h"

class SACSegmentNode : public FlowNode
{
    Q_OBJECT
    Q_PROPERTY_AUTO(int, methodType)
    Q_PROPERTY_AUTO(int, modelType)
    Q_PROPERTY_AUTO(double, distanceThreshold)
  public:
    SACSegmentNode() : FlowNode{FlowNode::Type::Poisson}, modelType_(0), methodType_(0), distanceThreshold_(1) { }
    ~SACSegmentNode();
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  private:
    std::thread workThread_;
  protected slots:
    void inNodeOutputChanged();
};

QML_DECLARE_TYPE(SACSegmentNode)

#endif //!__SAC_SEGMENT_NODE_H_