#ifndef __CLOUD_OUTPUT_NODE_H_
#define __CLOUD_OUTPUT_NODE_H_

#include "QuickQanava.h"

#include "flowNode.h"

class CloudOutputNode : public FlowNode {
    Q_OBJECT
  public:
    CloudOutputNode();
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  protected slots:
    void inNodeOutputChanged();
};

QML_DECLARE_TYPE(CloudOutputNode)

#endif // !__CLOUD_OUTPUT_NODE_H_
