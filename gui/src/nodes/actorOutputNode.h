#ifndef __ACTOR_OUTPUT_NODE_H_
#define __ACTOR_OUTPUT_NODE_H_

#include "QuickQanava.h"

#include "flowNode.h"

class ActorOutputNode : public FlowNode {
    Q_OBJECT
  public:
    ActorOutputNode();
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  protected slots:
    void inNodeOutputChanged();
};

QML_DECLARE_TYPE(ActorOutputNode)

#endif // !__ACTOR_OUTPUT_NODE_H_