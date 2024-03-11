#ifndef __MESH_OUTPUT_NODE_H_
#define __MESH_OUTPUT_NODE_H_

#include "QuickQanava.h"

#include "flowNode.h"

class MeshOutputNode : public FlowNode {
    Q_OBJECT
  public:
    MeshOutputNode();
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  protected slots:
    void inNodeOutputChanged();
};

QML_DECLARE_TYPE(MeshOutputNode)

#endif // !__MESH_OUTPUT_NODE_H_
