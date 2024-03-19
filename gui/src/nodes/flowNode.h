/**
 * @file flowNode.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __FLOWNODE_H_
#define __FLOWNODE_H_

#include "QuickQanava.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/ModelCoefficients.h>

#include <vtkSmartPointer.h>
#include <vtkActor.h>

Q_DECLARE_METATYPE(pcl::PointCloud<pcl::PointXYZRGB>::Ptr)
Q_DECLARE_METATYPE(pcl::PolygonMesh::Ptr)
Q_DECLARE_METATYPE(pcl::ModelCoefficients::Ptr)
Q_DECLARE_METATYPE(vtkSmartPointer<vtkActor>)

class FlowNode : public qan::Node
{
    Q_OBJECT
  public:
    enum class Type {
        CloudInput = 0,
        CloudOutput,
        MeshOutput,
        PassThroughFilter,
        StatisticalOutlierRemoval,
        GreedyProjectionTriangulation,
        Poisson,
        SACSegment,
        SplitOutput,
        IntersectionLine,
        ThreeLineIntersection,
        ActorOutput,
    };
    Q_ENUM(Type)

    explicit FlowNode( QQuickItem* parent = nullptr ) : FlowNode( Type::PassThroughFilter, parent ) {}
    explicit FlowNode( Type type, QQuickItem* parent = nullptr ) :
                                                                 qan::Node{parent}, _type{type} { /* Nil */ }
    virtual ~FlowNode() { /* Nil */ }

    FlowNode(const FlowNode&) = delete;
    FlowNode& operator=(const FlowNode&) = delete;
    FlowNode(FlowNode&&) = delete;
    FlowNode& operator=(FlowNode&&) = delete;

    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;

  public:
    Q_PROPERTY(Type type READ getType CONSTANT FINAL)
    inline  Type    getType() const noexcept { return _type; }
  protected:
    Type            _type{Type::PassThroughFilter};

  public slots:
    virtual void    inNodeOutputChanged();

  public:
    Q_PROPERTY(QVariant output READ getOutput WRITE setOutput NOTIFY outputChanged)
    inline QVariant getOutput() const noexcept { return _output; }
    void            setOutput(QVariant output) noexcept;
  protected:
    QVariant        _output;
  signals:
    void            outputChanged();
};

#endif // !__FLOWNODE_H_
