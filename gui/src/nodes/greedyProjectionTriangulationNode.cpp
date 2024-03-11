#include "greedyProjectionTriangulationNode.h"

#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/gp3.h>

using namespace pcl;
using namespace pcl::search;

QQmlComponent*  GreedyProjectionTriangulationNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent> delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/GreedyProjectionTriangulationNode.qml");
    return delegate.get();
}

GreedyProjectionTriangulationNode::~GreedyProjectionTriangulationNode() {
    if(workThread_.joinable()) {
        workThread_.join();
    }
}

void GreedyProjectionTriangulationNode::inNodeOutputChanged()
{
    FlowNode::inNodeOutputChanged();

    if(workThread_.joinable()) {
        workThread_.join();
    }

    workThread_ = std::thread([&]{
        PolygonMesh::Ptr mesh(new PolygonMesh);
        for (const auto inNode : get_in_nodes()) {
            const auto inFlowNode = qobject_cast<FlowNode*>(inNode);
            if (inFlowNode == nullptr ||
                !inFlowNode->getOutput().isValid())
                continue;

            auto inOutput = inFlowNode->getOutput().value<PointCloud<PointXYZRGB>::Ptr>();
            if (!inOutput->empty()) {
                NormalEstimation<PointXYZRGB, Normal> nes;
                PointCloud<Normal>::Ptr normals(new PointCloud<Normal>);
                pcl::search::KdTree<PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<PointXYZRGB>);

                nes.setInputCloud(inOutput);
                nes.setSearchMethod(kdTree);
                nes.setKSearch(kSearch_);
                nes.compute(*normals);

                PointCloud<PointXYZRGBNormal>::Ptr cloud_with_normals(new PointCloud<PointXYZRGBNormal>);
                concatenateFields(*inOutput, *normals, *cloud_with_normals);

                pcl::search::KdTree<PointXYZRGBNormal>::Ptr kdTree2(new pcl::search::KdTree<PointXYZRGBNormal>);
                kdTree2->setInputCloud(cloud_with_normals);

                GreedyProjectionTriangulation<PointXYZRGBNormal> gp;
                gp.setSearchRadius(searchRadius_);
                gp.setMu(multiplier_);
                gp.setMaximumNearestNeighbors(maxNearestNumber_);
                gp.setMaximumSurfaceAngle(maximumSurfaceAngle_);
                gp.setMaximumAngle(maximumAngle_ / 180 * 3.1415926535);
                gp.setMinimumAngle(minimumAngle_ / 180 * 3.1415926535);
                gp.setNormalConsistency(normalConsistency_);
                gp.setInputCloud(cloud_with_normals);
                gp.setSearchMethod(kdTree2);
                gp.reconstruct(*mesh);

                qInfo() << "[Greedy Projection Triangulation Node]: Completed!";
            }
            else {
                qInfo() << "[Greedy Projection Triangulation Node]: In node output cloud is empty!";
            }
        }

        setOutput(QVariant::fromValue(mesh));
    });
}
