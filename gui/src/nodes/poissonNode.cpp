#include "poissonNode.h"

#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/poisson.h>

using namespace pcl;
using namespace pcl::search;

QQmlComponent*  PoissonNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent> delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/PoissonNode.qml");
    return delegate.get();
}

PoissonNode::~PoissonNode() {
    if(workThread_.joinable()) {
        workThread_.join();
    }
}

void PoissonNode::inNodeOutputChanged()
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

                Poisson<PointXYZRGBNormal> pn;
                pn.setConfidence(confidence_);
                pn.setDegree(degree_);
                pn.setDepth(maxDepth_);
                pn.setIsoDivide(isoDivide_);
                pn.setOutputPolygons(false);
                pn.setSamplesPerNode(minSamplePoints_);
                pn.setScale(scale_);
                pn.setSolverDivide(solverDivide_);
                pn.setInputCloud(cloud_with_normals);
                pn.setSearchMethod(kdTree2);
                pn.performReconstruction(*mesh);

                qInfo() << "[Poisson Node]: Completed!";
            }
            else {
                qInfo() << "[Poisson Node]: cloud is empty!";
            }
        }

        setOutput(QVariant::fromValue(mesh));
    });
}
