#include "sacSegmentNode.h"

#include <pcl/common/io.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

using namespace pcl;

QQmlComponent*  SACSegmentNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent> delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/SACSegmentNode.qml");
    return delegate.get();
}

SACSegmentNode::~SACSegmentNode() {
    if(workThread_.joinable()) {
        workThread_.join();
    }
}

void SACSegmentNode::inNodeOutputChanged()
{
    FlowNode::inNodeOutputChanged();

    if(workThread_.joinable()) {
        workThread_.join();
    }

    workThread_ = std::thread([&]{
        PointCloud<PointXYZRGB>::Ptr modelCloud(new PointCloud<PointXYZRGB>);
        PointCloud<PointXYZRGB>::Ptr remainCloud(new PointCloud<PointXYZRGB>);
        PointIndicesPtr indices(new PointIndices);
        ModelCoefficientsPtr models(new ModelCoefficients);

        for (const auto inNode : get_in_nodes()) {
            const auto inFlowNode = qobject_cast<FlowNode*>(inNode);
            if (inFlowNode == nullptr ||
                !inFlowNode->getOutput().isValid())
                continue;

            auto inOutput = inFlowNode->getOutput().value<PointCloud<PointXYZRGB>::Ptr>();
            if (!inOutput->empty()) {
                SACSegmentation<PointXYZRGB> segmentTool;
                segmentTool.setOptimizeCoefficients(true);
                segmentTool.setModelType(modelType_);
                segmentTool.setMethodType(methodType_);
                segmentTool.setDistanceThreshold(distanceThreshold_);
                segmentTool.setInputCloud(inOutput);
                segmentTool.segment(*indices, *models);
                
                ExtractIndices<PointXYZRGB> extract;
                extract.setIndices(indices);
                extract.setNegative(false);
                extract.setInputCloud(inOutput);
                extract.filter(*modelCloud);
                extract.setNegative(true);
                extract.filter(*remainCloud);

                qInfo() << "[SAC Segment Node]: Completed!";
            }
            else {
                qInfo() << "[SAC Segment Node]: cloud is empty!";
            }
        }

        setOutput(QVariant::fromValue(std::vector<PointCloud<PointXYZRGB>::Ptr>{modelCloud, remainCloud}));
    });
}