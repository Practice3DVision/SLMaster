#include "statisticalOutlierRemovalNode.h"

#include <pcl/filters/statistical_outlier_removal.h>

using namespace pcl;

QQmlComponent*  StatisticalOutlierRemovalNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent>   delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/StatisticalOutlierRemovalNode.qml");
    return delegate.get();
}

StatisticalOutlierRemovalNode::~StatisticalOutlierRemovalNode() {
    if(workThread_.joinable()) {
        workThread_.join();
    }
}

void StatisticalOutlierRemovalNode::inNodeOutputChanged()
{
    FlowNode::inNodeOutputChanged();

    if(workThread_.joinable()) {
        workThread_.join();
    }

    workThread_ = std::thread([&]{
        PointCloud<PointXYZRGB>::Ptr outPutCloud(new PointCloud<PointXYZRGB>);
        for (const auto inNode : get_in_nodes()) {
            const auto inFlowNode = qobject_cast<FlowNode*>(inNode);
            if (inFlowNode == nullptr ||
                !inFlowNode->getOutput().isValid())
                continue;

            auto inOutput = inFlowNode->getOutput().value<PointCloud<PointXYZRGB>::Ptr>();
            if (!inOutput->empty()) {
                StatisticalOutlierRemoval<PointXYZRGB> filter;
                filter.setMeanK(k_);
                filter.setStddevMulThresh(stdThreshold_);
                filter.setInputCloud(inOutput);
                filter.filter(*outPutCloud);

                qInfo() << "[Statistical Outlier Removal Node]: Completed!";
            }
            else {
                qInfo() << "[Statistical Outlier Removal Node]: cloud is empty!";
            }
        }
        setOutput(QVariant::fromValue(outPutCloud));
    });
}
