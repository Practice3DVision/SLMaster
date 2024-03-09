#include "passThroughFilterNode.h"

#include <pcl/filters/passthrough.h>

using namespace pcl;

QQmlComponent*  PassThroughFilterNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent>   delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/PassThroughFilterNode.qml");
    return delegate.get();
}

PassThroughFilterNode::~PassThroughFilterNode() {
    if(workThread_.joinable()) {
        workThread_.join();
    }
}

void PassThroughFilterNode::inNodeOutputChanged()
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
                PassThrough<PointXYZRGB> filter;

                if(filterX_) {
                    filter.setInputCloud(inOutput);
                    filter.setFilterFieldName("x");
                    filter.setFilterLimits(minX_, maxX_);
                    filter.filter(*outPutCloud);
                }

                if(filterY_) {
                    outPutCloud->empty() ? filter.setInputCloud(inOutput) : filter.setInputCloud(outPutCloud);
                    filter.setFilterFieldName("y");
                    filter.setFilterLimits(minY_, maxY_);
                    filter.filter(*outPutCloud);
                }

                if(filterZ_) {
                    outPutCloud->empty() ? filter.setInputCloud(inOutput) : filter.setInputCloud(outPutCloud);
                    filter.setFilterFieldName("y");
                    filter.setFilterLimits(minZ_, maxZ_);
                    filter.filter(*outPutCloud);
                }

                qInfo() << "[Pass Through Filter Node]: Completed!";
            }
            else {
                qInfo() << "[Pass Through Filter Node]: cloud is empty!";
            }
        }
        setOutput(QVariant::fromValue(outPutCloud));
    });
}
