#include "intersectionLineNode.h"

#include <pcl/common/io.h>
#include <pcl/segmentation/sac_segmentation.h>

using namespace pcl;

void calcLine(ModelCoefficientsPtr coefsOfPlane1, ModelCoefficientsPtr coefsOfPlane2, ModelCoefficientsPtr coefsOfLine)
{
    
    //方向向量n=n1×n2=(b1*c2-c1*b2,c1*a2-a1*c2,a1*b2-b1*a2)
    ModelCoefficients temcoefs;
    double a1, b1, c1, d1, a2,b2, c2, d2;
    double tempy, tempz;
    a1= coefsOfPlane1->values[0];
    b1= coefsOfPlane1->values[1];
    c1= coefsOfPlane1->values[2];
    d1= coefsOfPlane1->values[3];
    a2= coefsOfPlane2->values[0];
    b2= coefsOfPlane2->values[1];
    c2= coefsOfPlane2->values[2];
    d2= coefsOfPlane2->values[3];
    tempz= -(d1 / b1 - d2 / b2) / (c1 / b1 - c2 / b2);
    tempy= (-c1 / b1)*tempz - d1 / b1;
    coefsOfLine->values.push_back(0.0);
    coefsOfLine->values.push_back(tempy);
    coefsOfLine->values.push_back(tempz);
    coefsOfLine->values.push_back(b1*c2 - c1*b2);
    coefsOfLine->values.push_back(c1*a2 - a1*c2);
    coefsOfLine->values.push_back(a1*b2 - b1*a2);
    coefsOfLine->values[3] = coefsOfLine->values[3] / std::sqrt(coefsOfLine->values[3] * coefsOfLine->values[3] + coefsOfLine->values[4] * coefsOfLine->values[4] + coefsOfLine->values[5] * coefsOfLine->values[5]);
    coefsOfLine->values[4] = coefsOfLine->values[4] / std::sqrt(coefsOfLine->values[3] * coefsOfLine->values[3] + coefsOfLine->values[4] * coefsOfLine->values[4] + coefsOfLine->values[5] * coefsOfLine->values[5]);
    coefsOfLine->values[5] = coefsOfLine->values[5] / std::sqrt(coefsOfLine->values[3] * coefsOfLine->values[3] + coefsOfLine->values[4] * coefsOfLine->values[4] + coefsOfLine->values[5] * coefsOfLine->values[5]);
}

QQmlComponent*  IntersectionLineNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent> delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/IntersectionLineNode.qml");
    return delegate.get();
}

IntersectionLineNode::~IntersectionLineNode() {
    if(workThread_.joinable()) {
        workThread_.join();
    }
}

void IntersectionLineNode::inNodeOutputChanged()
{
    FlowNode::inNodeOutputChanged();

    if(workThread_.joinable()) {
        workThread_.join();
    }

    workThread_ = std::thread([&]{
        std::vector<PointIndicesPtr> pointIndices;
        std::vector<ModelCoefficientsPtr> modelCoefficients;
        for (const auto inNode : get_in_nodes()) {
            const auto inFlowNode = qobject_cast<FlowNode*>(inNode);
            if (inFlowNode == nullptr ||
                !inFlowNode->getOutput().isValid())
                continue;

            auto inOutput = inFlowNode->getOutput().value<PointCloud<PointXYZRGB>::Ptr>();
            if (!inOutput->empty()) {
                SACSegmentation<PointXYZRGB> segmentTool;
                segmentTool.setOptimizeCoefficients(true);
                segmentTool.setModelType(SACMODEL_PLANE);
                segmentTool.setMethodType(SAC_RANSAC);
                segmentTool.setDistanceThreshold(FLT_MAX);
                segmentTool.setInputCloud(inOutput);

                PointIndicesPtr indices(new PointIndices);
                ModelCoefficientsPtr coefficients(new ModelCoefficients);
                segmentTool.segment(*indices, *coefficients);
                pointIndices.emplace_back(indices);
                modelCoefficients.emplace_back(coefficients);
            }
            else {
                qInfo() << "[Intersection Line Node]: cloud is empty!";
            }
        }
        
        if(modelCoefficients.size() == 2) {
            ModelCoefficientsPtr lines(new ModelCoefficients);
            calcLine(modelCoefficients[0], modelCoefficients[1], lines);

            setOutput(QVariant::fromValue(lines));

            qInfo() << "[Intersection Line Node]: Completed!";
        }
    });
}