#include "threeLineIntersectionNode.h"

#include <Eigen/Eigen>
#include <pcl/ModelCoefficients.h>

#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkPolyLine.h>
#include <vtkLineSource.h>
#include <vtkProperty.h>

using namespace pcl;
using namespace Eigen;

Vector3f calcLineIntersectionPoint(ModelCoefficientsPtr lineA, ModelCoefficientsPtr lineB) {
    Vector3f crossPoint;

    float t = (lineB->values[3] * (lineA->values[1] - lineB->values[1]) - lineB->values[4] * (lineA->values[0] - lineB->values[0])) / (lineA->values[3] * lineB->values[4] - lineA->values[4] * lineB->values[3]);
    crossPoint(0) = lineA->values[0] + lineA->values[3] * t;
    crossPoint(1) = lineA->values[1] + lineA->values[4] * t;
    crossPoint(2) = lineA->values[2] + lineA->values[5] * t;

    return crossPoint;
}

QQmlComponent* ThreeLineIntersectionLineNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent> delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ui/nodes/ThreeLineIntersectionNode.qml");
    return delegate.get();
}

ThreeLineIntersectionLineNode::~ThreeLineIntersectionLineNode() {
    if(workThread_.joinable()) {
        workThread_.join();
    }
}

void ThreeLineIntersectionLineNode::inNodeOutputChanged()
{
    FlowNode::inNodeOutputChanged();

    if(workThread_.joinable()) {
        workThread_.join();
    }

    workThread_ = std::thread([&]{
        std::vector<ModelCoefficientsPtr> lines;
        for (const auto inNode : get_in_nodes()) {
            const auto inFlowNode = qobject_cast<FlowNode*>(inNode);
            if (inFlowNode == nullptr ||
                !inFlowNode->getOutput().isValid())
                continue;

            auto inOutput = inFlowNode->getOutput().value<ModelCoefficientsPtr>();
            if (inOutput) {
                lines.emplace_back(inOutput);
            }
            else {
                qInfo() << "[Three Line Intersection Node]: cloud is empty!";
            }
        }
        
        if(lines.size() == 3) {
            auto crossPoint = calcLineIntersectionPoint(lines[0], lines[1]);
            Eigen::Vector3f direcLength, direcWidth, direcHeight;
            direcLength << lines[0]->values[3], lines[0]->values[4], lines[0]->values[5];
            direcWidth << lines[1]->values[3], lines[1]->values[4], lines[1]->values[5];
            direcHeight << lines[2]->values[3], lines[2]->values[4], lines[2]->values[5];
            Eigen::Vector3f posLengthEnd = crossPoint + length_ * direcLength;
            Eigen::Vector3f posWidthEnd = crossPoint + width_ * direcWidth;
            Eigen::Vector3f posHeightEnd = crossPoint + height_ * direcHeight;
        
            vtkNew<vtkPoints> points;
            points->InsertNextPoint(crossPoint(0), crossPoint(1), crossPoint(2));
            points->InsertNextPoint(posLengthEnd(0), posLengthEnd(1), posLengthEnd(2));
            points->InsertNextPoint(crossPoint(0), crossPoint(1), crossPoint(2));
            points->InsertNextPoint(posWidthEnd(0), posWidthEnd(1), posWidthEnd(2));
            points->InsertNextPoint(crossPoint(0), crossPoint(1), crossPoint(2));
            points->InsertNextPoint(posHeightEnd(0), posHeightEnd(1), posHeightEnd(2));
        
            vtkNew<vtkPolyLine> line;
            line->GetPointIds()->SetNumberOfIds(6);
            line->GetPointIds()->SetId(0, 0);
            line->GetPointIds()->SetId(1, 1);
            line->GetPointIds()->SetId(2, 2);
            line->GetPointIds()->SetId(3, 3);
            line->GetPointIds()->SetId(4, 4);
            line->GetPointIds()->SetId(5, 5);
        
            vtkNew<vtkCellArray> linesCellArray;
            linesCellArray->InsertNextCell(line);

            vtkNew<vtkPolyData> polyData;
            polyData->SetPoints(points);
            polyData->SetLines(linesCellArray);
        
            vtkNew<vtkPolyDataMapper> polyDataMapper;
            polyDataMapper->SetInputData(polyData);
        
            auto lineActor = vtkSmartPointer<vtkActor>::New();
            lineActor->SetMapper(polyDataMapper);
            lineActor->GetProperty()->SetColor(0, 1, 0);
            lineActor->GetProperty()->SetLineWidth(2);

            setOutput(QVariant::fromValue(lineActor));

            qInfo() << "[Three Line Intersection Node]: Completed!";
        }
    });
}