/**
 * @file VtkProcessEngine.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __VTK_PROCESS_ENGINE_H_
#define __VTK_PROCESS_ENGINE_H_

#include <QObject>
#include <QQmlApplicationEngine>
#include <QQuickVTKRenderWindow.h>
#include <QQuickWindow>
#include <QTimer>

#include <vtkAxesActor.h>
#include <vtkRenderer.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkScalarBarActor.h>

#include "vtkRenderItem.h"
#include "safeQueue.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

class VTKProcessEngine : public QObject {
    Q_OBJECT
    Q_PROPERTY(float progressVal READ progressVal NOTIFY progressValChanged)
    Q_PROPERTY(int pointSize READ pointSize NOTIFY pointSizeChanged FINAL)
    Q_PROPERTY(int postProcessPointSize READ postProcessPointSize NOTIFY postProcessPointSizeChanged FINAL)
public:
    static VTKProcessEngine* instance() { return vtkProcessEngine_; };
    void updateSelectedRec();
    void bindEngine(QQmlApplicationEngine* engine) { engine_ = engine; }
    vtkSmartPointer<vtkActor> getProcessActor() { return processActor_; }
    Q_INVOKABLE void bindScanRenderItem(VTKRenderItem* item);
    Q_INVOKABLE void bindPostProcessRenderItem(VTKRenderItem* item);
    Q_INVOKABLE void updateProcessCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
    Q_INVOKABLE void updateProcessMesh(pcl::PolygonMesh::Ptr mesh);
    Q_INVOKABLE void updateProcessActor(vtkSmartPointer<vtkActor> actor);
    Q_INVOKABLE void enablePointInfo(const bool isEnable);
    Q_INVOKABLE void clip(const bool isClipInner);
    Q_INVOKABLE void cancelClip();
    Q_INVOKABLE void saveCloud(const QString path);
    Q_INVOKABLE void enableAxes(const bool isEnable);
    Q_INVOKABLE void enableGrid(const bool isEnable);
    Q_INVOKABLE void enableOriention(const bool isEnable);
    Q_INVOKABLE void enableColorBar(const bool isEnable);
    Q_INVOKABLE void enableAreaSelected(const bool isEnable);
    Q_INVOKABLE void enableMesh(const bool isEnable);
    Q_INVOKABLE void enableCloud(const bool isEnable);
    Q_INVOKABLE void release();
    Q_INVOKABLE void colorizeCloud(QColor color);
    Q_INVOKABLE void cancelColorizeCloud();
    Q_INVOKABLE void jetDepthColorMap();
    Q_INVOKABLE void setBackgroundColor(QColor color);
    Q_INVOKABLE void setCameraViewPort(double x, double y, double z, double fx, double fy, double fz, double vx, double vy, double vz);
    Q_INVOKABLE void getCameraViewPort(double& x, double& y, double& z, double& fx, double& fy, double& fz, double& vx, double& vy, double& vz);
    Q_INVOKABLE void addAxesActorAndWidget(VTKRenderItem* item);
    Q_INVOKABLE void addGridActor(VTKRenderItem* item);
    Q_INVOKABLE void setCurRenderItem(VTKRenderItem* item);
    Q_INVOKABLE inline float progressVal() {
        return progressVal_;
    }
    Q_INVOKABLE int pointSize() { return pointSize_; }
    Q_INVOKABLE int postProcessPointSize() { return postProcessPointSize_; }
    Q_INVOKABLE void emplaceRenderCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
    void renderCloud();
signals:
    void progressValChanged();
    void paintRectangle();
    void pointInfoChanged(const float x, const float y, const float z);
    void pointSizeChanged();
    void postProcessPointSizeChanged();
    void postProcessOutput();
private:
    void passCloudToMapper();
    void initRenderItem(VTKRenderItem* item, const bool isPostProcess);
    void initOrientedWidget(VTKRenderItem* item, const bool isPostProcess);
    void initRenderWindow(VTKRenderItem* item, const bool isPostProcess);
    VTKProcessEngine();
    ~VTKProcessEngine();
    VTKProcessEngine(const VTKProcessEngine&) = delete;
    VTKProcessEngine& operator=(const VTKProcessEngine&) = delete;
    static VTKProcessEngine* vtkProcessEngine_;
    void detectSignals();
    float progressVal_;
    //std::atomic_bool __isProgressValChanged;
    //std::atomic_bool __isStatusLabelValChanged;
    QQmlApplicationEngine* engine_;
    VTKRenderItem* curItem_;
    VTKRenderItem* scanRenderItem_;
    VTKRenderItem* postProcessRenderItem_;
    vtkSmartPointer<vtkOrientationMarkerWidget> orientationWidget_;
    vtkSmartPointer<vtkOrientationMarkerWidget> postProcessOrientationWidget_;
    vtkNew<vtkAxesActor> axesActor_;
    vtkNew<vtkActor> gridActor_;
    vtkNew<vtkActor> cloud_;
    vtkNew<vtkActor> processActor_;
    bool isNext = false;
    vtkNew<vtkActor> mesh_;
    vtkNew<vtkAxesActor> orientedWidgetAxesActor_;
    vtkNew<vtkScalarBarActor> scalarBar_;
    SafeQueue<vtkSmartPointer<vtkMapper>> waitRenderedMappers_;
    SafeQueue<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> waitRenderedClouds_;
    vtkNew<vtkActor> processCloud_;
    vtkNew<vtkActor> processmesh_;
    vtkNew<vtkScalarBarActor> processScalarBar_;
    std::thread asyncThread_;
    std::thread passCloudToMapperThread_;
    std::atomic_bool vtkExit_;
    QSharedPointer<QTimer> timer_;
    int pointSize_;
    int postProcessPointSize_;
};

#endif //__VTK_PROCESS_ENGINE_H_
