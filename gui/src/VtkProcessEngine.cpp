#include "VtkProcessEngine.h"
#include "VtkCusInteractorStyleRubberBandPick.h"

#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyLine.h>
#include <vtkPolyDataMapper.h>
#include <vtkConeSource.h>
#include <vtkAxesActor.h>
#include <vtkCubeSource.h>
#include <vtkBoxRepresentation.h>
#include <vtkCallbackCommand.h>
#include <vtkCaptionActor2D.h>
#include <vtkTextProperty.h>
#include <vtkTextActor.h>
#include <vtkProperty.h>
#include <vtkBox.h>
#include <vtkBoxWidget2.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindow.h>
#include <vtkQWidgetRepresentation.h>
#include <QQuickVTKInteractiveWidget.h>
#include <QQuickVTKInteractorAdapter.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCamera.h>
#include <vtkLookupTable.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkUnStructuredGrid.h>
#include <vtkPolyVertex.h>
#include <vtkDataSetMapper.h>
#include <vtkDataSet.h>
#include <vtkTextProperty.h>
#include <vtkProperty2D.h>
#include <vtkAreaPicker.h>
#include <vtkProp3DCollection.h>
#include <vtkDataSetSurfaceFilter.h>
//#include <vtkSurfaceReconstructionFilter.h>
#include <vtkDelaunay2D.h>
#include <vtkContourFilter.h>
#include <vtkPLYWriter.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPolyDataNormals.h>

#include <pcl/PolygonMesh.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>

VTKProcessEngine* VTKProcessEngine::vtkProcessEngine_ = new VTKProcessEngine();
vtkNew<vtkActor> tempActor;

vtkNew<vtkLookupTable> lookupPre;
const double axesActor_length = 100.0;
const int16_t axesActor_label_font_size = 20;

void progressValCallbackFunc(vtkObject* obj,unsigned long eid,void* clientdata,void *calldata) {
    double* progressVal = static_cast<double*>(clientdata);
    *progressVal = *(static_cast<double*>(calldata));
}

void createLine(const double x1, const double y1, const double z1,
                const double x2, const double y2, const double z2,
                vtkSmartPointer<vtkPoints> points, vtkSmartPointer<vtkCellArray> cells)
{
    vtkSmartPointer<vtkPolyLine> line;
    line = vtkSmartPointer<vtkPolyLine>::New();
    line->GetPointIds()->SetNumberOfIds(2);

    vtkIdType id_1, id_2;
    id_1 = points->InsertNextPoint(x1, y1, z1);
    id_2 = points->InsertNextPoint(x2, y2, z2);

    line->GetPointIds()->SetId(0, id_1);
    line->GetPointIds()->SetId(1, id_2);

    cells->InsertNextCell(line);
}

void VTKProcessEngine::initOrientedWidget(VTKRenderItem* item, const bool isPostProcess) {
    orientedWidgetAxesActor_->SetTotalLength(axesActor_length, axesActor_length, axesActor_length);

    if(!isPostProcess) {
        orientationWidget_ = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
        orientationWidget_->SetOutlineColor(0.9300, 0.5700, 0.1300);
        orientationWidget_->SetOrientationMarker(orientedWidgetAxesActor_);
        orientationWidget_->SetInteractor(item->renderWindow()->renderWindow()->GetInteractor());
        orientationWidget_->SetViewport(0.8, 0, 0.95, 0.2);
        orientationWidget_->SetEnabled(true);
        orientationWidget_->On();
        orientationWidget_->InteractiveOff();
    }
    else {
        postProcessOrientationWidget_ = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
        postProcessOrientationWidget_->SetOutlineColor(0.9300, 0.5700, 0.1300);
        postProcessOrientationWidget_->SetOrientationMarker(orientedWidgetAxesActor_);
        postProcessOrientationWidget_->SetInteractor(item->renderWindow()->renderWindow()->GetInteractor());
        postProcessOrientationWidget_->SetViewport(0.8, 0, 0.95, 0.2);
        postProcessOrientationWidget_->SetEnabled(true);
        postProcessOrientationWidget_->On();
        postProcessOrientationWidget_->InteractiveOff();
    }
}

void VTKProcessEngine::addAxesActorAndWidget(VTKRenderItem* item) {
    axesActor_->SetTotalLength(axesActor_length, axesActor_length, axesActor_length);
    axesActor_->GetXAxisCaptionActor2D()->GetTextActor()->SetTextScaleModeToNone();
    axesActor_->GetYAxisCaptionActor2D()->GetTextActor()->SetTextScaleModeToNone();
    axesActor_->GetZAxisCaptionActor2D()->GetTextActor()->SetTextScaleModeToNone();
    axesActor_->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(axesActor_label_font_size);
    axesActor_->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(axesActor_label_font_size);
    axesActor_->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(axesActor_label_font_size);

    item->renderer()->AddActor(axesActor_);
}

void VTKProcessEngine::addGridActor(VTKRenderItem* item) {
    vtkNew<vtkPolyData> platformGrid;
    vtkSmartPointer<vtkPolyDataMapper> platformGridMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    platformGridMapper->SetInputData(platformGrid);

    gridActor_->SetMapper(platformGridMapper);
    gridActor_->GetProperty()->LightingOff();
    gridActor_->GetProperty()->SetColor(0.45, 0.45, 0.45);
    gridActor_->GetProperty()->SetOpacity(1);
    gridActor_->PickableOff();
    double platformWidth = 200.0;
    double platformDepth = 200.0;
    double gridBottomHeight = 0.15;
    uint16_t gridSize = 10;

    vtkSmartPointer<vtkPoints> gridPoints = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> gridCells = vtkSmartPointer<vtkCellArray>::New();

    for (int16_t i = -platformWidth / 2; i <= platformWidth / 2; i += gridSize)
    {
        createLine(i, -platformDepth / 2, gridBottomHeight, i, platformDepth / 2, gridBottomHeight, gridPoints, gridCells);
    }

    for (int16_t i = -platformDepth / 2; i <= platformDepth / 2; i += gridSize)
    {
        createLine(-platformWidth / 2, i, gridBottomHeight, platformWidth / 2, i, gridBottomHeight, gridPoints, gridCells);
    }

    platformGrid->SetPoints(gridPoints);
    platformGrid->SetLines(gridCells);


    item->renderer()->AddActor(gridActor_);
}

void VTKProcessEngine::initRenderWindow(VTKRenderItem* item, const bool isPostProcess) {
    vtkNew<vtkCusInteractorStyleRubberBandPick> style;
    style->SetDefaultRenderer(item->renderer());
    style->bindRenderItem(item);
    style->bindVtkProcessEngine(this);
    isPostProcess ? style->bindCloudActor(processCloud_) : style->bindCloudActor(cloud_);

    vtkNew<vtkAreaPicker> areaPicker;

    item->renderWindow()->renderWindow()->GetInteractor()->SetPicker(areaPicker);
    item->renderWindow()->renderWindow()->GetInteractor()->SetInteractorStyle(style);
}

void VTKProcessEngine::initRenderItem(VTKRenderItem* item, const bool isPostProcess) {
    initRenderWindow(item, isPostProcess);
    initOrientedWidget(item, isPostProcess);
    addAxesActorAndWidget(item);
    addGridActor(item);

    double camPositionX = -250;
    double camPositionY = -400;
    double camPositionZ = 400;

    item->renderer()->SetBackgroundAlpha(1);
    item->renderer()->GetActiveCamera()->SetPosition(camPositionX, camPositionY, camPositionZ);
    item->renderer()->GetActiveCamera()->SetViewUp(0.0, 0.0, 1.0);
    item->renderer()->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
    item->renderer()->GetActiveCamera()->SetClippingRange(0.01, 100000);
    isPostProcess ? item->renderer()->AddActor(processCloud_) : item->renderer()->AddActor(cloud_);
    isPostProcess ? item->renderer()->AddActor(processmesh_) : item->renderer()->AddActor(mesh_);
    isPostProcess ? item->renderer()->AddActor(processScalarBar_) : item->renderer()->AddActor(scalarBar_);
    item->renderer()->ResetCamera();
    item->renderer()->DrawOn();

    item->pushCommandToQueue([=]{
        item->update();
    });
}

void VTKProcessEngine::bindScanRenderItem(VTKRenderItem* item) {
    scanRenderItem_ = item;

    initRenderItem(scanRenderItem_, false);

    timer_.reset(new QTimer(item));
    timer_->setInterval(1);
    connect(timer_.get(), &QTimer::timeout, this, &VTKProcessEngine::renderCloud);
    timer_->start();
}

void VTKProcessEngine::bindPostProcessRenderItem(VTKRenderItem* item) {
    postProcessRenderItem_ = item;

    initRenderItem(postProcessRenderItem_, true);
}

void VTKProcessEngine::setBackgroundColor(QColor color) {
    if(scanRenderItem_) {
        scanRenderItem_->renderer()->SetBackground(color.redF(), color.greenF(), color.blueF());
        scanRenderItem_->update();
    }

    if(postProcessRenderItem_) {
        postProcessRenderItem_->renderer()->SetBackground(color.redF(), color.greenF(), color.blueF());
        postProcessRenderItem_->update();
    }
}

void VTKProcessEngine::enableAxes(const bool isEnable) {
    curItem_->pushCommandToQueue([=] {
        if(isEnable) {
            axesActor_->VisibilityOn();
        }
        else {
            axesActor_->VisibilityOff();
        }

        curItem_->update();
    });
}

void VTKProcessEngine::enableGrid(const bool isEnable) {
    curItem_->pushCommandToQueue([=] {
        if(isEnable) {
            gridActor_->VisibilityOn();
        }
        else {
            gridActor_->VisibilityOff();
        }

        curItem_->update();
    });
}

void VTKProcessEngine::enableOriention(const bool isEnable) {
    curItem_->pushCommandToQueue([=] {
        if(isEnable) {
            orientationWidget_->On();
        }
        else {
            orientationWidget_->Off();
        }

        curItem_->update();
    });
}

VTKProcessEngine::VTKProcessEngine() : pointSize_(0), postProcessPointSize_(0), vtkExit_(false), postProcessRenderItem_(nullptr), scanRenderItem_(nullptr) {
    passCloudToMapperThread_ = std::thread(&VTKProcessEngine::passCloudToMapper, this);

    cloud_->GetProperty()->SetRepresentationToPoints();
    cloud_->GetProperty()->SetPointSize(1);
    processCloud_->GetProperty()->SetRepresentationToPoints();
    processCloud_->GetProperty()->SetPointSize(1);

    scalarBar_->SetTitle("Z(mm)");
    scalarBar_->SetNumberOfLabels(20);
    scalarBar_->DrawAnnotationsOn();
    scalarBar_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    scalarBar_->GetPositionCoordinate()->SetValue(0.95f, 0.02f);
    scalarBar_->SetWidth(0.04);
    scalarBar_->SetHeight(0.5);
    scalarBar_->SetTextPositionToPrecedeScalarBar();
    scalarBar_->GetTitleTextProperty()->SetColor(1, 1, 1);
    scalarBar_->GetTitleTextProperty()->SetFontSize(4);
    scalarBar_->GetLabelTextProperty()->SetColor(1, 1, 1);
    scalarBar_->GetLabelTextProperty()->SetFontSize(4);
    scalarBar_->GetAnnotationTextProperty()->SetColor(1, 1, 1);
    scalarBar_->GetAnnotationTextProperty()->SetFontSize(4);

    processScalarBar_->SetTitle("Z(mm)");
    processScalarBar_->SetNumberOfLabels(20);
    processScalarBar_->DrawAnnotationsOn();
    processScalarBar_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    processScalarBar_->GetPositionCoordinate()->SetValue(0.95f, 0.02f);
    processScalarBar_->SetWidth(0.04);
    processScalarBar_->SetHeight(0.5);
    processScalarBar_->SetTextPositionToPrecedeScalarBar();
    processScalarBar_->GetTitleTextProperty()->SetColor(1, 1, 1);
    processScalarBar_->GetTitleTextProperty()->SetFontSize(4);
    processScalarBar_->GetLabelTextProperty()->SetColor(1, 1, 1);
    processScalarBar_->GetLabelTextProperty()->SetFontSize(4);
    processScalarBar_->GetAnnotationTextProperty()->SetColor(1, 1, 1);
    processScalarBar_->GetAnnotationTextProperty()->SetFontSize(4);

    scalarBar_->VisibilityOff();
    processScalarBar_->VisibilityOff();
}

VTKProcessEngine::~VTKProcessEngine() {
    vtkExit_.store(true, std::memory_order_release);

    if(passCloudToMapperThread_.joinable()) {
        passCloudToMapperThread_.join();
    }
}

void VTKProcessEngine::passCloudToMapper() {
    while(!vtkExit_.load(std::memory_order_acquire)) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

        if(!waitRenderedClouds_.try_pop(cloud)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        const int pointSize = cloud->points.size();

        vtkNew<vtkPolyVertex> polyVertex;
        polyVertex->GetPointIds()->SetNumberOfIds(pointSize);

        vtkNew<vtkLookupTable> lookup;
        lookup->SetNumberOfTableValues(pointSize);
        lookup->Build();

        vtkNew<vtkUnstructuredGrid> grid;
        vtkNew<vtkPoints> points;

        vtkNew<vtkFloatArray> scalars;
        scalars->SetName("colorTable");
        scalars->SetNumberOfTuples(pointSize);
        for (size_t i = 0; i< pointSize; ++i) {
            vtkIdType pid[1];
            pid[0] =  points->InsertNextPoint(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
            lookup->SetTableValue(i, cloud->points[i].r / 255.f, cloud->points[i].g / 255.f, cloud->points[i].b / 255.f, 1);
            polyVertex->GetPointIds()->SetId(i, i);
            scalars->InsertValue(i, i);
        }

        grid->Allocate(1, 1);
        grid->SetPoints(points);
        grid->GetPointData()->SetScalars(scalars);
        grid->InsertNextCell(polyVertex->GetCellType(), polyVertex->GetPointIds());

        vtkNew<vtkDataSetMapper> mapper;
        mapper->SetInputData(grid);
        mapper->SetLookupTable(lookup);
        mapper->ScalarVisibilityOn();
        mapper->SetScalarRange(0, pointSize - 1);

        waitRenderedMappers_.push(mapper);
    }
}

void VTKProcessEngine::emplaceRenderCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    waitRenderedClouds_.push(cloud);

    std::cout << "Cloud Size: " << waitRenderedClouds_.size() << std::endl;
}

void VTKProcessEngine::renderCloud() {
    vtkSmartPointer<vtkMapper> mapper;

    if(!waitRenderedMappers_.try_pop(mapper)) {
        return;
    }

    if(curItem_ == scanRenderItem_) {
        pointSize_ = mapper->GetInput()->GetNumberOfPoints();;
        emit pointSizeChanged();
    }
    else {
        postProcessPointSize_ = mapper->GetInput()->GetNumberOfPoints();
        emit postProcessPointSizeChanged();
    }

    std::cout << "Mapper Size: " << waitRenderedMappers_.size() << std::endl;

    //在渲染流程中进行操作，否则易出现内存泄露或上下文获取失败
    curItem_->pushCommandToQueue([&, mapper] {
        vtkSmartPointer<vtkActor> curCloud = curItem_ == postProcessRenderItem_ ? processCloud_ : cloud_;
        vtkSmartPointer<vtkScalarBarActor> curScalarBar = curItem_ == postProcessRenderItem_ ? processScalarBar_ : scalarBar_;
        curCloud->SetMapper(mapper);
        vtkNew<vtkLookupTable> zLookUpTable;
        zLookUpTable->SetNumberOfTableValues(10);
        zLookUpTable->SetHueRange(0.67, 0.0);
        zLookUpTable->SetTableRange(curCloud->GetZRange()[0], curCloud->GetZRange()[1]);
        zLookUpTable->Build();
        curScalarBar->SetLookupTable(zLookUpTable);

        vtkSmartPointer<vtkScalarBarActor> curScalarBarActor = curItem_ == postProcessRenderItem_ ? processScalarBar_ : scalarBar_;
        if(curScalarBarActor->GetVisibility()) {
            jetDepthColorMap();
        }

        curItem_->update();
    });
}

void VTKProcessEngine::setCurRenderItem(VTKRenderItem* item) {
    curItem_ = item;
}

void VTKProcessEngine::enableColorBar(const bool isEnable) {
    vtkSmartPointer<vtkScalarBarActor> curScalarBarActor = curItem_ == postProcessRenderItem_ ? processScalarBar_ : scalarBar_;
    isEnable ? curScalarBarActor->VisibilityOn() : curScalarBarActor->VisibilityOff();
}

void VTKProcessEngine::enableAreaSelected(const bool isEnable) {
          auto style = static_cast<vtkCusInteractorStyleRubberBandPick*>(curItem_->renderWindow()->renderWindow()->GetInteractor()->GetInteractorStyle());
          isEnable ? style->StartSelect() : style->stopSelect();
}

void VTKProcessEngine::updateSelectedRec() {
          emit paintRectangle();
}

void VTKProcessEngine::saveCloud(const QString path) {
          vtkSmartPointer<vtkActor> curCloud = curItem_ == postProcessRenderItem_ ? processCloud_ : cloud_;
          if(!curCloud->GetMapper()) {
        return ;
          }

          if(asyncThread_.joinable()) {
                asyncThread_.join();
          }

          asyncThread_ = std::thread([&, path, curCloud] {
              double progressVal;

              std::thread saveThread = std::thread([&] {
                  vtkNew<vtkDataSetSurfaceFilter> surfaceFilter;
                  surfaceFilter->SetInputData(curCloud->GetMapper()->GetInput());
                  surfaceFilter->Update();

                  auto lookup = vtkLookupTable::SafeDownCast(curCloud->GetMapper()->GetLookupTable());

                  vtkNew<vtkPLYWriter> writer;
                  writer->SetFileName(path.mid(8).toLocal8Bit().toStdString().c_str());
                  writer->SetInputData(surfaceFilter->GetOutput());
                  writer->SetLookupTable(lookup);
                  writer->SetArrayName("colorTable");
                  writer->SetFileTypeToASCII();

                  vtkNew<vtkCallbackCommand> progressCommand;
                  progressCommand->SetCallback(progressValCallbackFunc);
                  progressCommand->SetClientData(&progressVal);
                  writer->AddObserver(vtkCommand::ProgressEvent, progressCommand);

                  writer->Update();
              });

              while(progressVal < 1) {
                  progressVal_ = progressVal;
                  emit progressValChanged();
                  std::this_thread::sleep_for(std::chrono::milliseconds(300));
              }

              progressVal_ = 1;
              emit progressValChanged();

              if(saveThread.joinable()) {
                  saveThread.join();
              }
          });
}

void VTKProcessEngine::clip(const bool isClipInner) {
          vtkSmartPointer<vtkActor> curCloud = curItem_ == postProcessRenderItem_ ? processCloud_ : cloud_;

          if(!curCloud->GetMapper()) {
             cancelClip();
             return;
          }

          if(asyncThread_.joinable()) {
             asyncThread_.join();
          }

          asyncThread_ = std::thread([&, isClipInner] {
              double progressVal = 0;
              std::thread clipThread = std::thread([&] {
                  auto style = static_cast<vtkCusInteractorStyleRubberBandPick*>(curItem_->renderWindow()->renderWindow()->GetInteractor()->GetInteractorStyle());
                  style->clip(isClipInner, progressVal);
              });

              while(progressVal < 1) {
                  progressVal_ = progressVal;
                  emit progressValChanged();

                  std::this_thread::sleep_for(std::chrono::milliseconds(300));
              }

              vtkNew<vtkDataSetSurfaceFilter> surfaceFilter;
              surfaceFilter->SetInputData(cloud_->GetMapper()->GetInput());
              surfaceFilter->Update();

              vtkLookupTable::SafeDownCast(scalarBar_->GetLookupTable())->SetTableRange(surfaceFilter->GetOutput()->GetBounds()[4], surfaceFilter->GetOutput()->GetBounds()[5]);

              progressVal_ = 1;
              emit progressValChanged();

              if(clipThread.joinable()) {
                  clipThread.join();
              }
          });
}

void VTKProcessEngine::cancelClip() {
    auto style = static_cast<vtkCusInteractorStyleRubberBandPick*>(curItem_->renderWindow()->renderWindow()->GetInteractor()->GetInteractorStyle());
    style->cancelClip();
}

void VTKProcessEngine::enablePointInfo(const bool isEnable) {
          auto style = static_cast<vtkCusInteractorStyleRubberBandPick*>(curItem_->renderWindow()->renderWindow()->GetInteractor()->GetInteractorStyle());
          style->enablePointInfoMode(isEnable);
}

void VTKProcessEngine::release() {
    if(asyncThread_.joinable()) {
        asyncThread_.join();
    }
}

void VTKProcessEngine::colorizeCloud(QColor color) {
          vtkSmartPointer<vtkActor> curCloud = curItem_ == postProcessRenderItem_ ? processCloud_ : cloud_;

          if(!curCloud->GetMapper()) {
             return;
          }

          auto r = color.redF();
          auto g = color.greenF();
          auto b = color.blueF();
          curCloud->GetProperty()->SetColor(r, g, b);

          curCloud->GetMapper()->SetScalarVisibility(false);

          curItem_->update();
}

void VTKProcessEngine::cancelColorizeCloud() {
          vtkSmartPointer<vtkActor> curCloud = curItem_ == postProcessRenderItem_ ? processCloud_ : cloud_;

          if(!curCloud->GetMapper()) {
             return;
          }

          curCloud->GetMapper()->SetScalarVisibility(true);
          auto lookUp = static_cast<vtkLookupTable*>(curCloud->GetMapper()->GetLookupTable());
          lookUp->DeepCopy(lookupPre);

          curItem_->update();
}

void VTKProcessEngine::jetDepthColorMap() {
          vtkSmartPointer<vtkActor> curCloud = curItem_ == postProcessRenderItem_ ? processCloud_ : cloud_;

          if(!curCloud->GetMapper()) {
             return;
          }

          vtkNew<vtkDataSetSurfaceFilter> surfaceFilter;
          surfaceFilter->SetInputData(curCloud->GetMapper()->GetInput());
          surfaceFilter->Update();

          double zMin = surfaceFilter->GetOutput()->GetBounds()[4];
          double zMax = surfaceFilter->GetOutput()->GetBounds()[5];

          auto points = vtkUnstructuredGrid::SafeDownCast(curCloud->GetMapper()->GetInput())->GetPoints();
          vtkNew<vtkLookupTable> jetMapLookup;
          jetMapLookup->SetHueRange(0.67, 0.0);
          jetMapLookup->SetTableRange(zMin, zMax);
          jetMapLookup->Build();

          auto lookUp = static_cast<vtkLookupTable*>(curCloud->GetMapper()->GetLookupTable());
          lookupPre->DeepCopy(lookUp);

          auto scalars = vtkFloatArray::SafeDownCast(vtkUnstructuredGrid::SafeDownCast(vtkDataSetMapper::SafeDownCast(curCloud->GetMapper())->GetInput())->GetPointData()->GetScalars());

          for (size_t i = 0; i < scalars->GetNumberOfTuples(); ++i) {
             double tabColor[3];
             jetMapLookup->GetColor(points->GetPoint(i)[2], tabColor);
             lookUp->SetTableValue(scalars->GetValue(i), tabColor[0], tabColor[1], tabColor[2]);
          }

          curCloud->GetMapper()->SetScalarVisibility(true);
          curCloud->GetProperty()->SetRepresentationToPoints();

          curItem_->update();
}

void VTKProcessEngine::setCameraViewPort(double x, double y, double z, double fx, double fy, double fz, double vx, double vy, double vz) {
          vtkSmartPointer<vtkActor> curCloud = curItem_ == postProcessRenderItem_ ? processCloud_ : cloud_;
          vtkNew<vtkCamera> camera;

          if(curCloud->GetMapper()) {
             vtkNew<vtkDataSetSurfaceFilter> surfaceFilter;
             surfaceFilter->SetInputData(curCloud->GetMapper()->GetInput());
             surfaceFilter->Update();

             double bounds[6], center[3];
             surfaceFilter->GetOutput()->GetBounds(bounds);
             surfaceFilter->GetOutput()->GetCenter(center);

             if(x != 0) {
                 x < 0 ? camera->SetPosition(bounds[0] + x, center[1], center[2]) : camera->SetPosition(bounds[1] + x, center[1], center[2]);
                 x < 0 ? camera->SetFocalPoint(bounds[0] + x + 100, center[1], center[2]) : camera->SetFocalPoint(bounds[1] + x - 100, center[1], center[2]);
             }
             else if(y != 0) {
                 y < 0 ? camera->SetPosition(center[0], bounds[2] + y, center[2]) : camera->SetPosition(center[0], bounds[3] + y, center[2]);
                 y < 0 ? camera->SetFocalPoint(center[0], bounds[2] + y + 100, center[2]) : camera->SetFocalPoint(center[0], bounds[3] + y - 100, center[2]);
             }
             else if(z != 0) {
                 z < 0 ? camera->SetPosition(center[0], center[1], bounds[4] + z) : camera->SetPosition(center[0], center[1], bounds[5] + z);
                 z < 0 ? camera->SetFocalPoint(center[0], center[1], bounds[4] + z + 100) : camera->SetFocalPoint(center[0], center[1], bounds[5] + z - 100);
             }
          }
          else {
             camera->SetFocalPoint(fx, fy, fz);
             camera->SetPosition(x, y, z);
          }

          camera->SetViewUp(vx, vy, vz);
          curItem_->renderer()->SetActiveCamera(camera);

          curItem_->update();
}

void VTKProcessEngine::getCameraViewPort(double& x, double& y, double& z, double& fx, double& fy, double& fz, double& vx, double& vy, double& vz) {
   auto camera = curItem_->renderer()->GetActiveCamera();
   camera->GetViewUp(vx, vy, vz);
   camera->GetPosition(x, y, z);
   camera->GetFocalPoint(fx, fy, fz);
}

void VTKProcessEngine::enableMesh(const bool isEnable) {
   vtkSmartPointer<vtkActor> curMesh = curItem_ == postProcessRenderItem_ ? processmesh_ : mesh_;

    if(!curMesh->GetMapper()) {
        return;
    }

    isEnable ? curMesh->VisibilityOn() : curMesh->VisibilityOff();

    curItem_->update();
}

void VTKProcessEngine::enableCloud(const bool isEnable) {
    vtkSmartPointer<vtkActor> curCloud = curItem_ == postProcessRenderItem_ ? processCloud_ : cloud_;

    if(!curCloud->GetMapper()) {
        return;
    }

    isEnable ? curCloud->VisibilityOn() : curCloud->VisibilityOff();

    curItem_->update();
}

void VTKProcessEngine::updateProcessActor(vtkSmartPointer<vtkActor> actor) {
    curItem_ = postProcessRenderItem_;

    emit postProcessOutput();

    enableCloud(true);
    enableMesh(true);

    if(actor) {
        processActor_->ShallowCopy(actor);
        curItem_->renderer()->AddActor(processActor_);
        curItem_->update();
    }
}

void VTKProcessEngine::updateProcessCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    curItem_ = postProcessRenderItem_;
    //如果点云不为空，则跳转至结果页面
    if(!cloud->empty()) {
        emit postProcessOutput();
    }

    enableCloud(true);
    enableMesh(false);

    emplaceRenderCloud(cloud);
}

void VTKProcessEngine::updateProcessMesh(pcl::PolygonMesh::Ptr mesh) {
    curItem_ = postProcessRenderItem_;

    if(!mesh->polygons.empty()) {
        emit postProcessOutput();
    }

    vtkSmartPointer<vtkPolyData> meshVtk;
    pcl::VTKUtils::mesh2vtk(*mesh, meshVtk);

    vtkNew<vtkPolyDataMapper> meshMapper;
    meshMapper->SetInputData(meshVtk);

    vtkSmartPointer<vtkActor> curMesh = curItem_ == postProcessRenderItem_ ? processmesh_ : mesh_;
    curMesh->SetMapper(meshMapper);
    curMesh->GetProperty()->SetRepresentationToSurface();

    enableCloud(false);
    enableMesh(true);

    curItem_->update();
}
