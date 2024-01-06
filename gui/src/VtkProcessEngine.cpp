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

#include <pcl/filters/statistical_outlier_removal.h>
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

void VTKProcessEngine::initOrientedWidget(VTKRenderItem* item) {
    orientedWidgetAxesActor_->SetTotalLength(axesActor_length, axesActor_length, axesActor_length);
    orientationWidget_ = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    orientationWidget_->SetOutlineColor(0.9300, 0.5700, 0.1300);
    orientationWidget_->SetOrientationMarker(orientedWidgetAxesActor_);
    orientationWidget_->SetInteractor(item->renderWindow()->renderWindow()->GetInteractor());
    orientationWidget_->SetViewport(0.8, 0, 0.95, 0.2);
    orientationWidget_->SetEnabled(true);
    orientationWidget_->On();
    orientationWidget_->InteractiveOff();
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

void VTKProcessEngine::initRenderWindow(VTKRenderItem* item) {
    vtkNew<vtkCusInteractorStyleRubberBandPick> style;
    style->SetDefaultRenderer(item->renderer());
    style->bindRenderItem(item);
    style->bindVtkProcessEngine(this);
    style->bindCloudActor(cloud_);

    vtkNew<vtkAreaPicker> areaPicker;

    item->renderWindow()->renderWindow()->GetInteractor()->SetPicker(areaPicker);
    item->renderWindow()->renderWindow()->GetInteractor()->SetInteractorStyle(style);
}

void VTKProcessEngine::initRenderItem(VTKRenderItem* item) {
    initRenderWindow(item);
    initOrientedWidget(item);
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
    item->renderer()->AddActor(cloud_);
    item->renderer()->AddActor(mesh_);
    item->renderer()->AddActor(scalarBar_);
    item->renderer()->ResetCamera();
    item->renderer()->DrawOn();

    item->pushCommandToQueue([=]{
        item->update();
    });
}

void VTKProcessEngine::bindScanRenderItem(VTKRenderItem* item) {
    scanRenderItem_ = item;

    initRenderItem(scanRenderItem_);

    timer_ = new QTimer(scanRenderItem_);
    timer_->setInterval(10);
    connect(timer_, &QTimer::timeout, this, &VTKProcessEngine::renderCloud);
    timer_->start();
}

void VTKProcessEngine::bindPostProcessRenderItem(VTKRenderItem* item) {
    postProcessRenderItem_ = item;

    initRenderItem(postProcessRenderItem_);
}

void VTKProcessEngine::bindMeasurementRenderItem(VTKRenderItem* item) {
    measurementRenderItem_ = item;

    initRenderItem(measurementRenderItem_);
}

void VTKProcessEngine::setBackgroundColor(QColor color) {
    if(curItem_) {
        curItem_->renderer()->SetBackground(color.redF(), color.greenF(), color.blueF());
        curItem_->update();
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

VTKProcessEngine::VTKProcessEngine() : pointSize_(0), vtkExit_(false) {
    passCloudToMapperThread_ = std::thread(&VTKProcessEngine::passCloudToMapper, this);

    cloud_->GetProperty()->SetRepresentationToPoints();
    cloud_->GetProperty()->SetPointSize(1);

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

    enableColorBar(false);
}

VTKProcessEngine::~VTKProcessEngine() {
    vtkExit_.store(true, std::memory_order_release);

    if(passCloudToMapperThread_.joinable()) {
        passCloudToMapperThread_.join();
    }
}

void VTKProcessEngine::passCloudToMapper() {
    while(!vtkExit_.load(std::memory_order_acquire)) {
        if(waitRenderedClouds_.empty()) {
            continue;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
        waitRenderedClouds_.pop(cloud);

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
    pointSize_ = cloud->points.size();
    emit pointSizeChanged();

    waitRenderedClouds_.push(cloud);
}

void VTKProcessEngine::renderCloud() {
    if(waitRenderedMappers_.empty()) {
        return;
    }

        vtkSmartPointer<vtkMapper> mapper;
        waitRenderedMappers_.pop(mapper);

         curItem_->pushCommandToQueue([=] {
         {
            cloud_->SetMapper(mapper);
            vtkNew<vtkLookupTable> zLookUpTable;
            zLookUpTable->SetNumberOfTableValues(10);
            zLookUpTable->SetHueRange(0.67, 0.0);
            zLookUpTable->SetTableRange(cloud_->GetZRange()[0], cloud_->GetZRange()[1]);
            zLookUpTable->Build();
            scalarBar_->SetLookupTable(zLookUpTable);
            curItem_->update();
         }
         });
}

void VTKProcessEngine::setCurRenderItem(VTKRenderItem* item) {
    curItem_ = item;
}

void VTKProcessEngine::enableColorBar(const bool isEnable) {
          isEnable ? scalarBar_->VisibilityOn() : scalarBar_->VisibilityOff();
}

void VTKProcessEngine::enableAreaSelected(const bool isEnable) {
          auto style = static_cast<vtkCusInteractorStyleRubberBandPick*>(curItem_->renderWindow()->renderWindow()->GetInteractor()->GetInteractorStyle());
          isEnable ? style->StartSelect() : style->stopSelect();
}

void VTKProcessEngine::updateSelectedRec() {
          emit paintRectangle();
}

void VTKProcessEngine::saveCloud(const QString path) {
          if(!cloud_->GetMapper()) {
             return ;
          }

          if(asyncThread_.joinable()) {
             asyncThread_.join();
          }

          asyncThread_ = std::thread([&, path] {
              double progressVal;

              std::thread saveThread = std::thread([&] {
                  vtkNew<vtkDataSetSurfaceFilter> surfaceFilter;
                  surfaceFilter->SetInputData(cloud_->GetMapper()->GetInput());
                  surfaceFilter->Update();

                  auto lookup = vtkLookupTable::SafeDownCast(cloud_->GetMapper()->GetLookupTable());

                  vtkNew<vtkPLYWriter> writer;
                  writer->SetFileName(path.mid(8).toStdString().data());
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
          if(!cloud_->GetMapper()) {
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

void VTKProcessEngine::statisticalOutRemoval(const float stdThreshold, const float meanK) {
          if(!cloud_->GetMapper()) {
             return;
          }

          if(asyncThread_.joinable()) {
             asyncThread_.join();
          }

          asyncThread_ = std::thread([&, stdThreshold, meanK] {
              pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
              auto lookupTable = vtkLookupTable::SafeDownCast(cloud_->GetMapper()->GetLookupTable());

              vtkNew<vtkDataSetSurfaceFilter> surfaceFilter;
              surfaceFilter->SetInputData(cloud_->GetMapper()->GetInput());
              surfaceFilter->Update();
              auto points = surfaceFilter->GetOutput()->GetPoints();

              progressVal_ = 0.1;
              emit progressValChanged();

              for (size_t i = 0; i < points->GetNumberOfPoints(); ++i) {
                  double x = points->GetPoint(i)[0];
                  double y = points->GetPoint(i)[1];
                  double z = points->GetPoint(i)[2];
                  double r = lookupTable->GetTableValue(i)[0] * 255;
                  double g = lookupTable->GetTableValue(i)[1] * 255;
                  double b = lookupTable->GetTableValue(i)[2] * 255;
                  cloud->points.emplace_back(pcl::PointXYZRGB(x, y, z, r, g, b));
              }

              progressVal_ = 0.3;
              emit progressValChanged();

              cloud->is_dense = false;
              cloud->width = cloud->points.size();
              cloud->height = 1;

              pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> filter;
              filter.setStddevMulThresh(stdThreshold);
              filter.setMeanK(meanK);
              filter.setInputCloud(cloud);

              pcl::Indices indicesFiltered;
              filter.filter(indicesFiltered);

              progressVal_ = 0.7;
              emit progressValChanged();

              vtkNew<vtkPoints> pointsVertex;
              vtkNew<vtkPolyVertex> polyVertex;
              vtkNew<vtkFloatArray> scalars;
              scalars->SetName("colorTable");
              vtkNew<vtkUnstructuredGrid> grid;
              lookupTable->SetNumberOfTableValues(indicesFiltered.size());
              polyVertex->GetPointIds()->SetNumberOfIds(indicesFiltered.size());
              scalars->SetNumberOfTuples(indicesFiltered.size());
              lookupTable->Build();

              for (size_t i = 0; i< indicesFiltered.size(); ++i) {
                  vtkIdType pid[1];
                  pid[0] =  pointsVertex->InsertNextPoint(cloud->points[indicesFiltered[i]].x, cloud->points[indicesFiltered[i]].y, cloud->points[indicesFiltered[i]].z);
                  lookupTable->SetTableValue(i, cloud->points[indicesFiltered[i]].r / 255.f, cloud->points[indicesFiltered[i]].g / 255.f, cloud->points[indicesFiltered[i]].b / 255.f, 1);
                  polyVertex->GetPointIds()->SetId(i, i);
                  scalars->InsertValue(i, i);
              }

              progressVal_ = 0.95;
              emit progressValChanged();

              grid->Allocate(1, 1);
              grid->SetPoints(pointsVertex);
              grid->GetPointData()->SetScalars(scalars);
              grid->InsertNextCell(polyVertex->GetCellType(), polyVertex->GetPointIds());

              vtkDataSetMapper::SafeDownCast(cloud_->GetMapper())->SetInputData(grid);
              vtkDataSetMapper::SafeDownCast(cloud_->GetMapper())->SetScalarRange(0, indicesFiltered.size() - 1);

              vtkLookupTable::SafeDownCast(scalarBar_->GetLookupTable())->SetTableRange(cloud_->GetZRange()[0], cloud_->GetZRange()[1]);

              progressVal_ = 1;
              emit progressValChanged();
          });
}

void VTKProcessEngine::surfaceRestruction() {
          if(!cloud_->GetMapper()) {
             return;
          }

          if(asyncThread_.joinable()) {
             asyncThread_.join();
          }

          asyncThread_ = std::thread([&] {
              double progressVal = 0;

              std::thread workThread = std::thread([&] {
                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

                  vtkNew<vtkDataSetSurfaceFilter> surfaceFilter;
                  surfaceFilter->SetInputData(cloud_->GetMapper()->GetInput());
                  surfaceFilter->Update();
                  auto points = surfaceFilter->GetOutput()->GetPoints();

                  progressVal = 0.1;

                  for (size_t i = 0; i < points->GetNumberOfPoints(); ++i) {
                      double x = points->GetPoint(i)[0];
                      double y = points->GetPoint(i)[1];
                      double z = points->GetPoint(i)[2];
                      cloud->points.emplace_back(pcl::PointXYZ(x, y, z));
                  }

                  progressVal = 0.3;

                  cloud->is_dense = false;
                  cloud->width = cloud->points.size();
                  cloud->height = 1;

                  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
                  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
                  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
                  tree->setInputCloud(cloud);
                  normalEstimation.setInputCloud(cloud);
                  normalEstimation.setSearchMethod(tree);
                  normalEstimation.setKSearch(20);
                  normalEstimation.compute(*normals);

                  progressVal = 0.5;

                  pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals(new pcl::PointCloud<pcl::PointNormal>);
                  pcl::concatenateFields(*cloud, *normals, *cloudNormals);

                  pcl::search::KdTree<pcl::PointNormal>::Ptr searchTreeNormals(new pcl::search::KdTree<pcl::PointNormal>);
                  searchTreeNormals->setInputCloud(cloudNormals);

                  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
                  pcl::PolygonMesh meshPcl;

                  gp3.setSearchRadius(20);
                  gp3.setMu(2.5f);
                  gp3.setMaximumNearestNeighbors(200);
                  gp3.setMaximumSurfaceAngle(M_PI / 4);
                  gp3.setMinimumAngle(M_PI / 18);
                  gp3.setMaximumAngle(2 * M_PI / 3);
                  gp3.setNormalConsistency(false);
                  gp3.setInputCloud(cloudNormals);
                  gp3.setSearchMethod(searchTreeNormals);

                  gp3.reconstruct(meshPcl);

                  progressVal = 0.9;

                  vtkSmartPointer<vtkPolyData> meshVtk;
                  pcl::VTKUtils::mesh2vtk(meshPcl, meshVtk);

                  vtkNew<vtkPolyDataMapper> meshMapper;
                  meshMapper->SetInputData(meshVtk);
                  mesh_->SetMapper(meshMapper);
                  mesh_->GetProperty()->SetRepresentationToSurface();

                  progressVal = 1;
              });

              while(progressVal < 1) {
                  progressVal_ = progressVal;
                  emit progressValChanged();

                  std::this_thread::sleep_for(std::chrono::milliseconds(300));
              }

              progressVal_ = 1;
              emit progressValChanged();

              if(workThread.joinable()) {
                  workThread.join();
              }
          });
}

void VTKProcessEngine::colorizeCloud(QColor color) {
          if(!cloud_->GetMapper()) {
             return;
          }

          auto r = color.redF();
          auto g = color.greenF();
          auto b = color.blueF();
          cloud_->GetProperty()->SetColor(r, g, b);

          cloud_->GetMapper()->SetScalarVisibility(false);

          curItem_->update();
}

void VTKProcessEngine::cancelColorizeCloud() {
          if(!cloud_->GetMapper()) {
             return;
          }

          cloud_->GetMapper()->SetScalarVisibility(true);
          auto lookUp = static_cast<vtkLookupTable*>(cloud_->GetMapper()->GetLookupTable());
          lookUp->DeepCopy(lookupPre);

          curItem_->update();
}

void VTKProcessEngine::jetDepthColorMap() {
          if(!cloud_->GetMapper()) {
             return;
          }

          vtkNew<vtkDataSetSurfaceFilter> surfaceFilter;
          surfaceFilter->SetInputData(cloud_->GetMapper()->GetInput());
          surfaceFilter->Update();

          double zMin = surfaceFilter->GetOutput()->GetBounds()[4];
          double zMax = surfaceFilter->GetOutput()->GetBounds()[5];

          auto points = vtkUnstructuredGrid::SafeDownCast(cloud_->GetMapper()->GetInput())->GetPoints();
          vtkNew<vtkLookupTable> jetMapLookup;
          jetMapLookup->SetHueRange(0.67, 0.0);
          jetMapLookup->SetTableRange(zMin, zMax);
          jetMapLookup->Build();

          auto lookUp = static_cast<vtkLookupTable*>(cloud_->GetMapper()->GetLookupTable());
          lookupPre->DeepCopy(lookUp);

          auto scalars = vtkFloatArray::SafeDownCast(vtkUnstructuredGrid::SafeDownCast(vtkDataSetMapper::SafeDownCast(cloud_->GetMapper())->GetInput())->GetPointData()->GetScalars());

          for (size_t i = 0; i < scalars->GetNumberOfTuples(); ++i) {
             double tabColor[3];
             jetMapLookup->GetColor(points->GetPoint(i)[2], tabColor);
             lookUp->SetTableValue(scalars->GetValue(i), tabColor[0], tabColor[1], tabColor[2]);
          }

          cloud_->GetMapper()->SetScalarVisibility(true);
          cloud_->GetProperty()->SetRepresentationToPoints();

          curItem_->update();
}

void VTKProcessEngine::setCameraViewPort(double x, double y, double z, double fx, double fy, double fz, double vx, double vy, double vz) {
          vtkNew<vtkCamera> camera;

          if(cloud_->GetMapper()) {
             vtkNew<vtkDataSetSurfaceFilter> surfaceFilter;
             surfaceFilter->SetInputData(cloud_->GetMapper()->GetInput());
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
    if(!mesh_->GetMapper()) {
        return;
    }

    isEnable ? mesh_->VisibilityOn() : mesh_->VisibilityOff();

    curItem_->update();
}

void VTKProcessEngine::enableCloud(const bool isEnable) {
    if(!cloud_->GetMapper()) {
        return;
    }

    isEnable ? cloud_->VisibilityOn() : cloud_->VisibilityOff();

    curItem_->update();
}
