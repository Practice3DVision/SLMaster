/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkInteractorStyleRubberBandPick.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkCusInteractorStyleRubberBandPick.h"

#include "vtkAbstractPropPicker.h"
#include "vtkAreaPicker.h"
#include "vtkAssemblyPath.h"
#include "vtkObjectFactory.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkUnsignedCharArray.h"
#include "vtkPointPicker.h"
#include "vtkVertexGlyphFilter.h"
#include "vtkPointData.h"
#include "vtkMapper.h"
#include "vtkDataSet.h"
#include "vtkSmartPointer.h"
#include "vtkCallbackCommand.h"
#include "vtkFloatArray.h"
#include "vtkMergeCells.h"
#include "vtkLookupTable.h"

vtkStandardNewMacro(vtkCusInteractorStyleRubberBandPick);

#define VTKISRBP_ORIENT 0
#define VTKISRBP_SELECT 1

void progressCallbackFunc(vtkObject* obj,unsigned long eid,void* clientdata,void *calldata) {
    double* progressVal = static_cast<double*>(clientdata);
    *progressVal = *(static_cast<double*>(calldata));
}

//------------------------------------------------------------------------------
vtkCusInteractorStyleRubberBandPick::vtkCusInteractorStyleRubberBandPick() : pointInfoEnable_(false)
{
  this->CurrentMode = VTKISRBP_ORIENT;
  this->StartPosition[0] = this->StartPosition[1] = 0;
  this->EndPosition[0] = this->EndPosition[1] = 0;
  this->Moving = 0;
}

//------------------------------------------------------------------------------
vtkCusInteractorStyleRubberBandPick::~vtkCusInteractorStyleRubberBandPick()
{

}

//------------------------------------------------------------------------------
void vtkCusInteractorStyleRubberBandPick::StartSelect()
{
  this->CurrentMode = VTKISRBP_SELECT;
}

void vtkCusInteractorStyleRubberBandPick::stopSelect()
{
  this->CurrentMode = VTKISRBP_ORIENT;
}

//------------------------------------------------------------------------------
void vtkCusInteractorStyleRubberBandPick::OnLeftButtonDown()
{
  if (this->CurrentMode != VTKISRBP_SELECT)
  {
    // if not in rubber band mode, let the parent class handle it
    this->Superclass::OnLeftButtonDown();

    if(pointInfoEnable_) {
        this->Pick();
    }

    return;
  }

  if (!this->Interactor)
  {
    return;
  }
  this->Moving = 1;
  this->StartPosition[0] = this->Interactor->GetEventPosition()[0];
  this->StartPosition[1] = this->Interactor->GetEventPosition()[1];

  this->FindPokedRenderer(this->StartPosition[0], this->StartPosition[1]);
}

//------------------------------------------------------------------------------
void vtkCusInteractorStyleRubberBandPick::OnMouseMove()
{
  if (this->CurrentMode != VTKISRBP_SELECT)
  {
    // if not in rubber band mode,  let the parent class handle it
    this->Superclass::OnMouseMove();
    return;
  }
}

//------------------------------------------------------------------------------
void vtkCusInteractorStyleRubberBandPick::OnLeftButtonUp()
{
    this->Superclass::OnLeftButtonUp();
    return;
}

void vtkCusInteractorStyleRubberBandPick::OnRightButtonDown() {
    if (this->CurrentMode != VTKISRBP_SELECT)
    {
      // if not in rubber band mode,  let the parent class handle it
      this->Superclass::OnRightButtonDown();
      return;
    }

    if (!this->Interactor || !this->Moving)
    {
      return;
    }

    this->EndPosition[0] = this->Interactor->GetEventPosition()[0];
    this->EndPosition[1] = this->Interactor->GetEventPosition()[1];
    const int* size = this->Interactor->GetRenderWindow()->GetSize();
    if (this->EndPosition[0] > (size[0] - 1))
    {
      this->EndPosition[0] = size[0] - 1;
    }
    if (this->EndPosition[0] < 0)
    {
      this->EndPosition[0] = 0;
    }
    if (this->EndPosition[1] > (size[1] - 1))
    {
      this->EndPosition[1] = size[1] - 1;
    }
    if (this->EndPosition[1] < 0)
    {
      this->EndPosition[1] = 0;
    }

    // otherwise record the rubber band end coordinate and then fire off a pick
    if ((this->StartPosition[0] != this->EndPosition[0]) ||
      (this->StartPosition[1] != this->EndPosition[1]))
    {
      this->Pick();
    }
    this->Moving = 0;
}

//------------------------------------------------------------------------------
void vtkCusInteractorStyleRubberBandPick::Pick()
{
  if (this->CurrentMode == VTKISRBP_SELECT)
  {
    areaPick();
  }
  else {
    singlePick();
  }

  renderItem_->update();
  //this->Interactor->Render();
}

//------------------------------------------------------------------------------
void vtkCusInteractorStyleRubberBandPick::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void vtkCusInteractorStyleRubberBandPick::singlePick() {
    int* clickPos = this->GetInteractor()->GetEventPosition();
    vtkNew<vtkPointPicker> pointPicker;
    pointPicker->Pick(clickPos[0], clickPos[1], 0, this->GetDefaultRenderer());
    int idx = pointPicker->GetPointId();

    if(idx >= 0) {
        double point[3];
        pointPicker->GetDataSet()->GetPoint(idx, point);
        //statusBar_->setProperty("labelText", QString::fromStdString("Pos: " + std::to_string(point[0]) + ", " + std::to_string(point[1]) + ", " + std::to_string(point[2])));
        emit VTKProcessEngine::instance()->pointInfoChanged(point[0], point[1], point[2]);

        vtkNew<vtkPoints> points;
        vtkNew<vtkPolyVertex> polyVertex;
        vtkNew<vtkUnstructuredGrid> grid;
        polyVertex->GetPointIds()->SetNumberOfIds(1);

        vtkIdType pid[1];
        pid[0] =  points->InsertNextPoint(point[0], point[1], point[2]);
        polyVertex->GetPointIds()->SetId(0, 0);

        grid->Allocate(1, 1);
        grid->SetPoints(points);
        grid->InsertNextCell(polyVertex->GetCellType(), polyVertex->GetPointIds());

        vtkNew<vtkDataSetMapper> mapper;
        mapper->SetInputData(grid);

        highlightPoint_->SetMapper(mapper);
        highlightPoint_->GetProperty()->SetRepresentationToPoints();
        highlightPoint_->GetProperty()->SetPointSize(10);
        highlightPoint_->GetProperty()->SetColor(1, 0, 0);
        highlightPoint_->VisibilityOn();
    }
    else {
        //statusBar_->setProperty("labelText", QString::fromStdString("Pos: null, null, null"));
        highlightPoint_->VisibilityOff();
    }
}

void vtkCusInteractorStyleRubberBandPick::bindRenderItem(VTKRenderItem* renderItem) {
    renderItem_ = renderItem;
    renderItem_->renderer()->AddActor(highlightPoint_);
    renderItem_->renderer()->AddActor(areaSelectedPoints_);
}

void vtkCusInteractorStyleRubberBandPick::areaPick() {
    double rbcenter[3];
    const int* size = this->Interactor->GetRenderWindow()->GetSize();
    int min[2], max[2];
    min[0] =
      this->StartPosition[0] <= this->EndPosition[0] ? this->StartPosition[0] : this->EndPosition[0];
    if (min[0] < 0)
    {
      min[0] = 0;
    }
    if (min[0] >= size[0])
    {
      min[0] = size[0] - 2;
    }

    min[1] =
      this->StartPosition[1] <= this->EndPosition[1] ? this->StartPosition[1] : this->EndPosition[1];
    if (min[1] < 0)
    {
      min[1] = 0;
    }
    if (min[1] >= size[1])
    {
      min[1] = size[1] - 2;
    }

    max[0] =
      this->EndPosition[0] > this->StartPosition[0] ? this->EndPosition[0] : this->StartPosition[0];
    if (max[0] < 0)
    {
      max[0] = 0;
    }
    if (max[0] >= size[0])
    {
      max[0] = size[0] - 2;
    }

    max[1] =
      this->EndPosition[1] > this->StartPosition[1] ? this->EndPosition[1] : this->StartPosition[1];
    if (max[1] < 0)
    {
      max[1] = 0;
    }
    if (max[1] >= size[1])
    {
      max[1] = size[1] - 2;
    }

    rbcenter[0] = (min[0] + max[0]) / 2.0;
    rbcenter[1] = (min[1] + max[1]) / 2.0;
    rbcenter[2] = 0;

    if (cloudActor_->GetMapper())
    {
      vtkAreaPicker* areaPicker = vtkAreaPicker::SafeDownCast(this->Interactor->GetPicker());
      if (areaPicker != nullptr)
      {
        areaPicker->AreaPick(min[0], min[1], max[0], max[1], this->CurrentRenderer);

        vtkNew<vtkClipDataSet> __clipper;
        __clipper->SetClipFunction(areaPicker->GetFrustum());
        __clipper->SetInputData(cloudActor_->GetMapper()->GetInput());
        __clipper->SetValue(0.0);
        __clipper->SetInsideOut(true);
        __clipper->Update();

        vtkNew<vtkDataSetMapper> highLightMapper;
        highLightMapper->SetInputData(__clipper->GetOutput());
        highLightMapper->SetScalarVisibility(false);
        areaSelectedPoints_->SetMapper(highLightMapper);
        areaSelectedPoints_->GetProperty()->SetColor(244.0 / 255.0, 143.0 / 255.0, 177.0 / 255.0);
        areaSelectedPoints_->GetProperty()->SetRepresentationToPoints();
        areaSelectedPoints_->GetProperty()->SetPointSize(2);
        areaSelectedPoints_->VisibilityOn();

        renderItem_->update();
      }
    }
}

void vtkCusInteractorStyleRubberBandPick::clip(const bool isClipInner, double& progressVal) {
    if (!cloudActor_->GetMapper()) {
        return;
    }

    vtkAreaPicker* areaPicker = vtkAreaPicker::SafeDownCast(this->Interactor->GetPicker());
    vtkNew<vtkClipDataSet> clipper;
    clipper->SetClipFunction(areaPicker->GetFrustum());
    clipper->SetInputData(cloudActor_->GetMapper()->GetInput());
    clipper->SetValue(0.0);
    clipper->SetInsideOut(isClipInner);

    vtkNew<vtkCallbackCommand> progressCommand;
    progressCommand->SetCallback(progressCallbackFunc);
    progressCommand->SetClientData(&progressVal);
    clipper->AddObserver(vtkCommand::ProgressEvent, progressCommand);

    clipper->Update();

    vtkDataSetMapper::SafeDownCast(cloudActor_->GetMapper())->SetInputData(clipper->GetOutput());

    areaSelectedPoints_->VisibilityOff();
}

void vtkCusInteractorStyleRubberBandPick::bindCloudActor(vtkActor* cloudActor) {
    cloudActor_ = cloudActor;
}

void vtkCusInteractorStyleRubberBandPick::bindVtkProcessEngine(VTKProcessEngine* engine) {
    processEngine_ = engine;
}

void vtkCusInteractorStyleRubberBandPick::cancelClip() {
    areaSelectedPoints_->VisibilityOff();
}

void vtkCusInteractorStyleRubberBandPick::enablePointInfoMode(const bool isEnable) {
    pointInfoEnable_ = isEnable;

    if(!pointInfoEnable_) {
        highlightPoint_->VisibilityOff();
    }
}
