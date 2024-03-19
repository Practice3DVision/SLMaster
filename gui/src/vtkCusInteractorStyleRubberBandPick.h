#ifndef vtkCusInteractorStyleRubberBandPick_h
#define vtkCusInteractorStyleRubberBandPick_h

#include "vtkRenderItem.h"
#include "vtkProcessEngine.h"

#include "vtkInteractionStyleModule.h" // For export macro
#include "vtkInteractorStyleTrackballCamera.h"

#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkProperty.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
/**
 * @file vtkCusInteractorStyleRubberBandPick.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <vtkPolyDataMapper.h>
#include <vtkCellData.h>
#include <vtkVertex.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyVertex.h>
#include <vtkDataSetMapper.h>
#include <vtkAreaPicker.h>
#include <vtkPlanes.h>
#include <vtkExtractGeometry.h>
#include <vtkProp3DCollection.h>
#include <vtkClipDataSet.h>

#include <QObject>
#include <QString>
#include <QVariant>

class vtkUnsignedCharArray;

class VTKProcessEngine;

class vtkCusInteractorStyleRubberBandPick
  : public vtkInteractorStyleTrackballCamera
{
public:
  static vtkCusInteractorStyleRubberBandPick* New();
  vtkTypeMacro(vtkCusInteractorStyleRubberBandPick, vtkInteractorStyleTrackballCamera);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void clip(const bool isClipInner, double& progressVal);

  void cancelClip();

  void StartSelect();

  void stopSelect();

  void enablePointInfoMode(const bool isEnable);

  void bindRenderItem(VTKRenderItem* renderItem);

  void bindVtkProcessEngine(VTKProcessEngine* engine);

  void bindCloudActor(vtkActor* cloudActor);
  ///@{
  /**
   * Event bindings
   */
  void OnMouseMove() override;
  void OnLeftButtonDown() override;
  void OnLeftButtonUp() override;
  void OnRightButtonDown() override;
  ///@}

protected:
  vtkCusInteractorStyleRubberBandPick();
  ~vtkCusInteractorStyleRubberBandPick() override;

  virtual void Pick();

  int StartPosition[2];
  int EndPosition[2];

  int Moving;
  int CurrentMode;

private:
  vtkCusInteractorStyleRubberBandPick(const vtkCusInteractorStyleRubberBandPick&) = delete;
  void operator=(const vtkCusInteractorStyleRubberBandPick&) = delete;
  void singlePick();
  void areaPick();
  vtkNew<vtkActor> highlightPoint_;
  vtkNew<vtkActor> areaSelectedPoints_;
  VTKRenderItem* renderItem_;
  VTKProcessEngine* processEngine_;
  vtkActor* cloudActor_;
  vtkNew<vtkUnstructuredGrid> cloudData_;
  bool pointInfoEnable_;
};

#endif //!vtkCusInteractorStyleRubberBandPick_h
