#ifndef __SLCAMERA_FACTORY_H_
#define __SLCAMERA_FACTORY_H_

#include "common.h"

#include "slCamera.h"
#include "binoocularCamera.h"

namespace slmaster{

enum CameraType {
    Monocular = 0,
    Binocular,
    Triple
};

class SLMASTER_API SLCameraFactory {
  public:
    std::shared_ptr<SLCamera> getCamera(const CameraType cameraType);
    void setCameraJsonPath(const std::string& path);
  private:
    bool needUpdate_;
    std::string jsonPath_;
    std::shared_ptr<SLCamera> camera_;
    CameraType curCameraType_;
};
}

#endif // __SLCAMERAFACTORY_H_
