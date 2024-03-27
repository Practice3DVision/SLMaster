#include "slCameraFactory.h"

#include "binoocularCamera.h"
#include "monocularCamera.h"
#include "trinocularCamera.h"

using namespace std;

namespace slmaster {
namespace cameras {
shared_ptr<SLCamera> SLCameraFactory::getCamera(const CameraType cameraType) {
    if (!camera_.get() || curCameraType_ != cameraType || needUpdate_) {
        if (cameraType == CameraType::Monocular) {
            camera_.reset(new MonocularCamera(jsonPath_));
        } else if (cameraType == CameraType::Binocular) {
            camera_.reset(new BinocularCamera(jsonPath_));
        } else if (cameraType == CameraType::Triple) {
            camera_.reset(new TrinocularCamera(jsonPath_));
        }
    }

    curCameraType_ = cameraType;
    needUpdate_ = false;

    return camera_;
}

void SLCameraFactory::setCameraJsonPath(const string &path) {
    jsonPath_ = path;
    needUpdate_ = true;
}
} // namespace cameras
} // namespace slmaster
