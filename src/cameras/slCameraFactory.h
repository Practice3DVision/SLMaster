/**
 * @file slCameraFactory.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __SLCAMERA_FACTORY_H_
#define __SLCAMERA_FACTORY_H_

#include "../common.h"

#include "slCamera.h"

namespace slmaster {
namespace cameras {

enum CameraType { Monocular = 0, Binocular, Triple };

class SLMASTER_API SLCameraFactory {
  public:
    std::shared_ptr<SLCamera> getCamera(const CameraType cameraType);
    void setCameraJsonPath(const std::string &path);

  private:
    bool needUpdate_;
    std::string jsonPath_;
    std::shared_ptr<SLCamera> camera_;
    CameraType curCameraType_;
};
} // namespace cameras
} // namespace slmaster

#endif // __SLCAMERAFACTORY_H_
