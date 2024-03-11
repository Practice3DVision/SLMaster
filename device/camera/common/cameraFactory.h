#ifndef __CAMERA_FACTORY_H_
#define __CAMERA_FACTORY_H_

#include <string>
#include <unordered_map>

#include "huarayCamera.h"

/** @brief 结构光库 **/
namespace device {
/** @brief 相机库 **/
namespace camera {
/** @brief 相机工厂 **/
class DEVICE_API CameraFactory {
  public:
    CameraFactory() { };
    /**@brief 制造商*/
    enum CameraManufactor {
        Huaray = 0, //华睿科技
        Halcon      //海康机器人
    };

    Camera *getCamera(std::string cameraUserId,
                            CameraManufactor manufactor) {
        Camera *camera = nullptr;

        if (cameras_.count(cameraUserId)) {
            return cameras_[cameraUserId];
        } else {
            if (Huaray == manufactor) {
                camera = new HuarayCammera(cameraUserId);
                cameras_[cameraUserId] = camera;
            }
            // TODO@LiuYunhuang:增加海康相机支持
            else if (Halcon == manufactor) {
                camera = new HuarayCammera(cameraUserId);
                cameras_[cameraUserId] = camera;
            }
        }

        return camera;
    }
  private:
    std::unordered_map<std::string, Camera *> cameras_;
}; // class CameraFactory
} // namespace camera
} // namespace sl

#endif //__CAMERA_FACTORY_H_
