/**
 * @file huarayCamera.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __HUARAY_CAMERA_H_
#define __HUARAY_CAMERA_H_

#include "IMVApi.h"
#include "camera.h"
#include "safeQueue.hpp"

#include <opencv2/opencv.hpp>

/** @brief 相机库 */
namespace device {
/** @brief 设备控制库 */
namespace camera {
/** @brief 华睿相机控制类 **/
class DEVICE_API HuarayCammera : public Camera{
  public:
    explicit HuarayCammera(IN const std::string cameraUserId);
    ~HuarayCammera();
    CameraInfo getCameraInfo() override;
    bool connect() override;
    bool disConnect() override;
    SafeQueue<cv::Mat> &getImgs() override;
    bool pushImg(IN const cv::Mat &img) override;
    cv::Mat popImg() override;
    bool clearImgs() override;
    bool isConnect() override;
    cv::Mat capture() override;
    bool start() override;
    bool pause() override;
    bool setTrigMode(IN const TrigMode trigMode) override;
    bool setEnumAttribute(IN const std::string attributeName,
                          IN const std::string val) override;
    bool setStringAttribute(IN const std::string attributeName,
                            IN const std::string val) override;
    bool setNumberAttribute(IN const std::string attributeName,
                            IN const double val) override;
    bool setBooleanAttribute(IN const std::string attributeName,
                             IN const bool val) override;
    bool getEnumAttribute(IN const std::string attributeName,
                          OUT std::string &val) override;
    bool getStringAttribute(IN const std::string attributeName,
                            OUT std::string &val) override;
    bool getNumbericalAttribute(IN const std::string attributeName,
                                OUT double &val) override;
    bool getBooleanAttribute(IN const std::string attributeName,
                             OUT bool &val) override;
    int getFps() override;
    IMV_HANDLE* getHandle() { return pCamera_; }
  private:
    //相机ID
    const std::string cameraUserId_;
    //相机指针
    IMV_HANDLE *pCamera_;
    //相机捕获到的图片
    SafeQueue<cv::Mat> imgs_;
};
} // namespace camera
} // namespace device
#endif // __HUARAY_CAMERA_H_
