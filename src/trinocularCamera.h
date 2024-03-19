/**
 * @file trinocularCamera.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __TRINOCULAR_CAMERA_H_
#define __TRINOCULAR_CAMERA_H_

#include "common.h"
#include "slCamera.h"
#include "cameraFactory.h"
#include "projectorFactory.h"

#include <unordered_map>

namespace slmaster {
/** @brief 结构光相机 */
class SLMASTER_API TrinocularCamera : public SLCamera {
  public:
    /**
     * @brief 使用配置文件加载相机配置
     *
     * @param jsonPath json文件路径
     */
    TrinocularCamera(IN const std::string jsonPath);
    /**
     * @brief 结束时保存当前参数
     */
    ~TrinocularCamera();
    /**
     * @brief 获取相机信息
     *
     * @return SLCameraInfo 相机信息
     */
    SLCameraInfo getCameraInfo() override;
    /**
     * @brief 连接
     *
     * @return true 成功
     * @return false 失败
     */
    bool connect() override;
    /**
     * @brief 断开连接
     *
     * @return true 成功
     * @return false 失败
     */
    bool disConnect() override;
    /**
     * @brief 查询连接状态
     *
     * @return true 已连接
     * @return false 已断连
     */
    bool isConnect() override;
    /**
     * @brief 获取相机工厂
     * @return
     */
    device::camera::CameraFactory* getCameraFactory() override { return &cameraFactory_; }
    /**
     * @brief 获取投影仪工厂
     * @return
     */
    device::projector::ProjectorFactory* getProjectorFactory() override { return &projectorFactory_; }
    /**
     * @brief 烧录条纹
     * @param imgs 待烧录的条纹
     * @return
     */
    bool burnPatterns(const std::vector<cv::Mat>& imgs) override;
    /**
     * @brief 连续捕获
     * @param frameData 获取到的数据
     * @return
     */
    bool continuesCapture(SafeQueue<FrameData>& frameDataQueue) override;
    /**
     * @brief 停止连续捕获
     * @return
     */
    bool stopContinuesCapture() override;
    /**
     * @brief 获取一帧数据
     *
     * @param frameData 获取到的数据
     * @return true 成功
     * @return false 失败
     */
    bool capture(IN FrameData &frameData) override;
    /**
     * @brief 获取一帧数据(离线)
     *
     * @param leftImgs 输入的左相机图片
     * @param rightImgs 输入的右相机图片
     * @param frameData 获取到的数据
     * @return true 成功
     * @return false 失败
     */
    bool offlineCapture(IN const std::vector<std::vector<cv::Mat>>& leftImgs, OUT FrameData &frameData) override;
    /**
     * @brief 是否使能深度相机
     *
     * @param isEnable 使能标志位
     * @return true 成功
     * @return false 失败
     */
    bool setDepthCameraEnabled(IN const bool isEnable) override;
    /**
     * @brief 获取字符属性值
     *
     * @param attributeName 字符属性名称
     * @param val 当前字符属性值
     * @return true 成功
     * @return false 失败
     */
    bool getStringAttribute(IN const std::string attributeName,
                            OUT std::string &val) override;
    /**
     * @brief 获取数字属性值
     *
     * @param attributeName 数字属性名称
     * @param val 当前数字属性值
     * @return true 成功
     * @return false 失败
     */
    bool getNumbericalAttribute(IN const std::string attributeName,
                                OUT double &val) override;
    /**
     * @brief 获取布尔属性值
     *
     * @param attributeName 布尔属性名称
     * @param val 当前布尔属性值
     * @return true 成功
     * @return false 失败
     */
    bool getBooleanAttribute(IN const std::string attributeName,
                             OUT bool &val) override;
    /**
     * @brief 设置字符属性值
     *
     * @param attributeName 字符属性名称
     * @param val 需要设置的值
     * @return true 成功
     * @return false 失败
     */
    bool setStringAttribute(IN const std::string attributeName,
                            IN const std::string val) override;
    /**
     * @brief 设置数字属性值
     *
     * @param attributeName 数字属性名称
     * @param val 需要设置的值
     * @return true 成功
     * @return false 失败
     */
    bool setNumberAttribute(IN const std::string attributeName,
                            IN const double val) override;
    /**
     * @brief 设置布尔属性值
     *
     * @param attributeName 布尔属性名称
     * @param val 需要设置的值
     * @return true 成功
     * @return false 失败
     */
    bool setBooleanAttribute(IN const std::string attributeName,
                             IN const bool val) override;
    /**
     * @brief 重置相机配置
     *
     * @return true 成功
     * @return false 失败
     */
    bool resetCameraConfig() override;
    /**
     * @brief 更新相机
     *
     * @return true 成功
     * @return false 失败
     */
    bool updateCamera() override;
  private:
    /**
     * @brief 读取配置文件
     *
     * @param jsonPath 配置文件路径
     * @param jsonVal 配置值
     * @return true 成功
     * @return false 失败
     */
    bool loadParams(IN const std::string jsonPath, IN Json::Value &jsonVal);
    /**
     * @brief 写入配置文件
     *
     * @param jsonPath 配置文件路径
     * @param jsonVal 配置
     * @return true 成功
     * @return false 失败
     */
    bool saveParams(IN const std::string jsonPath, IN Json::Value &jsonVal);
    /**
     * @brief 解析数组
     *
     * @param jsonVal 配置值
     * @param isWrite 是否写入
     */
    void parseArray(IN Json::Value &jsonVal, IN const bool isWrite);
    /**
     * @brief 更新加速方法
     *
     */
    void updateAlgorithmMethod();
    /**
     * @brief 更新曝光时间
     *
     */
    void updateExposureTime();
    /**
     * @brief 更新深度相机使能
     *
     */
    void updateEnableDepthCamera();
    /**
     * @brief 更新投影仪亮度
     *
     */
    void updateLightStrength();
    /**
     * @brief 解码并重建
     */
    void decode(const std::vector<std::vector<cv::Mat>> &imgs,
                FrameData &frameData);
    /**
     * @brief 处理信号
     */
    void parseSignals();
    bool isInitial_;
    std::string jsonPath_;
    Json::Value jsonVal_;
    SLCameraInfo cameraInfo_;
    std::unordered_map<std::string, std::string> stringProperties_;
    std::unordered_map<std::string, float> numbericalProperties_;
    std::unordered_map<std::string, bool> booleanProperties_;
    std::unordered_map<std::string, bool> propertiesChangedSignals_;
    device::camera::CameraFactory cameraFactory_;
    device::projector::ProjectorFactory projectorFactory_;
    std::atomic_bool isCaptureStop_;
    std::thread imgCreateThread_;
    std::thread frameDataCreateThread_;
    SafeQueue<std::vector<std::vector<cv::Mat>>> imgsCreated_;
};
} // namespace sl

#endif // __TRINOCULAR_CAMERA_H_
