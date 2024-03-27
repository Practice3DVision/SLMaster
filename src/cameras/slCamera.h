/**
 * @file slCamera.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __SL_CAMERA_H_
#define __SL_CAMERA_H_

#include "../common.h"
#include "../device/device.h"
#include "caliInfo.h"
#include "json.h"
#include "pattern.h"


#include <fstream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

namespace slmaster {
namespace cameras {

typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<Point> Cloud;

/** @brief 单帧数据 */
struct SLMASTER_API FrameData {
    FrameData() : pointCloud_(new Cloud()) {}
    FrameData &operator=(const FrameData &data) {
        this->textureMap_ = data.textureMap_;
        this->depthMap_ = data.depthMap_;
        this->pointCloud_ = data.pointCloud_;
        return *this;
    }
    FrameData(const FrameData &data) {
        this->textureMap_ = data.textureMap_;
        this->depthMap_ = data.depthMap_;
        this->pointCloud_ = data.pointCloud_;
    }
    FrameData(FrameData &&data) {
        this->textureMap_ = data.textureMap_;
        this->depthMap_ = data.depthMap_;
        this->pointCloud_ = std::move(data.pointCloud_);
    }
    cv::Mat textureMap_;    // 纹理图
    cv::Mat depthMap_;      // 深度图
    Cloud::Ptr pointCloud_; // 点云
};
/** @brief 相机信息 */
struct SLMASTER_API SLCameraInfo {
    std::string cameraName_; // 相机名
    cv::Mat intrinsic_;      // 内参
    bool isFind_;            // 是否查找到该相机
};

// TODO@Evans
// Liu:将设备也修改为修饰器模式，去除结构光相机工厂模式，直接利用修饰器进行算法、设备配置
class SLMASTER_API SLCamera {
  public:
    /**
     * @brief 任何相机都应当严格的遵守json数据交换，为此拒绝其它构造方法
     *
     * @param jsonPath json文件路径
     */
    SLCamera(IN const std::string jsonPath){};
    /**
     * @brief 彻底析构
     *
     */
    virtual ~SLCamera() {}
    /**
     * @brief 获取相机信息
     *
     * @return SLCameraInfo 相机信息
     */
    virtual SLCameraInfo getCameraInfo() = 0;
    /**
     * @brief 连接
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool connect() = 0;
    /**
     * @brief 断开连接
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool disConnect() = 0;
    /**
     * @brief 查询连接状态
     *
     * @return true 已连接
     * @return false 已断连
     */
    virtual bool isConnect() = 0;
    /**
     * @brief 获取相机工厂
     * @return
     */
    virtual device::CameraFactory *getCameraFactory() = 0;
    /**
     * @brief 获取投影仪工厂
     * @return
     */
    virtual device::ProjectorFactory *getProjectorFactory() = 0;
    /**
     * @brief 设置条纹编解码方法
     * @param pattern_ 条纹编解码方法
     * @return
     */
    void setPattern(std::shared_ptr<Pattern> pattern) { pattern_ = pattern; }
    /**
     * @brief 烧录条纹
     * @param imgs 待烧录的条纹
     * @return
     */
    virtual bool burnPatterns(const std::vector<cv::Mat> &imgs) = 0;
    /**
     * @brief 获取一帧数据
     *
     * @param frameData 获取到的数据
     * @return true 成功
     * @return false 失败
     */
    virtual bool capture(OUT FrameData &frameData) = 0;
    /**
     * @brief 连续捕获
     * @param frameData 获取到的数据
     * @return
     */
    virtual bool continuesCapture(SafeQueue<FrameData> &frameDataQueue) = 0;
    /**
     * @brief 停止连续捕获
     * @return
     */
    virtual bool stopContinuesCapture() = 0;
    /**
     * @brief 获取一帧数据(离线)
     *
     * @param imgs 输入的图片
     * @param frameData 获取到的数据
     * @return true 成功
     * @return false 失败
     */
    virtual bool
    offlineCapture(IN const std::vector<std::vector<cv::Mat>> &imgs,
                   OUT FrameData &frameData) = 0;
    /**
     * @brief 是否使能深度相机(@note
     * 尽管我们可以使用setStringAttribute，但在此处强制重写的原因是如果没有关闭深度相机接口，进行手眼标定等只需纹理场合较为麻烦)
     *
     * @param isEnable 使能标志位
     * @return true 成功
     * @return false 失败
     */
    virtual bool setDepthCameraEnabled(const bool isEnable) = 0;
    /**
     * @brief 获取字符属性值
     *
     * @param attributeName 字符属性名称
     * @param val 当前字符属性值
     * @return true 成功
     * @return false 失败
     */
    virtual bool getStringAttribute(IN const std::string attributeName,
                                    OUT std::string &val) = 0;
    /**
     * @brief 获取数字属性值
     *
     * @param attributeName 数字属性名称
     * @param val 当前数字属性值
     * @return true 成功
     * @return false 失败
     */
    virtual bool getNumbericalAttribute(IN const std::string attributeName,
                                        OUT double &val) = 0;
    /**
     * @brief 获取布尔属性值
     *
     * @param attributeName 布尔属性名称
     * @param val 当前布尔属性值
     * @return true 成功
     * @return false 失败
     */
    virtual bool getBooleanAttribute(IN const std::string attributeName,
                                     OUT bool &val) = 0;
    /**
     * @brief 设置字符属性值
     *
     * @param attributeName 字符属性名称
     * @param val 需要设置的值
     * @return true 成功
     * @return false 失败
     */
    virtual bool setStringAttribute(IN const std::string attributeName,
                                    IN const std::string val) = 0;
    /**
     * @brief 设置数字属性值
     *
     * @param attributeName 数字属性名称
     * @param val 需要设置的值
     * @return true 成功
     * @return false 失败
     */
    virtual bool setNumberAttribute(IN const std::string attributeName,
                                    IN const double val) = 0;
    /**
     * @brief 设置布尔属性值
     *
     * @param attributeName 布尔属性名称
     * @param val 需要设置的值
     * @return true 成功
     * @return false 失败
     */
    virtual bool setBooleanAttribute(IN const std::string attributeName,
                                     IN const bool val) = 0;
    /**
     * @brief 重置相机配置
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool resetCameraConfig() = 0;
    /**
     * @brief 更新相机
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool updateCamera() = 0;
    /**
     * @brief 获取标定信息
     * @return
     */
    std::shared_ptr<CaliInfo> getCaliInfo() { return caliInfo_; }

  protected:
    inline bool readJsonFile(const std::string jsonPath, Json::Value &jsonVal) {
        std::ifstream readStream(jsonPath);
        if (!readStream.is_open()) {
            printf("read json file error: can't open this file! \n");
            return false;
        }
        Json::CharReaderBuilder readerBuilder;
        std::string errs;
        return Json::parseFromStream(readerBuilder, readStream, &jsonVal,
                                     &errs);
    }
    inline bool writeJsonFile(const std::string jsonPath,
                              Json::Value &jsonVal) {
        std::ofstream outStream(jsonPath);
        Json::StyledWriter jsonWriter;
        try {
            const std::string styleStr = jsonWriter.write(jsonVal);
            outStream << styleStr;
            outStream.close();
        } catch (...) {
            printf("Json parse error! \t Path: %s \n", jsonPath.data());
            outStream.close();
            return false;
        }

        return true;
    }

    std::shared_ptr<Pattern> pattern_ = nullptr;
    std::shared_ptr<CaliInfo> caliInfo_ = nullptr;

  private:
};

} // namespace cameras
} // namespace slmaster

#endif //!__SL_CAMERA_H
