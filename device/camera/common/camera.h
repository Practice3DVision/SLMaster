/**
 * @file camera.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __CAMERA_H_
#define __CAMERA_H_

#include <string>
#include <queue>

#include <opencv2/opencv.hpp>

#include "typeDef.h"
#include "safeQueue.hpp"

/** @brief 结构光库 **/
namespace device {
    /** @brief 相机库 **/
    namespace camera {
        /** @brief 相机信息 **/
        struct DEVICE_API CameraInfo {
            std::string cameraKey_;      // 相机序列号
            std::string cameraUserId_;   // 相机用户名
            std::string deviceType_;     // 相机传输数据类型
            bool isFind_;
        };

        /** @brief 触发方式 **/
        enum TrigMode {
            trigContinous = 0, // 连续拉流
            trigSoftware = 1,  // 软件触发
            trigLine = 2,      // 外部触发
        };

        /** @brief 相机控制类 **/
        class DEVICE_API Camera {
            public:
                /**
                 * @brief 析构函数
                 * 
                 */
                virtual ~Camera() { }
                /**
                 * @brief 获取相机信息
                 * 
                 * @return CameraInfo 相机信息
                 */
                virtual CameraInfo getCameraInfo() = 0;
                /**
                 * @brief 连接相机
                 * 
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool connect() = 0;
                /**
                 * @brief 断开相机
                 * 
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool disConnect() = 0;
                /**
                 * @brief 获取已捕获图片
                 * 
                 * @return SafeQueue<cv::Mat>& 捕获的图片
                 */
                virtual SafeQueue<cv::Mat>& getImgs() = 0;
                /**
                 * @brief 存入图片
                 * 
                 * @param img 图片
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool pushImg(IN const cv::Mat& img) = 0;
                /**
                 * @brief 丢弃最早的一张图片
                 * 
                 * @return cv::Mat 被丢弃的图片
                 */
                virtual cv::Mat popImg() = 0;
                /**
                 * @brief 清空所有捕获的图片
                 * 
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool clearImgs() = 0;
                /**
                 * @brief 是否已连接
                 * 
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool isConnect() = 0;
                /**
                 * @brief 捕获一帧图片
                 * 
                 * @return cv::Mat 捕获到的图片
                 */
                virtual cv::Mat capture() = 0;
                /**
                 * @brief 打开相机并持续捕获
                 * 
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool start() = 0;
                /**
                 * @brief 暂停相机持续捕获
                 * 
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool pause() = 0;
                /**
                 * @brief 设置触发模式
                 * 
                 * @param mode 触发模式
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool setTrigMode(IN TrigMode mode) = 0;
                /**
                 * @brief 设置枚举属性值
                 * 
                 * @param attributeName 枚举属性名称
                 * @param val 需要设置的值
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool setEnumAttribute(IN const std::string attributeName, IN const std::string val) = 0;
                /**
                 * @brief 设置字符属性值
                 * 
                 * @param attributeName 字符属性名称
                 * @param val 需要设置的值
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool setStringAttribute(IN const std::string attributeName, IN const std::string val) = 0;
                /**
                 * @brief 设置数字属性值
                 * 
                 * @param attributeName 数字属性名称
                 * @param val 需要设置的值
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool setNumberAttribute(IN const std::string attributeName, IN const double val) = 0;
                /**
                 * @brief 设置布尔属性值
                 * 
                 * @param attributeName 布尔属性名称
                 * @param val 需要设置的值
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool setBooleanAttribute(IN const std::string attributeName, IN const bool val) = 0;
                /**
                 * @brief 获取枚举属性值
                 * 
                 * @param attributeName 枚举属性名称
                 * @param val 当前枚举量
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool getEnumAttribute(IN const std::string attributeName, OUT std::string& val) = 0;
                /**
                 * @brief 获取字符属性值
                 * 
                 * @param attributeName 字符属性名称
                 * @param val 当前字符属性值
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool getStringAttribute(IN const std::string attributeName, OUT std::string& val) = 0;
                /**
                 * @brief 获取数字属性值
                 * 
                 * @param attributeName 数字属性名称
                 * @param val 当前数字属性值
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool getNumbericalAttribute(IN const std::string attributeName, OUT double& val) = 0;
                /**
                 * @brief 获取布尔属性值
                 * 
                 * @param attributeName 布尔属性名称
                 * @param val 当前布尔属性值
                 * @return true 成功
                 * @return false 失败
                 */
                virtual bool getBooleanAttribute(IN const std::string attributeName, OUT bool& val) = 0;
                /**
                 * @brief getFps 获取帧率
                 * @return 帧率
                 */
                virtual int getFps() = 0;
            private:
        };
    }
}

#endif //__CAMERA_H_
