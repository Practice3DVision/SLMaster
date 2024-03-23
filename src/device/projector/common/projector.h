/**
 * @file projector.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __PROJECTOR_H_
#define __PROJECTOR_H_

#include "typeDef.h"

#include <opencv2/opencv.hpp>

namespace slmaster {
/** @brief 设备库 */
namespace device {
/** @brief 投影仪LED灯 */
enum Illumination { Red = 0, Grren, Blue, RGB };

/** @brief 投影图案集合 */
struct DEVICE_API PatternOrderSet {
    std::vector<cv::Mat> imgs_; // 需要制作集合的图片
    // int __patternSetIndex;        //集合索引
    int patternArrayCounts_;    // 图片数组数量
    Illumination illumination_; // LED控制
    bool invertPatterns_;       // 反转图片
    bool isVertical_;           // 是否水平图片
    bool isOneBit_;             // 是否为一位深度
    int exposureTime_;          // 曝光时间(us)
    int preExposureTime_;       // 曝光前时间(us)
    int postExposureTime_;      // 曝光后时间(us)
};

/** @brief 相机信息 **/
struct DEVICE_API ProjectorInfo {
    std::string dlpEvmType_; // DLP评估模块
    int width_;              // 幅面宽度
    int height_;             // 幅面高度
    bool isFind_;            // 是否找到
};

/** @brief 投影仪控制类 */
class DEVICE_API Projector {
  public:
    virtual ~Projector() {}
    /**
     * @brief 获取投影仪信息
     *
     * @return ProjectorInfo 投影仪相关信息
     */
    virtual ProjectorInfo getInfo() = 0;
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
     * @brief 断开连接
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool isConnect() = 0;
    /**
     * @brief 从图案集制作投影序列
     *
     * @param table 投影图案集
     */
    virtual bool
    populatePatternTableData(IN std::vector<PatternOrderSet> table) = 0;
    /**
     * @brief 投影
     *
     * @param isContinue 是否连续投影
     * @return true 成功
     * @return false 失败
     */
    virtual bool project(IN const bool isContinue) = 0;
    /**
     * @brief 暂停
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool pause() = 0;
    /**
     * @brief 停止
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool stop() = 0;
    /**
     * @brief 恢复投影
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool resume() = 0;
    /**
     * @brief 投影下一帧
     * @warning 仅在步进模式下使用
     *
     * @return true
     * @return false
     */
    virtual bool step() = 0;
    /**
     * @brief 获取当前LED三色灯电流值
     *
     * @param r 红色电流值
     * @param g 绿色电流值
     * @param b 蓝色电流值
     * @return true 成功
     * @return false 失败
     */
    virtual bool getLEDCurrent(OUT double &r, OUT double &g, OUT double &b) = 0;
    /**
     * @brief 设置当前LED三色灯电流值
     *
     * @param r 红色电流值
     * @param g 绿色电流值
     * @param b 蓝色电流值
     * @return true 成功
     * @return false 失败
     */
    virtual bool setLEDCurrent(IN const double r, IN const double g,
                               IN const double b) = 0;
    /**
     * @brief 获取当前闪存图片数量
     *
     * @return int 图片数量
     */
    virtual int getFlashImgsNum() = 0;

  private:
};
} // namespace device
} // namespace slmaster

#endif //!__PROJECTOR_H_
