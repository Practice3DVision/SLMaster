/**
 * @file hiddenMethodCalibrator.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-04-06
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __HIDDEN_METHOD_CALIBRATOR_H_
#define __HIDDEN_METHOD_CALIBRATOR_H_

#include <opencv2/opencv.hpp>
#include <vector>

#include "caliPacker.h"
#include "calibrator.h"

namespace slmaster {
namespace calibration {
/**
 * @brief 线激光光平面消隐法标定
 * 
 */
class SLMASTER_API HiddenMethodCalibrator {
  public:
    /**
     * @brief 设置标定器
     * 
     * @param calibrator 标定器 
     */
    void setCalibrator(Calibrator *calibrator) { calibrator_ = calibrator; };
    /**
     * @brief 标定
     * 
     * @param imgs         图案（无激光-有激光-无激光-有激光...） 
     * @param intrinsic    内参
     * @param distort      畸变参数
     * @param chessSize    棋盘格尺寸
     * @param distance     棋盘格单元大小
     * @param lightPlaneEq 光平面方程
     * @param useCurrentFeaturePoints 使用当前标靶坐标点
     * @return double      误差
     */
    double calibration(const std::vector<cv::Mat> &imgs,
                       const cv::Mat &intrinsic, const cv::Mat& distort, const cv::Size chessSize,
                       const float distance, cv::Mat &lightPlaneEq, const bool useCurrentFeaturePoints);
    //角点
    std::vector<std::vector<cv::Point2f>> chessPoints_;
    //光条点集
    std::vector<std::vector<cv::Point2f>> laserStrips_;
  private:
    Calibrator* calibrator_;
};
} // namespace calibration
} // namespace slmaster

#endif //!__HIDDEN_METHOD_CALIBRATOR_H_