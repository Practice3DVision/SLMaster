/**
 * @file rotationAxisCalibration.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-04-06
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __ROTATION_AXIS_CALIBRATION_H_
#define __ROTATION_AXIS_CALIBRATION_H_

#include "calibrator.h"

#include <opencv2/opencv.hpp>

namespace slmaster {
namespace calibration {

class SLMASTER_API RotationAxisCalibration {
  public:
    void setCalibrator(Calibrator *calibrator) { calibrator_ = calibrator; };
    /**
     * @brief 标定
     *
     * @param imgs         标靶图案
     * @param intrinsic    内参
     * @param distort      畸变参数
     * @param chessSize    棋盘格尺寸
     * @param distance     棋盘格单元大小
     * @param rotatorAxis  转轴方程
     * @param useCurrentFeaturePoints 使用当前标靶坐标点
     * @return double      误差
     */
    double calibration(const std::vector<cv::Mat> &imgs,
                       const cv::Mat &intrinsic, const cv::Mat &distort,
                       const cv::Size chessSize, const float distance,
                       cv::Mat &rotatorAxis,
                       const bool useCurrentFeaturePoints);
    // 角点
    std::vector<std::vector<cv::Point2f>> chessPoints_;

  private:
    Calibrator *calibrator_;
};
} // namespace calibration
} // namespace slmaster

#endif //!__ROTATION_AXIS_CALIBRATION_H_