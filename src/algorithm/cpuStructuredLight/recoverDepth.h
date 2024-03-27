#ifndef __RECOVER_DEPTH_H_
#define __RECOVER_DEPTH_H_

#include "../../common.h"

#include <Eigen/Eigen>

#include <opencv2/opencv.hpp>

namespace slmaster {
namespace algorithm {
/**
 * @brief 逆相机三角测量模型
 *
 * @param phase         绝对相位
 * @param PL            左相机投影矩阵
 * @param PR            右相机投影矩阵
 * @param minDepth      最小深度
 * @param maxDepth      最大深度
 * @param pitch         节距
 * @param depth         深度图
 * @param isHonrizon    是否为水平条纹
 */
void SLMASTER_API reverseCamera(const cv::Mat &phase, const Eigen::Matrix4f &PL,
                                const Eigen::Matrix4f &PR, const float minDepth,
                                const float maxDepth, const float pitch,
                                cv::Mat &depth, const bool isHonrizon = false);
/**
 * @brief 基于绝对相位的立体匹配
 *
 * @param leftUnwrapMap         左相机绝对相位
 * @param rightUnwrapMap        右相机绝对相位
 * @param disparityMap          视差图
 * @param minDisparity          最小视差值
 * @param maxDisparity          最大视差值
 * @param confidenceThreshold   调制度阈值
 * @param maxCost               最大相位代价
 */
void SLMASTER_API matchWithAbsphase(
    const cv::Mat &leftUnwrapMap, const cv::Mat &rightUnwrapMap,
    cv::Mat &disparityMap, const int minDisparity, const int maxDisparity,
    const float confidenceThreshold, const float maxCost);
} // namespace algorithm
} // namespace slmaster

#endif //!__RECOVER_DEPTH_H_