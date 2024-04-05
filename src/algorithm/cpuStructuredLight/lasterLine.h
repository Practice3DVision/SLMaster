/**
 * @file lasterLine.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __LASER_LINE_H_
#define __LASER_LINE_H_

#include "../../common.h"

#include <opencv2/opencv.hpp>

namespace slmaster {
namespace algorithm {
    /**
     * @brief Steger中心线提取方法
     * 
     * @param img       待提取中心线的图片
     * @param outPoints 中心线提取结果
     * @param mask      ROI
     */
    void SLMASTER_API stegerExtract(const cv::Mat& img, std::vector<cv::Point2f>& outPoints, cv::InputArray& mask = cv::noArray());
}
}

#endif //!__LASER_LINE_H_