/**
 * @file defocusMethod.hpp
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __DEFOCUSMETHOD_H_
#define __DEFOCUSMETHOD_H_

#include "../../common.h"

#include <opencv2/opencv.hpp>

namespace slmaster {
namespace algorithm {
void twoDimensionErrorExpand(cv::Mat &img);

void binary(cv::Mat &img);

void opwm(cv::Mat &img, const int cycles, const float shiftVal,
          const bool isHonrizon);

} // namespace algorithm
} // namespace slmaster
#endif // !__DEFOCUSMETHOD_H_
