/**
 * @file pattern.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __PATTERN_H_
#define __PATTERN_H_

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "../common.h"

namespace slmaster {
namespace cameras {
    
class SLMASTER_API Pattern {
  public:
    virtual ~Pattern() {}
    virtual bool generate(std::vector<cv::Mat> &imgs) const = 0;
    virtual bool
    decode(IN const std::vector<std::vector<cv::Mat>> &patternImages,
           OUT cv::Mat &disparityMap, IN const bool isGpu) const = 0;
};

} // namespace cameras
} // namespace slmaster

#endif // !__PATTERN_H_
