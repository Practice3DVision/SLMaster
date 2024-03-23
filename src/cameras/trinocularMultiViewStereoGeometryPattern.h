/**
 * @file trinocularMultiViewStereoGeometryPattern.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __TRINOCULAR_MULTIVIEW_STEREO_GEOMETRY_PATTERN_H_
#define __TRINOCULAR_MULTIVIEW_STEREO_GEOMETRY_PATTERN_H_

#include "../common.h"
#include "pattern.h"

namespace slmaster {
namespace cameras {
// TODO@Evans
// Liu:使用修饰器模式会更好，避免未来方法增多导致的子类爆炸，需要在OpenCV中重新更改接口
class SLMASTER_API TrinocularMultiViewStereoGeometryPattern : public Pattern {
  public:
    TrinocularMultiViewStereoGeometryPattern();
    virtual bool generate(IN std::vector<cv::Mat> &imgs) const override final;
    virtual bool
    decode(IN const std::vector<std::vector<cv::Mat>> &patternImages,
           OUT cv::Mat &depthMap, IN const bool isGpu) const override final;

  private:
};
} // namespace cameras
} // namespace slmaster

#endif // __TRINOCULAR_MULTIVIEW_STEREO_GEOMETRY_PATTERN_H_
