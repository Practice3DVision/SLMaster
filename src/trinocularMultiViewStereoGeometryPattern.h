#ifndef __TRINOCULAR_MULTIVIEW_STEREO_GEOMETRY_PATTERN_H_
#define __TRINOCULAR_MULTIVIEW_STEREO_GEOMETRY_PATTERN_H_

#include "common.h"
#include "pattern.h"

namespace slmaster {
//TODO@LiuYunhuang:使用修饰器模式会更好，避免未来方法增多导致的子类爆炸，需要在OpenCV中重新更改接口
class SLMASTER_API TrinocularMultiViewStereoGeometryPattern : public Pattern {
  public:
    TrinocularMultiViewStereoGeometryPattern();
    virtual bool generate(IN std::vector<cv::Mat>& imgs) const override final;
    virtual bool decode(IN const std::vector< std::vector<cv::Mat> >& patternImages, OUT cv::Mat& depthMap, IN const bool isGpu) const override final;
  private:
};
}

#endif // __TRINOCULAR_MULTIVIEW_STEREO_GEOMETRY_PATTERN_H_
