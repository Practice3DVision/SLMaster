/**
 * @file binosSinusCompleGrayCodePattern.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __BINOSSINUSCOMPLEGRAYCODEPATTERN_H_
#define __BINOSSINUSCOMPLEGRAYCODEPATTERN_H_

#include "../common.h"
#include "pattern.h"

namespace slmaster {
namespace cameras {
// TODO@Evans
// Liu:使用修饰器模式会更好，避免未来方法增多导致的子类爆炸，需要在OpenCV中重新更改接口
class SLMASTER_API BinoSinusCompleGrayCodePattern : public Pattern {
  public:
    struct SLMASTER_API Params {
        Params()
            : width_(1920), height_(1080), shiftTime_(4), cycles_(40),
              horizontal_(false), confidenceThreshold_(5.f), minDisparity_(0),
              maxDisparity_(300), maxCost_(0.1f), costMinDiff_(0.0001f) {}
        int width_;
        int height_;
        int shiftTime_;
        int cycles_;
        bool horizontal_;
        float confidenceThreshold_;
        int minDisparity_;
        int maxDisparity_;
        float maxCost_;
        float costMinDiff_;
    };

    BinoSinusCompleGrayCodePattern();
    virtual bool generate(IN std::vector<cv::Mat> &imgs) const override final;
    virtual bool
    decode(IN const std::vector<std::vector<cv::Mat>> &patternImages,
           OUT cv::Mat &disparityMap, IN const bool isGpu) const override final;
    static std::shared_ptr<Pattern> create(const Params& params);
  private:
};
} // namespace cameras
} // namespace slmaster

#endif // __BINOSSINUSCOMPLEGRAYCODEPATTERN_H_
