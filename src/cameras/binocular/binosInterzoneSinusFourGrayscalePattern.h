/**
 * @file binoSInterzoneSinusFourGrayscalePattern.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __BINOS_INTERZONE_SINUS_FOUR_GRAYSCALE_PATTERN_H_
#define __BINOS_INTERZONE_SINUS_FOUR_GRAYSCALE_PATTERN_H_

#include "../../common.h"
#include "../pattern.h"

namespace slmaster {
namespace cameras {
class SLMASTER_API BinoInterzoneSinusFourGrayscalePattern : public Pattern {
  public:
    struct SLMASTER_API Params {
        Params()
            : width_(1920), height_(1080), shiftTime_(3), cycles_(16),
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

    BinoInterzoneSinusFourGrayscalePattern();
    virtual bool generate(IN std::vector<cv::Mat> &imgs) const override final;
    virtual bool
    decode(IN const std::vector<std::vector<cv::Mat>> &patternImages,
           OUT cv::Mat &disparityMap, IN const bool isGpu) const override final;
    static std::shared_ptr<Pattern> create(const Params& params);
  private:
};
} // namespace cameras
} // namespace slmaster

#endif // __BINOS_INTERZONE_SINUS_FOUR_GRAYSCALE_PATTERN_H_