/**
 * @file monoSinusCompleGrayCodePattern.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __MONOSINUSCOMPLEGRAYCODEPATTERN_H_
#define __MONOSINUSCOMPLEGRAYCODEPATTERN_H_

#include "../common.h"
#include "pattern.h"

namespace slmaster {
namespace cameras {
// TODO@Evans
// Liu:使用修饰器模式会更好，避免未来方法增多导致的子类爆炸，需要在OpenCV中重新更改接口
class SLMASTER_API MonoSinusCompleGrayCodePattern : public Pattern {
  public:
    struct SLMASTER_API Params {
        Params()
            : width_(1920), height_(1080), shiftTime_(4), cycles_(40),
              horizontal_(false), confidenceThreshold_(5.f) {}
        int width_;
        int height_;
        int shiftTime_;
        int cycles_;
        bool horizontal_;
        float confidenceThreshold_;
        float minDepth_;
        float maxDepth_;
        Eigen::Matrix4f PL1_;
        Eigen::Matrix4f PR4_;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    MonoSinusCompleGrayCodePattern();
    virtual bool generate(IN std::vector<cv::Mat> &imgs) const override final;
    virtual bool
    decode(IN const std::vector<std::vector<cv::Mat>> &patternImages,
           OUT cv::Mat &depthMap, IN const bool isGpu) const override final;
    static std::shared_ptr<Pattern> create(const Params& params);
  private:
};
} // namespace cameras
} // namespace slmaster

#endif // __MONOSINUSCOMPLEGRAYCODEPATTERN_H_
