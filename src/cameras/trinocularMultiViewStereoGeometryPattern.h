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
    struct SLMASTER_API Params {
        Params()
            : width_(1920), height_(1080), shiftTime_(4), cycles_(40),
              horizontal_(false), confidenceThreshold_(5.f), maxCost_(0.1f),
              costMinDiff_(0.0001f) {}
        int width_;
        int height_;
        int shiftTime_;
        int cycles_;
        bool horizontal_;
        float confidenceThreshold_;
        float maxCost_;
        float costMinDiff_;
        float costMaxDiff_;
        float minDepth_;
        float maxDepth_;
#ifdef OPENCV_WITH_CUDA_MODULE
        cv::cuda::GpuMat refUnwrappedMap_;
#endif
        Eigen::Matrix3f M1_;
        Eigen::Matrix3f M2_;
        Eigen::Matrix3f M3_;
        Eigen::Matrix3f M4_;
        Eigen::Matrix3f R12_;
        Eigen::Vector3f T12_;
        Eigen::Matrix3f R13_;
        Eigen::Vector3f T13_;
        Eigen::Matrix4f PL1_;
        Eigen::Matrix4f PR2_;
        Eigen::Matrix4f PR4_;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    TrinocularMultiViewStereoGeometryPattern();
    virtual bool generate(IN std::vector<cv::Mat> &imgs) const override final;
    virtual bool
    decode(IN const std::vector<std::vector<cv::Mat>> &patternImages,
           OUT cv::Mat &depthMap, IN const bool isGpu) const override final;
    static std::shared_ptr<Pattern> create(const Params& params);
  private:
    
};
} // namespace cameras
} // namespace slmaster

#endif // __TRINOCULAR_MULTIVIEW_STEREO_GEOMETRY_PATTERN_H_
