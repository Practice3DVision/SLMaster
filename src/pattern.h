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

#include "common.h"

namespace slmaster {
struct SLMASTER_API PatternParams {
    PatternParams() : width_(1920), height_(1080), shiftTime_(4), cycles_(40), horizontal_(false) {}
    int width_;
    int height_;
    int shiftTime_;
    int cycles_;
    bool horizontal_;
};

struct SLMASTER_API MonoPatternParams : public PatternParams {
    MonoPatternParams() : PatternParams(), confidenceThreshold_(5.f) {}
    float confidenceThreshold_;
};

struct SLMASTER_API BinoPatternParams : public PatternParams {
    BinoPatternParams() : PatternParams(), confidenceThreshold_(5.f), minDisparity_(0), maxDisparity_(300), maxCost_(0.1f), costMinDiff_(0.0001f) {}
    float confidenceThreshold_;
    int minDisparity_;
    int maxDisparity_;
    float maxCost_;
    float costMinDiff_;
};

struct SLMASTER_API TrinoPatternParams : public PatternParams {
    bool horizontal_;
    float confidenceThreshold_;
    float maxCost_;
    float costMinDiff_;
    float costMaxDiff_;
    float minDepth_;
    float maxDepth_;
#ifdef WITH_CUDASTRUCTUREDLIGHT_MODULE
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
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class SLMASTER_API Pattern {
  public:
    Pattern() : params_(nullptr) {}
    virtual ~Pattern() { params_.reset(); };
    virtual bool generate(std::vector<cv::Mat>& imgs) const = 0;
    virtual bool decode(IN const std::vector< std::vector<cv::Mat> >& patternImages, OUT cv::Mat& disparityMap, IN const bool isGpu) const = 0;

    std::shared_ptr<PatternParams> params_;
};
}

#endif// !__PATTERN_H_
