#ifndef __PATTERN_H_
#define __PATTERN_H_

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
    BinoPatternParams() : PatternParams(), confidenceThreshold_(5.f), minDisparity_(0), maxDisparity_(300), maxCost_(0.1f), costMinDiff_(0.0001f), costMaxDiff_(0.3f) {}
    float confidenceThreshold_;
    int minDisparity_;
    int maxDisparity_;
    float maxCost_;
    float costMinDiff_;
    float costMaxDiff_;
};

struct SLMASTER_API TrinoPatternParams : public PatternParams {
    bool horizontal_;
    float confidenceThreshold_;
    float maxCost_;
    float costMinDiff_;
    float costMaxDiff_;
    float minDepth_;
    float maxDepth_;
    cv::cuda::GpuMat refUnwrappedMap_;
    cv::cuda::GpuMat K_;
    cv::cuda::GpuMat M1_;
    cv::cuda::GpuMat M2_;
    cv::cuda::GpuMat M3_;
    cv::cuda::GpuMat D1_;
    cv::cuda::GpuMat D2_;
    cv::cuda::GpuMat D3_;
    cv::cuda::GpuMat R12_;
    cv::cuda::GpuMat T12_;
    cv::cuda::GpuMat R13_;
    cv::cuda::GpuMat T13_;
    cv::cuda::GpuMat R23_;
    cv::cuda::GpuMat T23_;
    cv::cuda::GpuMat PL2_;
    cv::cuda::GpuMat PR3_;
};

class SLMASTER_API Pattern {
  public:
    Pattern() : params_(nullptr) {}
    virtual ~Pattern() {
        if(params_) {
            delete params_;
            params_ = nullptr;
        }
    };
    virtual bool generate(std::vector<cv::Mat>& imgs) const = 0;
    virtual bool decode(IN const std::vector< std::vector<cv::Mat> >& patternImages, OUT cv::Mat& disparityMap, IN const bool isGpu) const = 0;

    PatternParams* params_;
};
}

#endif// !__PATTERN_H_
