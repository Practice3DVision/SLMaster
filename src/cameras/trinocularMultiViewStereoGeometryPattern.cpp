#include "trinocularMultiViewStereoGeometryPattern.h"

#include "../algorithm/algorithm.h"

using namespace cv;

namespace slmaster {
namespace cameras {

TrinocularMultiViewStereoGeometryPattern::
    TrinocularMultiViewStereoGeometryPattern() {
    params_.reset(new TrinoPatternParams());
}

bool TrinocularMultiViewStereoGeometryPattern::generate(
    std::vector<cv::Mat> &imgs) const {
#ifdef OPENCV_WITH_CUDA_MODULE
    auto trinoPatternParams = static_cast<TrinoPatternParams *>(params_.get());

    algorithm::MultiViewStereoGeometryPatternGPU::Params params;
    params.shiftTime = trinoPatternParams->shiftTime_;
    params.height = trinoPatternParams->height_;
    params.width = trinoPatternParams->width_;
    params.nbrOfPeriods = trinoPatternParams->cycles_;
    params.horizontal = trinoPatternParams->horizontal_;

    return algorithm::MultiViewStereoGeometryPatternGPU::create(params)
        ->generate(imgs);
#else
    return false;
#endif
}

bool TrinocularMultiViewStereoGeometryPattern::decode(
    const std::vector<std::vector<cv::Mat>> &patternImages, cv::Mat &depthMap,
    const bool isGpu) const {
    CV_Assert(patternImages.size() == 3);

    auto trinoPatternParams = static_cast<TrinoPatternParams *>(params_.get());

#ifdef OPENCV_WITH_CUDA_MODULE
    if (isGpu) {
        algorithm::MultiViewStereoGeometryPatternGPU::Params params;
        params.shiftTime = trinoPatternParams->shiftTime_;
        params.confidenceThreshold = trinoPatternParams->confidenceThreshold_;
        params.costMaxDiff = trinoPatternParams->costMaxDiff_;
        params.costMinDiff = trinoPatternParams->costMinDiff_;
        params.maxCost = trinoPatternParams->maxCost_;
        params.height = trinoPatternParams->height_;
        params.width = trinoPatternParams->width_;
        params.minDepth = trinoPatternParams->minDepth_;
        params.maxDepth = trinoPatternParams->maxDepth_;
        params.nbrOfPeriods = trinoPatternParams->cycles_;
        params.horizontal = trinoPatternParams->horizontal_;
        params.M1 = trinoPatternParams->M1_;
        params.M2 = trinoPatternParams->M2_;
        params.M3 = trinoPatternParams->M3_;
        params.R12 = trinoPatternParams->R12_;
        params.T12 = trinoPatternParams->T12_;
        params.R13 = trinoPatternParams->R13_;
        params.T13 = trinoPatternParams->T13_;
        params.PL1 = trinoPatternParams->PL1_;
        params.PR2 = trinoPatternParams->PR2_;
        params.PR4 = trinoPatternParams->PR4_;
        params.refUnwrappedMap = trinoPatternParams->refUnwrappedMap_;

        auto pattern =
            algorithm::MultiViewStereoGeometryPatternGPU::create(params);
        return pattern->decode(patternImages, depthMap,
                               algorithm::MULTI_VIEW_STEREO_GEOMETRY_GPU);
    }
#else
    std::clog
        << "TrinocularMultiViewStereoGeometryPattern is only used for GPU!"
        << std::endl;
#endif

    return false;
}
} // namespace cameras
} // namespace slmaster