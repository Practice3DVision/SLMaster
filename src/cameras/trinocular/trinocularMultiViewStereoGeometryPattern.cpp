#include "trinocularMultiViewStereoGeometryPattern.h"

#include "../../algorithm/algorithm.h"

using namespace cv;

namespace slmaster {
namespace cameras {

static TrinocularMultiViewStereoGeometryPattern::Params params__;

TrinocularMultiViewStereoGeometryPattern::
    TrinocularMultiViewStereoGeometryPattern() {
}

std::shared_ptr<Pattern> TrinocularMultiViewStereoGeometryPattern::create(const Params& params) {
    params__ = params;

    return std::make_shared<TrinocularMultiViewStereoGeometryPattern>();
}

bool TrinocularMultiViewStereoGeometryPattern::generate(
    std::vector<cv::Mat> &imgs) const {
#ifdef OPENCV_WITH_CUDA_MODULE

    algorithm::MultiViewStereoGeometryPatternGPU::Params params;
    params.shiftTime = params__.shiftTime_;
    params.height = params__.height_;
    params.width = params__.width_;
    params.nbrOfPeriods = params__.cycles_;
    params.horizontal = params__.horizontal_;

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

#ifdef OPENCV_WITH_CUDA_MODULE
    if (isGpu) {
        algorithm::MultiViewStereoGeometryPatternGPU::Params params;
        params.shiftTime = params__.shiftTime_;
        params.confidenceThreshold = params__.confidenceThreshold_;
        params.costMaxDiff = params__.costMaxDiff_;
        params.costMinDiff = params__.costMinDiff_;
        params.maxCost = params__.maxCost_;
        params.height = params__.height_;
        params.width = params__.width_;
        params.minDepth = params__.minDepth_;
        params.maxDepth = params__.maxDepth_;
        params.nbrOfPeriods = params__.cycles_;
        params.horizontal = params__.horizontal_;
        params.M1 = params__.M1_;
        params.M2 = params__.M2_;
        params.M3 = params__.M3_;
        params.R12 = params__.R12_;
        params.T12 = params__.T12_;
        params.R13 = params__.R13_;
        params.T13 = params__.T13_;
        params.PL1 = params__.PL1_;
        params.PR2 = params__.PR2_;
        params.PR4 = params__.PR4_;
        params.refUnwrappedMap = params__.refUnwrappedMap_;

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