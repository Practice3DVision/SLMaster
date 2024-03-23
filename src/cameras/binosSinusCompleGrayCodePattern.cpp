#include "binosSinusCompleGrayCodePattern.h"

#include "../algorithm/algorithm.h"

using namespace cv;

namespace slmaster {
namespace cameras {
BinoSinusCompleGrayCodePattern::BinoSinusCompleGrayCodePattern() {
    params_.reset(new BinoPatternParams());
}

bool BinoSinusCompleGrayCodePattern::generate(
    std::vector<cv::Mat> &imgs) const {
    auto binoPatternParams = static_cast<BinoPatternParams *>(params_.get());

    algorithm::SinusCompleGrayCodePattern::Params params;
    params.width = binoPatternParams->width_;
    params.height = binoPatternParams->height_;
    params.nbrOfPeriods = binoPatternParams->cycles_;
    params.horizontal = binoPatternParams->horizontal_;
    params.maxCost = binoPatternParams->maxCost_;
    params.minDisparity = binoPatternParams->minDisparity_;
    params.maxDisparity = binoPatternParams->maxDisparity_;
    params.confidenceThreshold = binoPatternParams->confidenceThreshold_;
    params.shiftTime = binoPatternParams->shiftTime_;

    return algorithm::SinusCompleGrayCodePattern::create(params)
        ->generate(imgs);
}

bool BinoSinusCompleGrayCodePattern::decode(
    const std::vector<std::vector<cv::Mat>> &patternImages,
    cv::Mat &disparityMap, const bool isGpu) const {
    CV_Assert(patternImages.size() >= 2);

    auto binoPatternParams = static_cast<BinoPatternParams *>(params_.get());

#ifdef OPENCV_WITH_CUDA_MODULE
    if (isGpu) {
        algorithm::SinusCompleGrayCodePatternGPU::Params params;
        params.shiftTime = binoPatternParams->shiftTime_;
        params.confidenceThreshold = binoPatternParams->confidenceThreshold_;
        params.costMinDiff = binoPatternParams->costMinDiff_;
        params.maxCost = binoPatternParams->maxCost_;
        params.height = binoPatternParams->height_;
        params.width = binoPatternParams->width_;
        params.nbrOfPeriods = binoPatternParams->cycles_;
        params.horizontal = binoPatternParams->horizontal_;
        params.minDisparity = binoPatternParams->minDisparity_;
        params.maxDisparity = binoPatternParams->maxDisparity_;

        auto pattern = algorithm::SinusCompleGrayCodePatternGPU::create(params);
        return pattern->decode(patternImages, disparityMap,
                               algorithm::SINUSOIDAL_COMPLEMENTARY_GRAY_CODE_GPU);
    }
#endif

    algorithm::SinusCompleGrayCodePattern::Params params;
    params.shiftTime = binoPatternParams->shiftTime_;
    params.width = binoPatternParams->width_;
    params.height = binoPatternParams->height_;
    params.nbrOfPeriods = binoPatternParams->cycles_;
    params.horizontal = binoPatternParams->horizontal_;
    params.maxCost = binoPatternParams->maxCost_;
    params.minDisparity = binoPatternParams->minDisparity_;
    params.maxDisparity = binoPatternParams->maxDisparity_;
    params.confidenceThreshold = binoPatternParams->confidenceThreshold_;

    auto pattern = algorithm::SinusCompleGrayCodePattern::create(params);
    return pattern->decode(
        patternImages, disparityMap, cv::noArray(), cv::noArray(),
        algorithm::SINUSOIDAL_COMPLEMENTARY_GRAY_CODE);

    return true;
}
} // namespace cameras
} // namespace slmaster