#include "binosSinusCompleGrayCodePattern.h"

#include "../algorithm/algorithm.h"

using namespace cv;

namespace slmaster {
namespace cameras {

static BinoSinusCompleGrayCodePattern::Params params__;

BinoSinusCompleGrayCodePattern::BinoSinusCompleGrayCodePattern() {}

bool BinoSinusCompleGrayCodePattern::generate(
    std::vector<cv::Mat> &imgs) const {
    algorithm::SinusCompleGrayCodePattern::Params params;
    params.width = params__.width_;
    params.height = params__.height_;
    params.nbrOfPeriods = params__.cycles_;
    params.horizontal = params__.horizontal_;
    params.maxCost = params__.maxCost_;
    params.minDisparity = params__.minDisparity_;
    params.maxDisparity = params__.maxDisparity_;
    params.confidenceThreshold = params__.confidenceThreshold_;
    params.shiftTime = params__.shiftTime_;

    return algorithm::SinusCompleGrayCodePattern::create(params)
        ->generate(imgs);
}

std::shared_ptr<Pattern> BinoSinusCompleGrayCodePattern::create(const Params& params) {
    params__ = params;

    return std::make_shared<BinoSinusCompleGrayCodePattern>();
}

bool BinoSinusCompleGrayCodePattern::decode(
    const std::vector<std::vector<cv::Mat>> &patternImages,
    cv::Mat &disparityMap, const bool isGpu) const {
    CV_Assert(patternImages.size() >= 2);

#ifdef OPENCV_WITH_CUDA_MODULE
    if (isGpu) {
        algorithm::SinusCompleGrayCodePatternGPU::Params params;
        params.shiftTime = params__.shiftTime_;
        params.confidenceThreshold = params__.confidenceThreshold_;
        params.costMinDiff = params__.costMinDiff_;
        params.maxCost = params__.maxCost_;
        params.height = params__.height_;
        params.width = params__.width_;
        params.nbrOfPeriods = params__.cycles_;
        params.horizontal = params__.horizontal_;
        params.minDisparity = params__.minDisparity_;
        params.maxDisparity = params__.maxDisparity_;

        auto pattern = algorithm::SinusCompleGrayCodePatternGPU::create(params);
        return pattern->decode(patternImages, disparityMap,
                               algorithm::SINUSOIDAL_COMPLEMENTARY_GRAY_CODE_GPU);
    }
#endif

    algorithm::SinusCompleGrayCodePattern::Params params;
    params.shiftTime = params__.shiftTime_;
    params.width = params__.width_;
    params.height = params__.height_;
    params.nbrOfPeriods = params__.cycles_;
    params.horizontal = params__.horizontal_;
    params.maxCost = params__.maxCost_;
    params.minDisparity = params__.minDisparity_;
    params.maxDisparity = params__.maxDisparity_;
    params.confidenceThreshold = params__.confidenceThreshold_;

    auto pattern = algorithm::SinusCompleGrayCodePattern::create(params);
    return pattern->decode(
        patternImages, disparityMap, cv::noArray(), cv::noArray(),
        algorithm::SINUSOIDAL_COMPLEMENTARY_GRAY_CODE);
}
} // namespace cameras
} // namespace slmaster