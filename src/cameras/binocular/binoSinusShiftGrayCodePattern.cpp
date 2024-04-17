#include "binoSinusShiftGrayCodePattern.h"

#include "../../algorithm/algorithm.h"

using namespace cv;

namespace slmaster {
namespace cameras {

static BinoSinusShiftGrayCodePattern::Params params__;

BinoSinusShiftGrayCodePattern::BinoSinusShiftGrayCodePattern() {}

bool BinoSinusShiftGrayCodePattern::generate(std::vector<cv::Mat> &imgs) const {
    algorithm::SinusShiftGrayCodePattern::Params params;
    params.width = params__.width_;
    params.height = params__.height_;
    params.nbrOfPeriods = params__.cycles_;
    params.horizontal = params__.horizontal_;
    params.maxCost = params__.maxCost_;
    params.minDisparity = params__.minDisparity_;
    params.maxDisparity = params__.maxDisparity_;
    params.confidenceThreshold = params__.confidenceThreshold_;
    params.shiftTime = params__.shiftTime_;

    return algorithm::SinusShiftGrayCodePattern::create(params)->generate(imgs);
}

std::shared_ptr<Pattern>
BinoSinusShiftGrayCodePattern::create(const Params &params) {
    params__ = params;

    return std::make_shared<BinoSinusShiftGrayCodePattern>();
}

bool BinoSinusShiftGrayCodePattern::decode(
    const std::vector<std::vector<cv::Mat>> &patternImages,
    cv::Mat &disparityMap, const bool isGpu) const {
    CV_Assert(patternImages.size() >= 2);

#ifdef OPENCV_WITH_CUDA_MODULE
    if (isGpu) {
        /*
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
        */
        return false;
    }
#endif

    algorithm::SinusShiftGrayCodePattern::Params params;
    params.shiftTime = params__.shiftTime_;
    params.width = params__.width_;
    params.height = params__.height_;
    params.nbrOfPeriods = params__.cycles_;
    params.horizontal = params__.horizontal_;
    params.maxCost = params__.maxCost_;
    params.minDisparity = params__.minDisparity_;
    params.maxDisparity = params__.maxDisparity_;
    params.confidenceThreshold = params__.confidenceThreshold_;

    auto pattern = algorithm::SinusShiftGrayCodePattern::create(params);
    return pattern->decode(patternImages, disparityMap,
                           algorithm::SINUSOIDAL_SHIFT_GRAY_CODE);
}
} // namespace cameras
} // namespace slmaster