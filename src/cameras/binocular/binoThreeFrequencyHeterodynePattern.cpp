#include "binoThreeFrequencyHeterodynePattern.h"

#include "../../algorithm/algorithm.h"

using namespace cv;

namespace slmaster {
namespace cameras {

static BinoThreeFrequencyHeterodynePattern::Params params__;

BinoThreeFrequencyHeterodynePattern::BinoThreeFrequencyHeterodynePattern() {}

bool BinoThreeFrequencyHeterodynePattern::generate(
    std::vector<cv::Mat> &imgs) const {
    algorithm::ThreeFrequencyHeterodynePattern::Params params;
    params.width = params__.width_;
    params.height = params__.height_;
    params.nbrOfPeriods = params__.cycles_;
    params.horizontal = params__.horizontal_;
    params.maxCost = params__.maxCost_;
    params.minDisparity = params__.minDisparity_;
    params.maxDisparity = params__.maxDisparity_;
    params.confidenceThreshold = params__.confidenceThreshold_;
    params.shiftTime = params__.shiftTime_;

    return algorithm::ThreeFrequencyHeterodynePattern::create(params)->generate(
        imgs);
}

std::shared_ptr<Pattern>
BinoThreeFrequencyHeterodynePattern::create(const Params &params) {
    params__ = params;

    return std::make_shared<BinoThreeFrequencyHeterodynePattern>();
}

bool BinoThreeFrequencyHeterodynePattern::decode(
    const std::vector<std::vector<cv::Mat>> &patternImages,
    cv::Mat &disparityMap, const bool isGpu) const {
    CV_Assert(patternImages.size() >= 2);

#ifdef OPENCV_WITH_CUDA_MODULE
    if (isGpu) {
    }
#endif

    algorithm::ThreeFrequencyHeterodynePattern::Params params;
    params.shiftTime = params__.shiftTime_;
    params.width = params__.width_;
    params.height = params__.height_;
    params.nbrOfPeriods = params__.cycles_;
    params.horizontal = params__.horizontal_;
    params.maxCost = params__.maxCost_;
    params.minDisparity = params__.minDisparity_;
    params.maxDisparity = params__.maxDisparity_;
    params.confidenceThreshold = params__.confidenceThreshold_;

    auto pattern = algorithm::ThreeFrequencyHeterodynePattern::create(params);
    return pattern->decode(patternImages, disparityMap,
                           algorithm::THREE_FREQUENCY_HETERODYNE);
}
} // namespace cameras
} // namespace slmaster