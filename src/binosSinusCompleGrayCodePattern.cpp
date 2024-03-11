#include "binosSinusCompleGrayCodePattern.h"

#include <opencv2/structured_light.hpp>

using namespace cv;

#ifdef WITH_CUDASTRUCTUREDLIGHT_MODULE
#include <opencv2/cudastructuredlight.hpp>
#endif

namespace slmaster {

BinoSinusCompleGrayCodePattern::BinoSinusCompleGrayCodePattern() {
    params_.reset(new BinoPatternParams());
}

bool BinoSinusCompleGrayCodePattern::generate(std::vector<cv::Mat>& imgs) const {
    auto binoPatternParams = static_cast<BinoPatternParams*>(params_.get());

    structured_light::SinusCompleGrayCodePattern::Params params;
    params.width = binoPatternParams->width_;
    params.height = binoPatternParams->height_;
    params.nbrOfPeriods = binoPatternParams->cycles_;
    params.horizontal = binoPatternParams->horizontal_;
    params.maxCost = binoPatternParams->maxCost_;
    params.minDisparity = binoPatternParams->minDisparity_;
    params.maxDisparity = binoPatternParams->maxDisparity_;
    params.confidenceThreshold = binoPatternParams->confidenceThreshold_;
    params.shiftTime = binoPatternParams->shiftTime_;

    return structured_light::SinusCompleGrayCodePattern::create(params)->generate(imgs);
}

bool BinoSinusCompleGrayCodePattern::decode( const std::vector< std::vector<cv::Mat> >& patternImages, cv::Mat& disparityMap, const bool isGpu) const {
    CV_Assert(patternImages.size() == 2);

    auto binoPatternParams = static_cast<BinoPatternParams*>(params_.get());

#ifdef WITH_CUDASTRUCTUREDLIGHT_MODULE
    if(isGpu) {
        cv::cuda::SinusCompleGrayCodePattern::Params params;
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

        auto pattern = cv::cuda::SinusCompleGrayCodePattern::create(params);
        return pattern->decode(patternImages, disparityMap, cv::cuda::SINUSOIDAL_COMPLEMENTARY_GRAY_CODE);
    }
#endif

    structured_light::SinusCompleGrayCodePattern::Params params;
    params.shiftTime = binoPatternParams->shiftTime_;
    params.width = binoPatternParams->width_;
    params.height = binoPatternParams->height_;
    params.nbrOfPeriods = binoPatternParams->cycles_;
    params.horizontal = binoPatternParams->horizontal_;
    params.maxCost = binoPatternParams->maxCost_;
    params.minDisparity = binoPatternParams->minDisparity_;
    params.maxDisparity = binoPatternParams->maxDisparity_;
    params.confidenceThreshold = binoPatternParams->confidenceThreshold_;

    auto pattern = structured_light::SinusCompleGrayCodePattern::create(params);
    return pattern->decode(patternImages, disparityMap, cv::noArray(), cv::noArray(), cv::structured_light::SINUSOIDAL_COMPLEMENTARY_GRAY_CODE);

    return true;
}
}
