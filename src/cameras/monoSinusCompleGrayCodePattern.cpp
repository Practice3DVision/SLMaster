#include "monoSinusCompleGrayCodePattern.h"

#include "../algorithm/algorithm.h"

using namespace cv;

namespace slmaster {
namespace cameras {

MonoSinusCompleGrayCodePattern::MonoSinusCompleGrayCodePattern() {
    params_.reset(new BinoPatternParams());
}

bool MonoSinusCompleGrayCodePattern::generate(
    std::vector<cv::Mat> &imgs) const {
    auto monoPatternParams = static_cast<MonoPatternParams *>(params_.get());

    algorithm::SinusCompleGrayCodePattern::Params params;
    params.width = monoPatternParams->width_;
    params.height = monoPatternParams->height_;
    params.nbrOfPeriods = monoPatternParams->cycles_;
    params.horizontal = monoPatternParams->horizontal_;
    params.confidenceThreshold = monoPatternParams->confidenceThreshold_;
    params.shiftTime = monoPatternParams->shiftTime_;

    return algorithm::SinusCompleGrayCodePattern::create(params)
        ->generate(imgs);
}

bool MonoSinusCompleGrayCodePattern::decode(
    const std::vector<std::vector<cv::Mat>> &patternImages, cv::Mat &depthMap,
    const bool isGpu) const {
    CV_Assert(patternImages.size() >= 1);

    auto monoPatternParams = static_cast<MonoPatternParams *>(params_.get());

#ifdef OPENCV_WITH_CUDA_MODULE
    if (isGpu) {
        algorithm::SinusCompleGrayCodePatternGPU::Params params;
        params.shiftTime = monoPatternParams->shiftTime_;
        params.confidenceThreshold = monoPatternParams->confidenceThreshold_;
        params.height = monoPatternParams->height_;
        params.width = monoPatternParams->width_;
        params.nbrOfPeriods = monoPatternParams->cycles_;
        params.horizontal = monoPatternParams->horizontal_;

        auto pattern = algorithm::SinusCompleGrayCodePatternGPU::create(params);

        std::vector<std::vector<Mat>> imgsDivided(
            2); // leftPhaseImgs, leftGrayImgs
        for (int i = 0; i < patternImages[0].size(); ++i) {
            if (i < params.shiftTime) {
                imgsDivided[0].push_back(patternImages[0][i]);
            } else {
                imgsDivided[1].push_back(patternImages[0][i]);
            }
        }

        std::vector<Mat> imgsDividedMerged(2);
        parallel_for_(Range(0, 2), [&](const Range &range) {
            for (int i = range.start; i < range.end; ++i) {
                merge(imgsDivided[i], imgsDividedMerged[i]);
            }
        });

        std::vector<cv::cuda::GpuMat> imgsDividedMergedDev(2);
        cv::cuda::GpuMat wrappedMapDev, confidenceMapDev, unwrappedMapDev;
        imgsDividedMergedDev[0].upload(imgsDividedMerged[0]);
        imgsDividedMergedDev[1].upload(imgsDividedMerged[1]);
        pattern->computeWrappedAndConfidenceMap(
            imgsDividedMergedDev[0], wrappedMapDev, confidenceMapDev);
        pattern->unwrapPhaseMap(imgsDividedMergedDev[1], wrappedMapDev,
                                confidenceMapDev, unwrappedMapDev,
                                params.confidenceThreshold);

        return pattern->decode(patternImages, depthMap,
                               algorithm::SINUSOIDAL_COMPLEMENTARY_GRAY_CODE_GPU);
    }
#endif

    algorithm::SinusCompleGrayCodePattern::Params params;

    auto pattern = algorithm::SinusCompleGrayCodePattern::create(params);
    // return pattern->decode(patternImages, disparityMap, cv::noArray(),
    // cv::noArray(), cv::structured_light::SINUSOIDAL_COMPLEMENTARY_GRAY_CODE);

    return true;
}
} // namespace cameras
} // namespace slmaster
