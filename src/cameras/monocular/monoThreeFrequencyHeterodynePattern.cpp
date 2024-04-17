#include "monoThreeFrequencyHeterodynePattern.h"

#include "../../algorithm/algorithm.h"

using namespace cv;
using namespace cv::cuda;
using namespace std;

namespace slmaster {

using namespace algorithm;

namespace cameras {

static MonoThreeFrequencyHeterodynePattern::Params params__;

MonoThreeFrequencyHeterodynePattern::MonoThreeFrequencyHeterodynePattern() {}

std::shared_ptr<Pattern>
MonoThreeFrequencyHeterodynePattern::create(const Params &params) {
    params__ = params;

    return std::make_shared<MonoThreeFrequencyHeterodynePattern>();
}

bool MonoThreeFrequencyHeterodynePattern::generate(vector<Mat> &imgs) const {

    ThreeFrequencyHeterodynePattern::Params params;
    params.width = params__.width_;
    params.height = params__.height_;
    params.nbrOfPeriods = params__.cycles_;
    params.horizontal = params__.horizontal_;
    params.confidenceThreshold = params__.confidenceThreshold_;
    params.shiftTime = params__.shiftTime_;

    return ThreeFrequencyHeterodynePattern::create(params)->generate(imgs);
}

bool MonoThreeFrequencyHeterodynePattern::decode(
    const vector<vector<Mat>> &patternImages, Mat &depthMap,
    const bool isGpu) const {
    CV_Assert(patternImages.size() >= 1);

#ifdef OPENCV_WITH_CUDA_MODULE
    if (isGpu) {
    }
#endif

    ThreeFrequencyHeterodynePattern::Params params;
    params.shiftTime = params__.shiftTime_;
    params.confidenceThreshold = params__.confidenceThreshold_;
    params.height = params__.height_;
    params.width = params__.width_;
    params.nbrOfPeriods = params__.cycles_;
    params.horizontal = params__.horizontal_;

    auto pattern = ThreeFrequencyHeterodynePattern::create(params);
    Mat wrappedMap, confidenceMap, floorMap, unwrappedMap;

    pattern->computePhaseMap(patternImages[0], wrappedMap);
    pattern->computeConfidenceMap(patternImages[0], confidenceMap);
    pattern->computeFloorMap(patternImages[0], confidenceMap, floorMap);
    pattern->unwrapPhaseMap(wrappedMap, floorMap, confidenceMap, unwrappedMap);

    reverseCamera(unwrappedMap, params__.PL1_, params__.PR4_,
                  params__.minDepth_, params__.maxDepth_,
                  static_cast<float>(params__.horizontal_ ? params__.height_
                                                          : params__.width_) /
                      params__.cycles_,
                  depthMap, params__.horizontal_);

    return true;
}
} // namespace cameras
} // namespace slmaster