#include "cudaMultiViewStereoGeometryPattern.hpp"

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {

using namespace cuda;

class CV_EXPORTS_W MultiViewStereoGeometryPatternGPU_Impl final
    : public MultiViewStereoGeometryPatternGPU {
  public:
    // Constructor
    explicit MultiViewStereoGeometryPatternGPU_Impl(
        const MultiViewStereoGeometryPatternGPU::Params parameters =
            MultiViewStereoGeometryPatternGPU::Params());
    // Destructor
    virtual ~MultiViewStereoGeometryPatternGPU_Impl() override {};

    // Generate psp sinusoidal patterns
    bool generate(std::vector<cv::Mat> &patternImages) override;

    // decode patterns and compute disparity map.
    bool decode(const std::vector<std::vector<Mat>> &patternImages,
                Mat &disparityMap, const int flags = 0,
                Stream &stream = Stream::Null()) const override;

    // Compute a wrapped phase map from the sinusoidal patterns
    void computeWrappedAndConfidenceMap(
        const GpuMat &patternImages, GpuMat &wrappedPhaseMap,
        GpuMat &confidenceMap,
        Stream &stream = Stream::Null()) const override;

    // Unwrap the wrapped phase map to remove phase ambiguities
    void unwrapPhaseMap(const GpuMat &refUnwapMap,
                        const GpuMat &wrappedPhaseMap,
                        const GpuMat &confidenceMap, GpuMat &unwrappedPhaseMap,
                        Stream &stream = Stream::Null()) const override;

    // Use polynomials to fit the structured light system model and recover the
    // rough depth from the absolute phase map.
    void polynomialFitting(const GpuMat &unwrappedPhaseMap,
                           GpuMat &coarseDepthMap,
                           Stream &stream = Stream::Null()) const override;

    // Use multi-view stereo geometry constraints to remove rough depth map
    // noise or optimize accuracy.
    void multiViewStereoRefineDepth(
        const GpuMat &coarseDepthMap, const GpuMat &wrappedMap1,
        const GpuMat &confidenceMap1, const GpuMat &wrappedMap2,
        const GpuMat &confidenceMap2, const GpuMat &wrappedMap3,
        const GpuMat &confidenceMap3, GpuMat &refineDepthMap,
        Stream &stream = Stream::Null()) const override;

  private:
    Params params;
};

// Default parameters value
MultiViewStereoGeometryPatternGPU_Impl::Params::Params() {
    width = 1280;
    height = 720;
    nbrOfPeriods = 32;
    shiftTime = 4;
    minDisparity = 0;
    maxDisparity = 320;
    horizontal = false;
    confidenceThreshold = 5.f;
    maxCost = 0.1f;
    costMinDiff = 0.0001f;
    costMaxDiff = 1.f;
}

MultiViewStereoGeometryPatternGPU_Impl::MultiViewStereoGeometryPatternGPU_Impl(
    const MultiViewStereoGeometryPatternGPU_Impl::Params parameters)
    : params(parameters) {}

void MultiViewStereoGeometryPatternGPU_Impl::computeWrappedAndConfidenceMap(
    const GpuMat &patternImages, GpuMat &wrappedPhaseMap, GpuMat &confidenceMap,
    Stream &stream) const {

    calcPSPWrappedAndConfidenceMap(patternImages, wrappedPhaseMap,
                                   confidenceMap, stream);
}

void MultiViewStereoGeometryPatternGPU_Impl::unwrapPhaseMap(
    const GpuMat &refUnwapMap, const GpuMat &wrappedPhaseMap,
    const GpuMat &confidenceMap, GpuMat &unwrappedPhaseMap,
    Stream &stream) const {
    unwrapWithRefUnwrappedMap(wrappedPhaseMap, confidenceMap, refUnwapMap,
                              unwrappedPhaseMap, params.confidenceThreshold,
                              stream);
}

bool MultiViewStereoGeometryPatternGPU_Impl::generate(
    std::vector<cv::Mat> &pattern) {
    pattern.clear();

    const int height = params.horizontal ? params.width : params.height;
    const int width = params.horizontal ? params.height : params.width;
    const int pixelsPerPeriod = width / params.nbrOfPeriods;
    // generate phase-shift imgs.
    for (int i = 0; i < params.shiftTime; ++i) {
        Mat intensityMap = Mat::zeros(height, width, CV_8UC1);
        const float shiftVal =
            static_cast<float>(CV_2PI) / params.shiftTime * i;

        for (int j = 0; j < height; ++j) {
            auto intensityMapPtr = intensityMap.ptr<uchar>(j);
            for (int k = 0; k < width; ++k) {
                // Set the fringe starting intensity to 0 so that it corresponds
                // to the complementary graycode interval.
                const float wrappedPhaseVal =
                    (k % pixelsPerPeriod) /
                        static_cast<float>(pixelsPerPeriod) *
                        static_cast<float>(CV_2PI) -
                    static_cast<float>(CV_PI);
                intensityMapPtr[k] = static_cast<uchar>(
                    127.5 + 127.5 * cos(wrappedPhaseVal + shiftVal));
            }
        }

        intensityMap = params.horizontal ? intensityMap.t() : intensityMap;
        pattern.push_back(intensityMap);
    }

    return true;
}

void MultiViewStereoGeometryPatternGPU_Impl::polynomialFitting(
    const GpuMat &unwrappedPhaseMap, GpuMat &coarseDepthMap,
    Stream &stream) const {
    /*
    device::cudastructuredlight::polynomialFitting(unwrappedPhaseMap, params.M1,
                               params.K, params.minDepth, params.maxDepth,
                               coarseDepthMap, stream);
    */
    reverseCamera(
        unwrappedPhaseMap, params.PL1, params.PR4, params.minDepth,
        params.maxDepth,
        static_cast<float>(params.horizontal ? params.height : params.width) /
            params.nbrOfPeriods,
        coarseDepthMap, params.horizontal, stream);
}

void MultiViewStereoGeometryPatternGPU_Impl::multiViewStereoRefineDepth(
    const GpuMat &coarseDepthMap, const GpuMat &wrappedMap1,
    const GpuMat &confidenceMap1, const GpuMat &wrappedMap2,
    const GpuMat &confidenceMap2, const GpuMat &wrappedMap3,
    const GpuMat &confidenceMap3, GpuMat &refineDepthMap,
    Stream &stream) const {
    // TODO@LiuYunhuang:测试垂直相位情况
    multiViewStereoGeometry(
        coarseDepthMap, params.M1, wrappedMap1, confidenceMap1, params.M2,
        params.R12, params.T12, wrappedMap2, confidenceMap2, params.M3,
        params.R13, params.T13, wrappedMap3, confidenceMap3, params.PL1,
        params.PR2, refineDepthMap, params.confidenceThreshold, params.maxCost,
        params.horizontal, stream);
}

bool MultiViewStereoGeometryPatternGPU_Impl::decode(
    const std::vector<std::vector<Mat>> &patternImages, Mat &depthMap,
    const int flags, Stream &stream) const {
    CV_Assert(!patternImages.empty() && patternImages.size() == 3);

    if (flags == MULTI_VIEW_STEREO_GEOMETRY_GPU) {
        std::vector<Mat> imgsDividedMerged(
            3); // PhaseImgs-Camera1, PhaseImgs-Camera2, PhaseImgs-Camera3
        parallel_for_(Range(0, 3), [&](const Range &range) {
            for (int i = range.start; i < range.end; ++i) {
                merge(patternImages[i], imgsDividedMerged[i]);
            }
        });

        std::vector<Stream> streams(3, Stream(cudaStreamNonBlocking));
        std::vector<GpuMat> imgsDividedMergedDev(3);
        std::vector<GpuMat> wrappedMapDev(3), confidenceMap(3);
        GpuMat unwrappedMap, coarseDepthMap, refineDepthMap;
        imgsDividedMergedDev[0].upload(imgsDividedMerged[0], streams[0]);
        computeWrappedAndConfidenceMap(imgsDividedMergedDev[0],
                                       wrappedMapDev[0], confidenceMap[0],
                                       streams[0]);
        imgsDividedMergedDev[1].upload(imgsDividedMerged[1], streams[1]);
        computeWrappedAndConfidenceMap(imgsDividedMergedDev[1],
                                       wrappedMapDev[1], confidenceMap[1],
                                       streams[1]);
        imgsDividedMergedDev[2].upload(imgsDividedMerged[2], streams[2]);
        computeWrappedAndConfidenceMap(imgsDividedMergedDev[2],
                                       wrappedMapDev[2], confidenceMap[2],
                                       streams[2]);
        unwrapPhaseMap(params.refUnwrappedMap, wrappedMapDev[0],
                       confidenceMap[0], unwrappedMap, streams[0]);
        polynomialFitting(unwrappedMap, coarseDepthMap, streams[0]);

        streams[0].waitForCompletion();
        streams[1].waitForCompletion();
        streams[2].waitForCompletion();

        multiViewStereoRefineDepth(coarseDepthMap, wrappedMapDev[0],
                                   confidenceMap[0], wrappedMapDev[1],
                                   confidenceMap[1], wrappedMapDev[2],
                                   confidenceMap[2], refineDepthMap);

        refineDepthMap.download(depthMap);
    }

    return true;
}

Ptr<MultiViewStereoGeometryPatternGPU> MultiViewStereoGeometryPatternGPU::create(
    const MultiViewStereoGeometryPatternGPU::Params &params) {
    return makePtr<MultiViewStereoGeometryPatternGPU_Impl>(params);
}
} // namespace cuda
} // namespace cv
