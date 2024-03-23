#include "cuda_sinus_comple_graycode_pattern.hpp"

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {

using namespace cuda;

class CV_EXPORTS_W SinusCompleGrayCodePatternGPU_Impl CV_FINAL
    : public SinusCompleGrayCodePatternGPU {
  public:
    // Constructor
    explicit SinusCompleGrayCodePatternGPU_Impl(
        const SinusCompleGrayCodePatternGPU::Params &parameters =
            SinusCompleGrayCodePatternGPU::Params());
    // Destructor
    virtual ~SinusCompleGrayCodePatternGPU_Impl() override {};

    // Generate sinusoidal and complementary graycode patterns
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
    void unwrapPhaseMap(const GpuMat &grayImgs, const GpuMat &wrappedPhaseMap,
                        const GpuMat &confidenceMap, GpuMat &unwrappedPhaseMap,
                        const float shadowMask = 0.f,
                        Stream &stream = Stream::Null()) const override;

    // Compute disparity
    void computeDisparity(const GpuMat &lhsUnwrapMap,
                          const GpuMat &rhsUnwrapMap, GpuMat &disparityMap,
                          Stream &stream = Stream::Null()) const override;

  private:
    Params params;
};
// Default parameters value
SinusCompleGrayCodePatternGPU_Impl::Params::Params() {
    width = 1280;
    height = 720;
    nbrOfPeriods = 40;
    shiftTime = 4;
    minDisparity = 0;
    maxDisparity = 320;
    horizontal = false;
    confidenceThreshold = 5.f;
    maxCost = 0.1f;
    costMinDiff = 0.0001f;
}

SinusCompleGrayCodePatternGPU_Impl::SinusCompleGrayCodePatternGPU_Impl(
    const SinusCompleGrayCodePatternGPU::Params &parameters)
    : params(parameters) {}

void SinusCompleGrayCodePatternGPU_Impl::computeWrappedAndConfidenceMap(
    const GpuMat &patternImages, GpuMat &wrappedPhaseMap, GpuMat &confidenceMap,
    Stream &stream) const {

    calcPSPWrappedAndConfidenceMap(patternImages, wrappedPhaseMap,
                                   confidenceMap, stream);
}

void SinusCompleGrayCodePatternGPU_Impl::unwrapPhaseMap(
    const GpuMat &grayImgs, const GpuMat &wrappedPhaseMap,
    const GpuMat &confidenceMap, GpuMat &unwrappedPhaseMap,
    const float confidenceThreshold, Stream &stream) const {
    unwrapSinusCompleGraycodeMap(grayImgs, wrappedPhaseMap, confidenceMap,
                                 unwrappedPhaseMap, confidenceThreshold,
                                 stream);
}

bool SinusCompleGrayCodePatternGPU_Impl::generate(std::vector<cv::Mat> &pattern) {
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
    // generate complementary graycode imgs.
    const int grayCodeImgsCount =
        static_cast<int>(std::log2(params.nbrOfPeriods)) + 1;
    std::vector<uchar> encodeSequential = {0, 255};
    for (int i = 0; i < grayCodeImgsCount; ++i) {
        Mat intensityMap = Mat::zeros(height, width, CV_8UC1);
        const int pixelsPerBlock =
            static_cast<int>(width / encodeSequential.size());
        for (size_t j = 0; j < encodeSequential.size(); ++j) {
            intensityMap(Rect(static_cast<int>(j) * pixelsPerBlock, 0,
                              pixelsPerBlock, height)) = encodeSequential[j];
        }

        const int lastSequentialSize =
            static_cast<int>(encodeSequential.size());
        for (int j = lastSequentialSize - 1; j >= 0; --j) {
            encodeSequential.push_back(encodeSequential[j]);
        }

        intensityMap = params.horizontal ? intensityMap.t() : intensityMap;
        pattern.push_back(intensityMap);
    }

    return true;
}

void SinusCompleGrayCodePatternGPU_Impl::computeDisparity(
    const GpuMat &leftUnwrapMap, const GpuMat &rightUnwrapMap,
    GpuMat &disparityMap, Stream &stream) const {
    StereoMatchParams matchParams;
    matchParams.maxCost = params.maxCost;
    matchParams.minDisp = params.minDisparity;
    matchParams.maxDisp = params.maxDisparity;
    matchParams.costMinDiff = params.costMinDiff;

    stereoMatch(leftUnwrapMap, rightUnwrapMap, matchParams, disparityMap,
                stream);
}

bool SinusCompleGrayCodePatternGPU_Impl::decode(
    const std::vector<std::vector<Mat>> &patternImages, Mat &disparityMap,
    const int flags, Stream &stream) const {
    CV_Assert(!patternImages.empty());

    if (flags == SINUSOIDAL_COMPLEMENTARY_GRAY_CODE_GPU) {
        std::vector<std::vector<Mat>> imgsDivided(
            4); // leftPhaseImgs, leftGrayImgs, rightPhaseImgs, rightGrayImgs
        for (int i = 0; i < patternImages[0].size(); ++i) {
            if (i < params.shiftTime) {
                imgsDivided[0].push_back(patternImages[0][i]);
                imgsDivided[2].push_back(patternImages[1][i]);
            } else {
                imgsDivided[1].push_back(patternImages[0][i]);
                imgsDivided[3].push_back(patternImages[1][i]);
            }
        }

        std::vector<Mat> imgsDividedMerged(4);
        parallel_for_(Range(0, 4), [&](const Range &range) {
            for (int i = range.start; i < range.end; ++i) {
                merge(imgsDivided[i], imgsDividedMerged[i]);
            }
        });

        std::vector<Stream> streams(2, Stream(cudaStreamNonBlocking));
        std::vector<GpuMat> imgsDividedMergedDev(4);
        std::vector<GpuMat> wrappedMapDev(2), confidenceMap(2), unwrappedMap(2);
        imgsDividedMergedDev[0].upload(imgsDividedMerged[0], streams[0]);
        computeWrappedAndConfidenceMap(imgsDividedMergedDev[0],
                                       wrappedMapDev[0], confidenceMap[0],
                                       streams[0]);
        imgsDividedMergedDev[2].upload(imgsDividedMerged[2], streams[1]);
        computeWrappedAndConfidenceMap(imgsDividedMergedDev[2],
                                       wrappedMapDev[1], confidenceMap[1],
                                       streams[1]);
        imgsDividedMergedDev[1].upload(imgsDividedMerged[1], streams[0]);
        unwrapPhaseMap(imgsDividedMergedDev[1], wrappedMapDev[0],
                       confidenceMap[0], unwrappedMap[0],
                       params.confidenceThreshold, streams[0]);
        imgsDividedMergedDev[3].upload(imgsDividedMerged[3], streams[1]);
        unwrapPhaseMap(imgsDividedMergedDev[3], wrappedMapDev[1],
                       confidenceMap[1], unwrappedMap[1],
                       params.confidenceThreshold, streams[1]);

        streams[0].waitForCompletion();
        streams[1].waitForCompletion();

        cv::cuda::GpuMat disparityMapDev;
        computeDisparity(unwrappedMap[0], unwrappedMap[1], disparityMapDev);

        disparityMapDev.download(disparityMap);
    }

    return true;
}

Ptr<SinusCompleGrayCodePatternGPU> SinusCompleGrayCodePatternGPU::create(
    const SinusCompleGrayCodePatternGPU::Params &params) {
    return makePtr<SinusCompleGrayCodePatternGPU_Impl>(params);
}
} // namespace cuda
} // namespace cv
