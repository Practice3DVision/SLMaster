#include "threeFrequencyHeterodynePattern.hpp"

#include "recoverDepth.h"

using namespace cv;

namespace slmaster {
namespace algorithm {
class ThreeFrequencyHeterodynePattern_Impl final
    : public ThreeFrequencyHeterodynePattern {
  public:
    // Constructor
    explicit ThreeFrequencyHeterodynePattern_Impl(
        const ThreeFrequencyHeterodynePattern::Params &parameters =
            ThreeFrequencyHeterodynePattern::Params());
    // Destructor
    virtual ~ThreeFrequencyHeterodynePattern_Impl() CV_OVERRIDE {};

    // Generate sinusoidal and complementary graycode patterns
    bool generate(OutputArrayOfArrays patternImages) CV_OVERRIDE;

    // decode patterns and compute disparity map.
    bool decode(const std::vector<std::vector<Mat>> &patternImages,
                OutputArray disparityMap,
                int flags = 0) const CV_OVERRIDE;

    // Compute a confidence map from sinusoidal patterns
    void computeConfidenceMap(InputArrayOfArrays patternImages,
                              OutputArray confidenceMap) const CV_OVERRIDE;

    // Compute a wrapped phase map from the sinusoidal patterns
    void computePhaseMap(InputArrayOfArrays patternImages,
                         OutputArray wrappedPhaseMap) const CV_OVERRIDE;

    // Compute a floor map from complementary graycode patterns and
    // wrappedPhaseMap.
    void computeFloorMap(InputArrayOfArrays wrappedMaps,
                         InputArray confidenceMap,
                         OutputArray floorMap) const CV_OVERRIDE;

    // Unwrap the wrapped phase map to remove phase ambiguities
    void unwrapPhaseMap(InputArray wrappedPhaseMap, InputArray floorMap,
                        InputArray confidenceMap,
                        OutputArray unwrappedPhaseMap) const CV_OVERRIDE;

    // Compute disparity
    void computeDisparity(InputArray lhsUnwrapMap, InputArray rhsUnwrapMap,
                          OutputArray disparityMap) const CV_OVERRIDE;

  private:
    Params params;
};
// Default parameters value
ThreeFrequencyHeterodynePattern_Impl::Params::Params() {
    width = 1280;
    height = 720;
    nbrOfPeriods = 32;
    shiftTime = 4;
    minDisparity = 0;
    maxDisparity = 320;
    horizontal = false;
    confidenceThreshold = 5.f;
    maxCost = 0.1f;
}

ThreeFrequencyHeterodynePattern_Impl::ThreeFrequencyHeterodynePattern_Impl(
    const ThreeFrequencyHeterodynePattern::Params &parameters)
    : params(parameters) {}

void ThreeFrequencyHeterodynePattern_Impl::computeConfidenceMap(
    InputArrayOfArrays patternImages, OutputArray confidenceMap) const {
    const std::vector<Mat> &imgs =
        *static_cast<const std::vector<Mat> *>(patternImages.getObj());

    Mat &confidence = *static_cast<Mat *>(confidenceMap.getObj());

    CV_Assert(imgs.size() >= static_cast<size_t>(params.shiftTime));

    confidence = Mat::zeros(imgs[0].size(), CV_32FC1);

    const int height = imgs[0].rows;
    const int width = imgs[0].cols;

    const float shiftVal = static_cast<float>(CV_2PI) / params.shiftTime;

    parallel_for_(Range(0, height), [&](const Range &range) {
        std::vector<const uchar *> imgsPtrs(params.shiftTime);

        for (int i = range.start; i < range.end; ++i) {
            auto confidencePtr = confidence.ptr<float>(i);

            for (int j = 0; j < params.shiftTime; ++j) {
                imgsPtrs[j] = imgs[j].ptr<uchar>(i);
            }

            for (int j = 0; j < width; ++j) {
                float molecules = 0.f, denominator = 0.f;
                for (int k = 0; k < params.shiftTime; ++k) {
                    molecules += imgsPtrs[k][j] * sin(k * shiftVal);
                    denominator += imgsPtrs[k][j] * cos(k * shiftVal);
                }

                confidencePtr[j] =
                    2.f / params.shiftTime *
                    sqrt(molecules * molecules + denominator * denominator);
            }
        }
    });
}

void ThreeFrequencyHeterodynePattern_Impl::computePhaseMap(
    InputArrayOfArrays patternImages, OutputArray wrappedPhaseMap) const {
    const std::vector<Mat> &imgs =
        *static_cast<const std::vector<Mat> *>(patternImages.getObj());
    std::vector<Mat> &wrappedPhase =
        *static_cast<std::vector<Mat> *>(wrappedPhaseMap.getObj());

    CV_Assert(imgs.size() == static_cast<size_t>(params.shiftTime * 3));

    const int height = imgs[0].rows;
    const int width = imgs[0].cols;

    wrappedPhase.clear();
    wrappedPhase.resize(3);
    for (int i = 0; i < 3; ++i) {
        wrappedPhase[i] = Mat::zeros(height, width, CV_32FC1);
    }

    const double shiftVal = CV_2PI / params.shiftTime;

    parallel_for_(Range(0, height), [&](const Range &range) {
        std::vector<const uchar *> imgsPtrs(imgs.size());
        std::vector<float *> wrappedPhasesPtr(3);

        for (int i = range.start; i < range.end; ++i) {
            for (int j = 0; j < 3; ++j) {
                wrappedPhasesPtr[j] = wrappedPhase[j].ptr<float>(i);
            }

            for (int j = 0; j < imgs.size(); ++j) {
                imgsPtrs[j] = imgs[j].ptr<uchar>(i);
            }

            for (int j = 0; j < width; ++j) {
                for (int k = 0; k < 3; ++k) {
                    double molecules = 0, denominator = 0;
                    for (int s = 0; s < params.shiftTime; ++s) {
                        molecules += imgsPtrs[s + params.shiftTime * k][j] *
                                     sin(s * shiftVal);
                        denominator += imgsPtrs[s + params.shiftTime * k][j] *
                                       cos(s * shiftVal);
                    }

                    float phase = -atan2(molecules, denominator);

                    phase = phase > CV_PI
                                ? 3.1415926
                                : (phase < -CV_PI ? -3.1415926 : phase);

                    wrappedPhasesPtr[k][j] = phase;
                }
            }
        }
    });
}

void ThreeFrequencyHeterodynePattern_Impl::computeFloorMap(
    InputArrayOfArrays wrappedMaps, InputArray confidenceMap,
    OutputArray floorMap) const {
    const std::vector<Mat> &wrappedPhases =
        *static_cast<const std::vector<Mat> *>(wrappedMaps.getObj());
    const Mat &confidence = *static_cast<const Mat *>(confidenceMap.getObj());
    Mat &floor = *static_cast<Mat *>(floorMap.getObj());

    CV_Assert(!confidence.empty() && wrappedPhases.size() == 3);

    const int height = confidence.rows;
    const int width = confidence.cols;
    floor = Mat::zeros(height, width, CV_16UC1);

    std::vector<int> weeks = {params.nbrOfPeriods, params.nbrOfPeriods - 6,
                              params.nbrOfPeriods - 11};

    float ratio123d12 = 6.f;
    float ratio2d1 = params.nbrOfPeriods / 6.f;

    parallel_for_(Range(0, height), [&](const Range &range) {
        std::vector<const float *> wrapMapsPtrs(3);

        for (int i = range.start; i < range.end; ++i) {
            for (int j = 0; j < 3; ++j) {
                wrapMapsPtrs[j] = wrappedPhases[j].ptr<float>(i);
            }

            auto confidencePtr = confidence.ptr<float>(i);
            auto floorPtr = floor.ptr<uint16_t>(i);

            for (int j = 0; j < width; ++j) {
                if(confidencePtr[j] < params.confidenceThreshold) {
                    continue;
                }

                float p12 = wrapMapsPtrs[0][j] - wrapMapsPtrs[1][j];
                p12 += p12 < 0 ? CV_2PI : 0;
                float p23 = wrapMapsPtrs[1][j] - wrapMapsPtrs[2][j];
                p23 += p23 < 0 ? CV_2PI : 0;
                float p123 = p12 - p23;
                p123 += p123 < 0 ? CV_2PI : 0;

                int floor12 = round((p123 * ratio123d12 - p12) / CV_2PI);
                floor12 = floor12 == -1 ? ratio123d12 - 1 : floor12;
                auto unwrap12 = p12 + floor12 * CV_2PI;
                int floor1 =
                    round((unwrap12 * ratio2d1 - wrapMapsPtrs[0][j] - CV_PI) /
                          CV_2PI);
                floor1 = floor1 >= params.nbrOfPeriods ? 0 : floor1;

                floorPtr[j] = floor1;
            }
        }
    });
}

void ThreeFrequencyHeterodynePattern_Impl::unwrapPhaseMap(
    InputArray wrappedPhaseMap, InputArray floorMap, InputArray confidenceMap,
    OutputArray unwrappedPhaseMap) const {
    const Mat &wrappedPhase =
        *static_cast<const Mat *>(wrappedPhaseMap.getObj());
    const Mat &floor = *static_cast<const Mat *>(floorMap.getObj());
    const Mat &confidence = *static_cast<const Mat *>(confidenceMap.getObj());
    Mat &unwrappedPhase = *static_cast<Mat *>(unwrappedPhaseMap.getObj());

    CV_Assert(!wrappedPhase.empty() && !floor.empty());

    const int height = wrappedPhase.rows;
    const int width = wrappedPhase.cols;
    unwrappedPhase = Mat::zeros(height, width, CV_32FC1);

    parallel_for_(Range(0, height), [&](const Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto wrappedPhasePtr = wrappedPhase.ptr<float>(i);
            auto floorPtr = floor.ptr<uint16_t>(i);
            auto confidencePtr = confidence.ptr<float>(i);
            auto unwrappedPhasePtr = unwrappedPhase.ptr<float>(i);
            for (int j = 0; j < width; ++j) {
                // we add CV_PI to make wrap map to begin with 0.
                if (confidencePtr[j] > params.confidenceThreshold) {
                    unwrappedPhasePtr[j] =
                        wrappedPhasePtr[j] + CV_2PI * floorPtr[j] + CV_PI;
                    continue;
                }

                unwrappedPhasePtr[j] = 0.f;
            }
        }
    });
}

bool ThreeFrequencyHeterodynePattern_Impl::generate(
    OutputArrayOfArrays pattern) {
    std::vector<Mat> &imgs = *static_cast<std::vector<Mat> *>(pattern.getObj());
    imgs.clear();
    const int height = params.horizontal ? params.width : params.height;
    const int width = params.horizontal ? params.height : params.width;
    std::vector<int> weeks = {params.nbrOfPeriods, params.nbrOfPeriods - 6,
                              params.nbrOfPeriods - 11};
    // generate phase-shift imgs.
    for (int s = 0; s < 3; ++s) {
        // can be float point
        const float pixelsPerPeriod = static_cast<float>(width) / weeks[s];

        for (int i = 0; i < params.shiftTime; ++i) {
            const float shiftVal = CV_2PI / params.shiftTime * i;

            Mat intensityMap = Mat::zeros(height, width, CV_8UC1);

            for (int j = 0; j < height; ++j) {
                auto intensityMapPtr = intensityMap.ptr<uchar>(j);
                for (int k = 0; k < width; ++k) {
                    // Set the fringe starting intensity to 0 so that it
                    // corresponds to the complementary graycode interval.
                    const float denom =
                        k / pixelsPerPeriod - floor(k / pixelsPerPeriod);
                    const float wrappedPhaseVal = (2 * denom - 1) * CV_PI;
                    intensityMapPtr[k] = static_cast<uchar>(
                        127.5 + 127.5 * cos(wrappedPhaseVal + shiftVal));
                }
            }

            intensityMap = params.horizontal ? intensityMap.t() : intensityMap;
            imgs.push_back(intensityMap);
        }
    }

    return true;
}

// TODO@Evans Liu: 增加水平条纹y轴方向支持
void ThreeFrequencyHeterodynePattern_Impl::computeDisparity(
    InputArray leftUnwrapMap, InputArray rightUnwrapMap,
    OutputArray disparityMap) const {
    const Mat &leftUnwrap = *static_cast<const Mat *>(leftUnwrapMap.getObj());
    const Mat &rightUnwrap = *static_cast<const Mat *>(rightUnwrapMap.getObj());
    Mat &disparity = *static_cast<Mat *>(disparityMap.getObj());

    matchWithAbsphase(leftUnwrap, rightUnwrap, disparity, params.minDisparity,
                      params.maxDisparity, params.confidenceThreshold,
                      params.maxCost);
}

bool ThreeFrequencyHeterodynePattern_Impl::decode(
    const std::vector<std::vector<Mat>> &patternImages,
    OutputArray disparityMap, int flags) const {

    CV_Assert(!patternImages.empty());

    Mat &disparity = *static_cast<Mat *>(disparityMap.getObj());

    if (flags == THREE_FREQUENCY_HETERODYNE) {
        std::vector<cv::Mat> confidenceMap(2);
        std::vector<std::vector<cv::Mat>> wrappedMap(2);
        std::vector<cv::Mat> floorMap(2);
        std::vector<cv::Mat> unwrapMap(2);

        parallel_for_(Range(0, 2), [&](const Range &range) {
            // calculate confidence map
            computeConfidenceMap(patternImages[range.start],
                                 confidenceMap[range.start]);
            // calculate wrapped phase map
            computePhaseMap(patternImages[range.start],
                            wrappedMap[range.start]);
            // calculate floor map
            computeFloorMap(wrappedMap[range.start], confidenceMap[range.start],
                            floorMap[range.start]);
            // calculate unwrapped map
            unwrapPhaseMap(wrappedMap[range.start], floorMap[range.start],
                           confidenceMap[range.start], unwrapMap[range.start]);
        });

        // calculate disparity map
        computeDisparity(unwrapMap[0], unwrapMap[1], disparity);
    }

    return true;
}

Ptr<ThreeFrequencyHeterodynePattern> ThreeFrequencyHeterodynePattern::create(
    const ThreeFrequencyHeterodynePattern::Params &params) {
    return makePtr<ThreeFrequencyHeterodynePattern_Impl>(params);
}
} // namespace algorithm
} // namespace slmaster
