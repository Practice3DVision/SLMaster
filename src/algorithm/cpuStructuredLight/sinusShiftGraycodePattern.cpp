#include "sinusShiftGraycodePattern.hpp"

#include "recoverDepth.h"

using namespace cv;

namespace slmaster {
namespace algorithm {
class SinusShiftGrayCodePattern_Impl final : public SinusShiftGrayCodePattern {
  public:
    // Constructor
    explicit SinusShiftGrayCodePattern_Impl(
        const SinusShiftGrayCodePattern::Params &parameters =
            SinusShiftGrayCodePattern::Params());
    // Destructor
    virtual ~SinusShiftGrayCodePattern_Impl() CV_OVERRIDE {};

    // Generate sinusoidal and complementary graycode patterns
    bool generate(OutputArrayOfArrays patternImages) CV_OVERRIDE;

    // decode patterns and compute disparity map.
    bool decode(const std::vector<std::vector<Mat>> &patternImages,
                OutputArray disparityMap,
                InputArrayOfArrays blackImages = noArray(),
                InputArrayOfArrays whiteImages = noArray(),
                int flags = 0) const CV_OVERRIDE;

    // Compute a confidence map from sinusoidal patterns
    void computeConfidenceMap(InputArrayOfArrays patternImages,
                              OutputArray confidenceMap) const CV_OVERRIDE;

    // Compute a wrapped phase map from the sinusoidal patterns
    void computePhaseMap(InputArrayOfArrays patternImages,
                         OutputArray wrappedPhaseMap) const CV_OVERRIDE;

    // Compute a floor map from complementary graycode patterns and
    // wrappedPhaseMap.
    void computeFloorMap(InputArrayOfArrays patternImages,
                         InputArray confidenceMap, InputArray wrappedPhaseMap,
                         OutputArray floorMap) const CV_OVERRIDE;

    // Unwrap the wrapped phase map to remove phase ambiguities
    void unwrapPhaseMap(InputArray wrappedPhaseMap, InputArray floorMap,
                        OutputArray unwrappedPhaseMap,
                        InputArray shadowMask = noArray()) const CV_OVERRIDE;

    // Compute disparity
    void computeDisparity(InputArray lhsUnwrapMap, InputArray rhsUnwrapMap,
                          OutputArray disparityMap) const CV_OVERRIDE;

  private:
    Params params;
};
// Default parameters value
SinusShiftGrayCodePattern_Impl::Params::Params() {
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

SinusShiftGrayCodePattern_Impl::SinusShiftGrayCodePattern_Impl(
    const SinusShiftGrayCodePattern::Params &parameters)
    : params(parameters) {}

void SinusShiftGrayCodePattern_Impl::computeConfidenceMap(
    InputArrayOfArrays patternImages, OutputArray confidenceMap) const {
    const std::vector<Mat> &imgs =
        *static_cast<const std::vector<Mat> *>(patternImages.getObj());

    Mat &confidence = *static_cast<Mat *>(confidenceMap.getObj());

    CV_Assert(imgs.size() >= static_cast<size_t>(params.shiftTime));

    confidence = Mat::zeros(imgs[0].size(), CV_32FC1);

    for (int i = 0; i < params.shiftTime; ++i) {
        cv::Mat fltImg;
        imgs[i].convertTo(fltImg, CV_32FC1);
        confidence += fltImg / params.shiftTime;
    }
}

void SinusShiftGrayCodePattern_Impl::computePhaseMap(
    InputArrayOfArrays patternImages, OutputArray wrappedPhaseMap) const {
    const std::vector<Mat> &imgs =
        *static_cast<const std::vector<Mat> *>(patternImages.getObj());
    Mat &wrappedPhase = *static_cast<Mat *>(wrappedPhaseMap.getObj());

    CV_Assert(imgs.size() >= static_cast<size_t>(params.shiftTime));

    const int height = imgs[0].rows;
    const int width = imgs[0].cols;
    wrappedPhase = Mat::zeros(height, width, CV_32FC1);

    const float shiftVal = static_cast<float>(CV_2PI) / params.shiftTime;

    parallel_for_(Range(0, height), [&](const Range &range) {
        std::vector<const uchar *> imgsPtrs(params.shiftTime);

        for (int i = range.start; i < range.end; ++i) {
            auto wrappedPhasePtr = wrappedPhase.ptr<float>(i);

            for (int j = 0; j < params.shiftTime; ++j) {
                imgsPtrs[j] = imgs[j].ptr<uchar>(i);
            }

            for (int j = 0; j < width; ++j) {
                float molecules = 0.f, denominator = 0.f;
                for (int k = 0; k < params.shiftTime; ++k) {
                    molecules += imgsPtrs[k][j] * sin(k * shiftVal);
                    denominator += imgsPtrs[k][j] * cos(k * shiftVal);
                }

                wrappedPhasePtr[j] = -atan2(molecules, denominator);
            }
        }
    });
}

void SinusShiftGrayCodePattern_Impl::computeFloorMap(
    InputArrayOfArrays patternImages, InputArray confidenceMap,
    InputArray wrappedPhaseMap, OutputArray floorMap) const {
    const std::vector<Mat> &imgs =
        *static_cast<const std::vector<Mat> *>(patternImages.getObj());
    const Mat &confidence = *static_cast<const Mat *>(confidenceMap.getObj());
    const Mat &wrappedPhase =
        *static_cast<const Mat *>(wrappedPhaseMap.getObj());
    Mat &floor = *static_cast<Mat *>(floorMap.getObj());

    CV_Assert(!imgs.empty() && !confidence.empty() && !wrappedPhase.empty());
    CV_Assert(std::pow(2, imgs.size() - params.shiftTime) ==
              params.nbrOfPeriods);

    const int height = imgs[0].rows;
    const int width = imgs[0].cols;
    floor = Mat::zeros(height, width, CV_16UC1);

    const int grayCodeImgsCount =
        static_cast<int>(imgs.size() - params.shiftTime);
    // 计算初始格雷码
    parallel_for_(Range(0, height), [&](const Range &range) {
        std::vector<const uchar *> imgsPtrs(grayCodeImgsCount);

        for (int i = range.start; i < range.end; ++i) {
            for (int j = 0; j < grayCodeImgsCount; ++j) {
                imgsPtrs[j] = imgs[params.shiftTime + j].ptr<uchar>(i);
            }
            auto confidencePtr = confidence.ptr<float>(i);
            auto wrappedPhasePtr = wrappedPhase.ptr<float>(i);
            auto floorPtr = floor.ptr<uint16_t>(i);
            for (int j = 0; j < width; ++j) {
                uint16_t curK = 0, tempVal = 0;
                for (int k = 0; k < grayCodeImgsCount; ++k) {
                    tempVal ^= imgsPtrs[k][j] > confidencePtr[j];
                    curK = (curK << 1) + tempVal;
                }

                if (curK >= params.nbrOfPeriods) {
                    floorPtr[j] = 0;
                    continue;
                }

                floorPtr[j] = curK;
            }
        }
    });

    // 进一步修正格雷码（垂直条纹情况）
    if (!params.horizontal) {
        parallel_for_(Range(0, height), [&](const Range &range) {
            for (int i = range.start; i < range.end; ++i) {
                auto floorPtr = floor.ptr<uint16_t>(i);
                auto confidencePtr = confidence.ptr<float>(i);
                auto wrappedPhasePtr = wrappedPhase.ptr<float>(i);
                std::vector<std::pair<int, float>> middleLocs(
                    params.nbrOfPeriods, std::make_pair(0, FLT_MAX));

                for (int j = 0; j < width; ++j) {
                    int kFloor = floorPtr[j];

                    if (kFloor == 0) {
                        continue;
                    }

                    auto cost = abs(abs(wrappedPhasePtr[j]) - CV_PI);

                    if (cost < middleLocs[kFloor].second &&
                        confidencePtr[j] > params.confidenceThreshold) {
                        middleLocs[kFloor].first = j;
                        middleLocs[kFloor].second = cost;
                    }
                }

                for (int j = 0; j < width; ++j) {
                    int kFloor = floorPtr[j];

                    if (kFloor == 0) {
                        continue;
                    }

                    if ((abs(wrappedPhasePtr[j]) < CV_2PI / 3 &&
                         j < middleLocs[kFloor].first) ||
                        wrappedPhasePtr[j] >= CV_2PI / 3) {
                        floorPtr[j] -= 1;
                    }
                }
            }
        });

        return;
    }

    // 进一步修正格雷码（水平条纹情况）
    parallel_for_(Range(0, floor.cols), [&](const Range &range) {
        for (int j = range.start; j < range.end; ++j) {
            std::vector<std::pair<int, float>> middleLocs(
                params.nbrOfPeriods, std::make_pair(0, FLT_MAX));

            for (int i = 0; i < height; ++i) {
                int kFloor = floor.ptr<uint16_t>(i)[j];

                if (kFloor == 0) {
                    continue;
                }

                auto cost = abs(abs(wrappedPhase.ptr<float>(i)[j]) - CV_PI);

                if (abs(wrappedPhase.ptr<float>(i)[j] - CV_PI) <
                        middleLocs[kFloor].second &&
                    confidence.ptr<float>(i)[j] > params.confidenceThreshold) {
                    middleLocs[kFloor].first = j;
                    middleLocs[kFloor].second = cost;
                }
            }

            for (int i = 0; i < height; ++i) {
                int kFloor = floor.ptr<uint16_t>(i)[j];

                if (kFloor == 0) {
                    continue;
                }

                if ((abs(wrappedPhase.ptr<float>(i)[j]) < CV_2PI / 3 &&
                     j < middleLocs[kFloor].first) ||
                    wrappedPhase.ptr<float>(i)[j] >= CV_2PI / 3) {
                    floor.ptr<uint16_t>(i)[j] -= 1;
                }
            }
        }
    });
}

void SinusShiftGrayCodePattern_Impl::unwrapPhaseMap(
    InputArray wrappedPhaseMap, InputArray floorMap,
    OutputArray unwrappedPhaseMap, InputArray shadowMask) const {
    const Mat &wrappedPhase =
        *static_cast<const Mat *>(wrappedPhaseMap.getObj());
    const Mat &floor = *static_cast<const Mat *>(floorMap.getObj());
    Mat &unwrappedPhase = *static_cast<Mat *>(unwrappedPhaseMap.getObj());

    CV_Assert(!wrappedPhase.empty() && !floor.empty());

    const int height = wrappedPhase.rows;
    const int width = wrappedPhase.cols;
    unwrappedPhase = Mat::zeros(height, width, CV_32FC1);

    cv::Mat shadow;
    if (!shadowMask.empty()) {
        shadow = *static_cast<Mat *>(shadowMask.getObj());
    }

    parallel_for_(Range(0, height), [&](const Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto wrappedPhasePtr = wrappedPhase.ptr<float>(i);
            auto floorPtr = floor.ptr<uint16_t>(i);
            auto unwrappedPhasePtr = unwrappedPhase.ptr<float>(i);
            const uchar *shadowPtr =
                shadow.empty() ? nullptr : shadow.ptr<uchar>(i);
            for (int j = 0; j < width; ++j) {
                // we add CV_PI to make wrap map to begin with 0.
                if (shadowPtr) {
                    if (shadowPtr[j]) {
                        if (floorPtr[j] < params.nbrOfPeriods) {
                            unwrappedPhasePtr[j] =
                                wrappedPhasePtr[j] +
                                static_cast<float>(CV_2PI) * floorPtr[j] +
                                static_cast<float>(CV_PI);
                        } else {
                            unwrappedPhasePtr[j] = 0.f;
                        }
                    } else {
                        unwrappedPhasePtr[j] = 0.f;
                    }
                } else {
                    if (floorPtr[j] < params.nbrOfPeriods) {
                        unwrappedPhasePtr[j] =
                            wrappedPhasePtr[j] +
                            static_cast<float>(CV_2PI) * floorPtr[j] +
                            static_cast<float>(CV_PI);
                    } else {
                        unwrappedPhasePtr[j] = 0.f;
                    }
                }
            }
        }
    });
}

bool SinusShiftGrayCodePattern_Impl::generate(OutputArrayOfArrays pattern) {
    std::vector<Mat> &imgs = *static_cast<std::vector<Mat> *>(pattern.getObj());
    imgs.clear();
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
        imgs.push_back(intensityMap);
    }
    // generate shift graycode imgs.
    const int grayCodeImgsCount =
        static_cast<int>(std::log2(params.nbrOfPeriods));
    std::vector<uchar> encodeSequential = {0, 255};
    const int pixelsLastHalfBlock = (width / pow(2, grayCodeImgsCount)) / 2;
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

        // shift img
        Mat cloneIntensityMap = intensityMap.clone();
        cloneIntensityMap(Rect(0, 0, pixelsLastHalfBlock, intensityMap.rows))
            .copyTo(
                intensityMap(Rect(intensityMap.cols - pixelsLastHalfBlock, 0,
                                  pixelsLastHalfBlock, intensityMap.rows)));
        cloneIntensityMap(Rect(pixelsLastHalfBlock, 0,
                               intensityMap.cols - pixelsLastHalfBlock,
                               intensityMap.rows))
            .copyTo(
                intensityMap(Rect(0, 0, intensityMap.cols - pixelsLastHalfBlock,
                                  intensityMap.rows)));

        intensityMap = params.horizontal ? intensityMap.t() : intensityMap;
        imgs.push_back(intensityMap);
    }

    return true;
}

// TODO@Evans Liu: 增加水平条纹y轴方向支持
void SinusShiftGrayCodePattern_Impl::computeDisparity(
    InputArray leftUnwrapMap, InputArray rightUnwrapMap,
    OutputArray disparityMap) const {
    const Mat &leftUnwrap = *static_cast<const Mat *>(leftUnwrapMap.getObj());
    const Mat &rightUnwrap = *static_cast<const Mat *>(rightUnwrapMap.getObj());
    Mat &disparity = *static_cast<Mat *>(disparityMap.getObj());

    matchWithAbsphase(leftUnwrap, rightUnwrap, disparity, params.minDisparity,
                      params.maxDisparity, params.confidenceThreshold,
                      params.maxCost);
}

bool SinusShiftGrayCodePattern_Impl::decode(
    const std::vector<std::vector<Mat>> &patternImages,
    OutputArray disparityMap, InputArrayOfArrays blackImages,
    InputArrayOfArrays whiteImages, int flags) const {
    CV_UNUSED(blackImages);
    CV_UNUSED(whiteImages);

    CV_Assert(!patternImages.empty());

    Mat &disparity = *static_cast<Mat *>(disparityMap.getObj());

    if (flags == SINUSOIDAL_COMPLEMENTARY_GRAY_CODE) {
        std::vector<cv::Mat> confidenceMap(2);
        std::vector<cv::Mat> wrappedMap(2);
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
            computeFloorMap(patternImages[range.start],
                            confidenceMap[range.start], wrappedMap[range.start],
                            floorMap[range.start]);
            // calculate unwrapped map
            unwrapPhaseMap(wrappedMap[range.start], floorMap[range.start],
                           unwrapMap[range.start],
                           confidenceMap[range.start] >
                               params.confidenceThreshold);
        });

        // calculate disparity map
        computeDisparity(unwrapMap[0], unwrapMap[1], disparity);
    }

    return true;
}

Ptr<SinusShiftGrayCodePattern> SinusShiftGrayCodePattern::create(
    const SinusShiftGrayCodePattern::Params &params) {
    return makePtr<SinusShiftGrayCodePattern_Impl>(params);
}
} // namespace algorithm
} // namespace slmaster