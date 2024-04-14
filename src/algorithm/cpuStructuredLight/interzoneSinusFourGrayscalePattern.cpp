#include "interzoneSinusFourGrayscalePattern.hpp"

#include <unordered_map>

#include "recoverDepth.h"

using namespace cv;

namespace slmaster {
namespace algorithm {
class InterzoneSinusFourGrayscalePattern_Impl final
    : public InterzoneSinusFourGrayscalePattern {
  public:
    // Constructor
    explicit InterzoneSinusFourGrayscalePattern_Impl(
        const InterzoneSinusFourGrayscalePattern::Params &parameters =
            InterzoneSinusFourGrayscalePattern::Params());
    // Destructor
    virtual ~InterzoneSinusFourGrayscalePattern_Impl() CV_OVERRIDE {};

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
                         InputArray confidenceMap,
                         OutputArray floorMap) const CV_OVERRIDE;

    // Unwrap the wrapped phase map to remove phase ambiguities
    void unwrapPhaseMap(cv::InputArray wrappedPhaseMap,
                        cv::InputArray confidenceMap, cv::InputArray floorMap,
                        cv::OutputArray unwrappedPhaseMap) const CV_OVERRIDE;

    // Compute disparity
    void computeDisparity(InputArray lhsUnwrapMap, InputArray rhsUnwrapMap,
                          OutputArray disparityMap) const CV_OVERRIDE;

  private:
    // threshod four grayscale floor img
    void threshod(const Mat &img, std::vector<float> &threshodVal,
                  Mat &out) const;
    // k-means cluster
    float kMeans(const Mat &img, const Mat &confidenceMap,
                 std::vector<float> &threshod) const;
    // tansform to uniform value
    int lookupUniform(int val);
    Params params;
    mutable std::unordered_map<int, int> floorLUKTable;
};

int InterzoneSinusFourGrayscalePattern_Impl::lookupUniform(int val) {
    if (val == 0)
        return 0;
    if (val == 85)
        return 1;
    if (val == 170)
        return 2;
    if (val == 255)
        return 3;

    return INT_MAX;
}

// Default parameters value
InterzoneSinusFourGrayscalePattern_Impl::Params::Params() {
    width = 1280;
    height = 720;
    nbrOfPeriods = 16;
    shiftTime = 3;
    minDisparity = 0;
    maxDisparity = 320;
    horizontal = false;
    confidenceThreshold = 5.f;
    maxCost = 0.1f;
}

InterzoneSinusFourGrayscalePattern_Impl::
    InterzoneSinusFourGrayscalePattern_Impl(
        const InterzoneSinusFourGrayscalePattern::Params &parameters)
    : params(parameters) {
    // build lookup table
    std::vector<Mat> imgs;
    generate(imgs);
}

void InterzoneSinusFourGrayscalePattern_Impl::computeConfidenceMap(
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

void InterzoneSinusFourGrayscalePattern_Impl::computePhaseMap(
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

float InterzoneSinusFourGrayscalePattern_Impl::kMeans(
    const Mat &img, const Mat &confidenceMap,
    std::vector<float> &threshod) const {
    CV_Assert(!img.empty());

    if (threshod.empty())
        threshod = {-1.f, -1.f / 3, 1.f / 3, 1.f};

    std::vector<std::pair<float, int>> sumGray(4, std::make_pair(0.f, 0));
    float score = 0.f;
    std::mutex mutex;
    parallel_for_(Range(0, img.rows), [&](const Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto ptrImg = img.ptr<float>(i);
            auto ptrConfidenceMap = confidenceMap.ptr<float>(i);

            for (int j = 0; j < img.cols; ++j) {
                // skip lower confidence pixels
                if (ptrConfidenceMap[j] < params.confidenceThreshold) {
                    continue;
                }
                // find minimum distance
                float minDistance = FLT_MAX;
                int minDistanceK = INT_MAX;
                for (int k = 0; k < 4; ++k) {
                    float distance = abs(ptrImg[j] - threshod[k]);

                    if (distance < minDistance) {
                        minDistance = distance;
                        minDistanceK = k;
                    }
                }

                std::lock_guard<std::mutex> lock(mutex);
                sumGray[minDistanceK].first += ptrImg[j];
                ++sumGray[minDistanceK].second;
                score += minDistance;
            }
        }
    });

    for (int k = 0; k < 4; ++k) {
        threshod[k] = sumGray[k].first / sumGray[k].second;
    }

    score /= (sumGray[0].second + sumGray[1].second + sumGray[2].second +
              sumGray[3].second);

    return score;
}

void InterzoneSinusFourGrayscalePattern_Impl::threshod(
    const Mat &img, std::vector<float> &threshodVal, Mat &out) const {
    CV_Assert(!img.empty() && threshodVal.size() == 4);

    out = Mat::zeros(img.size(), CV_8UC1);

    const float half01 = (threshodVal[1] - threshodVal[0]) / 2.f;
    const float half12 = (threshodVal[2] - threshodVal[1]) / 2.f;
    const float half23 = (threshodVal[3] - threshodVal[2]) / 2.f;

    parallel_for_(Range(0, img.rows), [&](const Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto ptrImg = img.ptr<float>(i);
            auto ptrOut = out.ptr<uchar>(i);

            for (int j = 0; j < img.cols; ++j) {
                if (ptrImg[j] >= -FLT_MAX &&
                    ptrImg[j] < threshodVal[0] + half01) {
                    ptrOut[j] = 0;
                } else if (ptrImg[j] >= threshodVal[0] + half01 &&
                           ptrImg[j] < threshodVal[1] + half12) {
                    ptrOut[j] = 1;
                } else if (ptrImg[j] >= threshodVal[1] + half12 &&
                           ptrImg[j] < threshodVal[2] + half23) {
                    ptrOut[j] = 2;
                } else if (ptrImg[j] >= threshodVal[2] + half23 && ptrImg[j]) {
                    ptrOut[j] = 3;
                }
            }
        }
    });
}

void InterzoneSinusFourGrayscalePattern_Impl::computeFloorMap(
    InputArrayOfArrays patternImages, InputArray confidenceMap,
    OutputArray floorMap) const {
    const std::vector<Mat> &imgs =
        *static_cast<const std::vector<Mat> *>(patternImages.getObj());
    const Mat &confidence = *static_cast<const Mat *>(confidenceMap.getObj());
    Mat &floor = *static_cast<Mat *>(floorMap.getObj());

    CV_Assert(!imgs.empty() && !confidence.empty());
    CV_Assert(std::pow(4, imgs.size() - params.shiftTime) ==
              params.nbrOfPeriods);

    const int height = imgs[0].rows;
    const int width = imgs[0].cols;
    floor = Mat::zeros(height, width, CV_16UC1);

    const int grayCodeImgsCount = imgs.size() - params.shiftTime;
    // compute texture
    Mat texture = Mat::zeros(confidence.size(), CV_32FC1);
    parallel_for_(Range(0, params.height), [&](const Range &range) {
        std::vector<const uchar *> imgsPtrs(params.shiftTime);
        for (int i = range.start; i < range.end; ++i) {
            for (int j = 0; j < params.shiftTime; ++j) {
                imgsPtrs[j] = imgs[j].ptr<uchar>(i);
            }

            auto texturePtr = texture.ptr<float>(i);

            for (int j = 0; j < params.width; ++j) {
                for (auto ptr : imgsPtrs) {
                    texturePtr[j] += ptr[j];
                }

                texturePtr[j] /= imgsPtrs.size();
            }
        }
    });
    // normalize
    std::vector<Mat> normalizeGrayImgs(imgs.size() - params.shiftTime);
    parallel_for_(
        Range(params.shiftTime, imgs.size()), [&](const Range &range) {
            for (int i = range.start; i < range.end; ++i) {
                imgs[i].convertTo(normalizeGrayImgs[i - params.shiftTime],
                                  CV_32FC1);
                normalizeGrayImgs[i - params.shiftTime] =
                    (normalizeGrayImgs[i - params.shiftTime] - texture) /
                    confidence;
            }
        });
    // median filter
    parallel_for_(Range(0, normalizeGrayImgs.size()), [&](const Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            medianBlur(normalizeGrayImgs[i], normalizeGrayImgs[i], 5);
        }
    });
    // k-means cluster and threshod
    std::vector<Mat> threshodGrayImgs(grayCodeImgsCount);
    for (int i = 0; i < grayCodeImgsCount; ++i) {
        std::vector<float> threshodVal;
        int count = 0;
        float score = FLT_MAX;

        do {
            score = kMeans(normalizeGrayImgs[i], confidence, threshodVal);
        } while (++count < 5 && score > 20.f);

        threshod(normalizeGrayImgs[i], threshodVal, threshodGrayImgs[i]);
    }
    // compute floor map
    parallel_for_(Range(0, height), [&](const Range &range) {
        std::vector<const uchar *> imgsPtrs(grayCodeImgsCount);

        for (int i = range.start; i < range.end; ++i) {
            for (int j = 0; j < grayCodeImgsCount; ++j) {
                imgsPtrs[j] = threshodGrayImgs[j].ptr<uchar>(i);
            }

            auto ptrFloor = floor.ptr<uint16_t>(i);

            for (int j = 0; j < width; ++j) {
                int val = 0;

                for (int k = 0; k < imgsPtrs.size(); ++k) {
                    val += imgsPtrs[k][j] * pow(4, grayCodeImgsCount - 1 - k);
                }

                ptrFloor[j] = floorLUKTable[val];
            }
        }
    });
}

void InterzoneSinusFourGrayscalePattern_Impl::unwrapPhaseMap(
    InputArray wrappedPhaseMap, InputArray confidenceMap, InputArray floorMap,
    OutputArray unwrappedPhaseMap) const {
    const Mat &wrappedPhase =
        *static_cast<const Mat *>(wrappedPhaseMap.getObj());
    const Mat &confidence = *static_cast<const Mat *>(confidenceMap.getObj());
    const Mat &floor = *static_cast<const Mat *>(floorMap.getObj());
    Mat &unwrappedPhase = *static_cast<Mat *>(unwrappedPhaseMap.getObj());

    CV_Assert(!wrappedPhase.empty() && !confidence.empty() && !floor.empty());

    const int height = floor.rows;
    const int width = floor.cols;
    unwrappedPhase = Mat::zeros(height, width, CV_32FC1);

    parallel_for_(Range(0, height), [&](const Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto wrappedPhasePtr = wrappedPhase.ptr<float>(i);
            auto confidencePtr = confidence.ptr<float>(i);
            auto floorPtr = floor.ptr<uint16_t>(i);
            auto unwrappedPhasePtr = unwrappedPhase.ptr<float>(i);

            // find midlle wrapped phase location
            std::vector<std::pair<int, float>> middleLocs(
                params.nbrOfPeriods, std::make_pair(0, FLT_MAX));

            for (int j = 0; j < width; ++j) {
                int kFloor = floorPtr[j];

                auto cost = abs(wrappedPhasePtr[j]);

                if (cost < middleLocs[kFloor].second &&
                    confidencePtr[j] > params.confidenceThreshold) {
                    middleLocs[kFloor].first = j;
                    middleLocs[kFloor].second = cost;
                }
            }

            for (int j = 0; j < width; ++j) {
                if (confidencePtr[j] < params.confidenceThreshold) {
                    continue;
                }

                int kFloor = floorPtr[j];

                if (abs(wrappedPhasePtr[j]) < CV_PI / 3) {
                    unwrappedPhasePtr[j] =
                        wrappedPhasePtr[j] + CV_2PI * kFloor + CV_PI;
                    continue;
                }

                float phi =
                    wrappedPhasePtr[j] +
                    (j < middleLocs[kFloor].first ? 1 : -1) * CV_2PI / 3;
                phi = phi > CV_PI ? phi - CV_2PI
                                  : (phi <= -CV_PI ? phi + CV_2PI : phi);
                unwrappedPhasePtr[j] =
                    phi + CV_2PI * kFloor + CV_PI +
                    (j < middleLocs[kFloor].first ? -1 : 1) * CV_2PI / 3;
            }
        }
    });
}

bool InterzoneSinusFourGrayscalePattern_Impl::generate(
    OutputArrayOfArrays pattern) {
    CV_Assert(params.shiftTime % 2 != 0);

    std::vector<Mat> &imgs = *static_cast<std::vector<Mat> *>(pattern.getObj());
    imgs.clear();
    const int height = params.horizontal ? params.width : params.height;
    const int width = params.horizontal ? params.height : params.width;
    const int pixelsPerPeriod = width / params.nbrOfPeriods;
    // generate phase-shift imgs.
    for (int i = 0; i < params.shiftTime; ++i) {
        Mat intensityMap = Mat::zeros(height, width, CV_8UC1);
        const float shiftVal = CV_2PI / params.shiftTime * i;

        for (int j = 0; j < height; ++j) {
            auto intensityMapPtr = intensityMap.ptr<uchar>(j);
            for (int k = 0; k < width; ++k) {
                // Set the fringe starting intensity to 0 so that it corresponds
                // to the complementary graycode interval.
                const float wrappedPhaseVal =
                    (k % pixelsPerPeriod) /
                        static_cast<float>(pixelsPerPeriod) * CV_2PI -
                    CV_PI;
                intensityMapPtr[k] =
                    127.5 + 127.5 * cos(wrappedPhaseVal + shiftVal);
            }
        }

        intensityMap = params.horizontal ? intensityMap.t() : intensityMap;
        imgs.push_back(intensityMap);
    }
    // generate four grayscale graycode imgs.
    const int grayCodeImgsCount = std::log(params.nbrOfPeriods) / std::log(4);
    std::vector<uchar> encodeSequential = {0, 85, 170, 255};
    for (int i = 0; i < grayCodeImgsCount; ++i) {
        Mat intensityMap = Mat::zeros(height, width, CV_8UC1);
        const int pixelsPerBlock = width / encodeSequential.size();
        for (size_t j = 0; j < encodeSequential.size(); ++j) {
            intensityMap(Rect(static_cast<int>(j) * pixelsPerBlock, 0,
                              pixelsPerBlock, height)) = encodeSequential[j];
        }

        std::vector<uchar> copyEncodeSequential(encodeSequential.begin(),
                                                encodeSequential.end());

        for (int j = copyEncodeSequential.size() - 1; j >= 0; --j) {
            encodeSequential.push_back(encodeSequential[j]);
        }

        for (auto val : copyEncodeSequential) {
            encodeSequential.push_back(val);
        }

        for (int j = copyEncodeSequential.size() - 1; j >= 0; --j) {
            encodeSequential.push_back(encodeSequential[j]);
        }

        intensityMap = params.horizontal ? intensityMap.t() : intensityMap;
        imgs.push_back(intensityMap);
    }
    // build floor lookup table
    auto lastBlockPixels = (params.horizontal ? params.height : params.width) /
                           pow(4, grayCodeImgsCount);
    if (!params.horizontal) {
        int val = 0;

        for (int j = 0; j < params.width; j += lastBlockPixels) {
            int floor = 0;

            for (int i = params.shiftTime; i < imgs.size(); ++i) {
                floor += lookupUniform(imgs[i].ptr<uchar>(0)[j]) *
                         pow(4, grayCodeImgsCount - 1 - (i - params.shiftTime));
            }

            floorLUKTable[floor] = val++;
        }
    } else {
        int val = 0;

        for (int i = 0; i < params.height; i += lastBlockPixels) {
            int floor = 0;

            for (int j = params.shiftTime; j < imgs.size(); ++j) {
                floor += lookupUniform(imgs[j].ptr<uchar>(i)[0]) *
                         pow(4, grayCodeImgsCount - (i - params.shiftTime));
            }

            floorLUKTable[floor] = val++;
        }
    }

    return true;
}

// TODO@Evans Liu: 增加水平条纹y轴方向支持
void InterzoneSinusFourGrayscalePattern_Impl::computeDisparity(
    InputArray leftUnwrapMap, InputArray rightUnwrapMap,
    OutputArray disparityMap) const {
    const Mat &leftUnwrap = *static_cast<const Mat *>(leftUnwrapMap.getObj());
    const Mat &rightUnwrap = *static_cast<const Mat *>(rightUnwrapMap.getObj());
    Mat &disparity = *static_cast<Mat *>(disparityMap.getObj());

    matchWithAbsphase(leftUnwrap, rightUnwrap, disparity, params.minDisparity,
                      params.maxDisparity, params.confidenceThreshold,
                      params.maxCost);
}

bool InterzoneSinusFourGrayscalePattern_Impl::decode(
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
                            confidenceMap[range.start], floorMap[range.start]);
            // calculate unwrapped map
            unwrapPhaseMap(wrappedMap[range.start], confidenceMap[range.start],
                           floorMap[range.start], unwrapMap[range.start]);
        });

        // calculate disparity map
        computeDisparity(unwrapMap[0], unwrapMap[1], disparity);
    }

    return true;
}

Ptr<InterzoneSinusFourGrayscalePattern>
InterzoneSinusFourGrayscalePattern::create(
    const InterzoneSinusFourGrayscalePattern::Params &params) {
    return makePtr<InterzoneSinusFourGrayscalePattern_Impl>(params);
}
} // namespace algorithm
} // namespace slmaster