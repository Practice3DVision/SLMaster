#ifndef __DEFOCUSMETHOD_H_
#define __DEFOCUSMETHOD_H_

#include "common.h"

#include <opencv2/opencv.hpp>

namespace slmaster {
void twoDimensionErrorExpand(cv::Mat& img) {
    CV_Assert(!img.empty() && img.type() == CV_8UC1);

    const int rows = img.rows;
    const int cols = img.cols;
    const float threshod = 127.5;
    const float efficient[4] = { 7.0f / 16.0f, 5.0f / 16.0f, 3.0f / 16.0f, 1.0f / 16.0f};

    cv::parallel_for_(cv::Range(0, img.rows), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            auto imgPtr = img.ptr(i);
            for (int j = 0; j < cols; ++j) {
                const int error = imgPtr[j] - (imgPtr[j] < threshod ? 0 : 255);
                if (j + 1 < cols) {
                    imgPtr[j + 1] += efficient[0] * error;
                }

                if (i + 1 < rows) {
                    img.ptr(i + 1)[j] += efficient[1] * error;
                    if (j - 1 >= 0) {
                        img.ptr(i + 1)[j] += efficient[2] * error;
                    }
                    if (j + 1 < cols) {
                        img.ptr(i + 1)[j] += efficient[3] * error;
                    }
                }
            }
        }
    });

    cv::parallel_for_(cv::Range(0, img.rows), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            auto imgPtr = img.ptr(i);
            for (int j = 0; j < cols; ++j) {
                imgPtr[j] = imgPtr[j] < 127.5f ? 0 : 1;
            }
        }
    });
}

void binary(cv::Mat& img) {
    CV_Assert(!img.empty() && img.type() == CV_8UC1);

    const int rows = img.rows;
    const int cols = img.cols;

    cv::parallel_for_(cv::Range(0, img.rows), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            auto imgPtr = img.ptr(i);
            for (int j = 0; j < cols; ++j) {
                if(imgPtr[j] < 127.5f) {
                    imgPtr[j] = 0;
                }
                else {
                    imgPtr[j] = 1;
                }
            }
        }
    });
}

int opwmGetBinaryVal(const float wrappedPhaseVal, const bool inverse) {
    //const float truncatPhases[4] = { 0.173234670244565, 0.447810198774037, 1.06049800055750, 1.08708997788648};
    const float truncatPhases[4] = { 0.152789739321315, 0.490089110274083, 1.05070711562070, 1.11882783612695};

    if((wrappedPhaseVal >= truncatPhases[0] && wrappedPhaseVal <= truncatPhases[1]) || (wrappedPhaseVal >= truncatPhases[2] && wrappedPhaseVal <= truncatPhases[3])) {
        return inverse ? 1 : 0;
    }
    else {
        return inverse ? 0 : 1;
    }
}

void opwm(cv::Mat& img, const int cycles, const float shiftVal, const bool isHonrizon) {
    const int height = isHonrizon ? img.cols : img.rows;
    const int width = isHonrizon ? img.rows : img.cols;
    const int pixelsPerPeriod = width / cycles;
    // generate phase-shift imgs.
    img = cv::Mat::zeros(height, width, CV_8UC1);

    for (int j = 0; j < height; ++j) {
        auto intensityMapPtr = img.ptr<uchar>(j);
        for (int k = 0; k < width; ++k) {
            //move pi / 2 to owe basic encoing wave
            float wrappedPhaseVal =
                (k % pixelsPerPeriod) /
                    static_cast<float>(pixelsPerPeriod) * static_cast<float>(CV_2PI) + shiftVal - CV_PI / 2;

            while(wrappedPhaseVal > CV_2PI) {
                wrappedPhaseVal -= CV_2PI;
            }

            while(wrappedPhaseVal < 0) {
                wrappedPhaseVal += CV_2PI;
            }

            bool isInverse = false;
            while (wrappedPhaseVal > CV_PI / 2) {
                if(wrappedPhaseVal >= CV_PI / 2 && wrappedPhaseVal < CV_PI) {
                    wrappedPhaseVal = CV_PI - wrappedPhaseVal;
                }
                else {
                    wrappedPhaseVal = CV_2PI - wrappedPhaseVal;
                    isInverse = true;
                }
            }

            intensityMapPtr[k] = opwmGetBinaryVal(wrappedPhaseVal, isInverse);
        }
    }
}

}
#endif // !__DEFOCUSMETHOD_H_
