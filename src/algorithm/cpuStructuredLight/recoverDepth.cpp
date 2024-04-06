#include "recoverDepth.h"

using namespace cv;
using namespace Eigen;

namespace slmaster {
namespace algorithm {
void reverseCamera(const Mat &phase, const Matrix4f &PL, const Matrix4f &PR,
                   const float minDepth, const float maxDepth,
                   const float pitch, Mat &depth, const bool isHonrizon) {
    CV_Assert(!phase.empty());

    const int rows = phase.rows;
    const int cols = phase.cols;

    depth = Mat::zeros(rows, cols, CV_32FC1);

    parallel_for_(Range(0, rows), [&](const Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto ptrPhase = phase.ptr<float>(i);
            auto ptrDepth = depth.ptr<float>(i);

            for (int j = 0; j < cols; ++j) {
                if (ptrPhase[j] < 0.001f) {
                    ptrDepth[j] = 0.f;
                    continue;
                }

                const float uv = (ptrPhase[j] / CV_2PI) * pitch;
                const int indexUV = isHonrizon ? 1 : 0;

                Matrix3f mapL;
                mapL << PL(0, 0) - PL(2, 0) * j, PL(0, 1) - PL(2, 1) * j,
                    PL(0, 2) - PL(2, 2) * j, PL(1, 0) - PL(2, 0) * i,
                    PL(1, 1) - PL(2, 1) * i, PL(1, 2) - PL(2, 2) * i,
                    PR(indexUV, 0) - PR(2, 0) * uv,
                    PR(indexUV, 1) - PR(2, 1) * uv,
                    PR(indexUV, 2) - PR(2, 2) * uv;

                Vector3f mapR;
                mapR << PL(2, 3) * j - PL(0, 3), PL(2, 3) * i - PL(1, 3),
                    PR(2, 3) * uv - PR(indexUV, 3);

                Vector3f camPoint = mapL.inverse() * mapR;

                if (camPoint(2, 0) > minDepth && camPoint(2, 0) < maxDepth)
                    ptrDepth[j] = camPoint(2, 0);
            }
        }
    });
}

void matchWithAbsphase(const cv::Mat &leftUnwrap, const cv::Mat &rightUnwrap,
                       cv::Mat &disparity, const int minDisparity,
                       const int maxDisparity, const float confidenceThreshold,
                       const float maxCost) {
    CV_Assert(!leftUnwrap.empty() && !rightUnwrap.empty());

    const int height = leftUnwrap.rows;
    const int width = leftUnwrap.cols;
    disparity = cv::Mat::zeros(height, width, CV_32FC1);

    parallel_for_(Range(0, height), [&](const Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto leftUnwrapPtr = leftUnwrap.ptr<float>(i);
            auto rightUnwrapPtr = rightUnwrap.ptr<float>(i);
            auto disparityPtr = disparity.ptr<float>(i);

            for (int j = 0; j < width; ++j) {
                auto leftVal = leftUnwrapPtr[j];

                if (std::abs(leftVal) < 0.001f) {
                    continue;
                }

                float minCost = FLT_MAX;
                int bestDisp = 0;

                for (int d = minDisparity; d < maxDisparity; ++d) {
                    if (j - d < 0 || j - d > width - 1) {
                        continue;
                    }

                    const float curCost =
                        std::abs(leftVal - rightUnwrapPtr[j - d]);

                    if (curCost < minCost) {
                        minCost = curCost;
                        bestDisp = d;
                    }
                }

                if (minCost < maxCost) {
                    if (bestDisp == minDisparity || bestDisp == maxDisparity) {
                        disparityPtr[j] = bestDisp;
                        continue;
                    }

                    const float preCost =
                        std::abs(leftVal - rightUnwrapPtr[j - (bestDisp - 1)]);
                    const float nextCost =
                        std::abs(leftVal - rightUnwrapPtr[j - (bestDisp + 1)]);
                    const float denom =
                        std::max(0.0001f, preCost + nextCost - 2 * minCost);

                    disparityPtr[j] =
                        bestDisp + (preCost - nextCost) / (denom * 2.f);
                }
            }
        }
    });
}

void recoverDepthFromLightPlane(const std::vector<cv::Point2f> &points,
                                const cv::Mat &lightPlaneEq,
                                const cv::Mat &intrinsic,
                                std::vector<cv::Point3f> &points3D) {
    const float cx = intrinsic.ptr<double>(0)[2];
    const float cy = intrinsic.ptr<double>(1)[2];
    const float fx = intrinsic.ptr<double>(0)[0];
    const float fy = intrinsic.ptr<double>(1)[1];
    const float A = lightPlaneEq.ptr<double>(0)[0];
    const float B = lightPlaneEq.ptr<double>(1)[0];
    const float C = lightPlaneEq.ptr<double>(2)[0];
    const float D = lightPlaneEq.ptr<double>(3)[0];

    points3D.resize(points.size());

    parallel_for_(Range(0, points.size()), [&](const Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            points3D[i].z = -D / ((A * (points[i].x - cx)) / fx + (B * (points[i].y - cy)) / fy + C);
            points3D[i].x = (points[i].x - cx) / fx * points3D[i].z;
            points3D[i].y = (points[i].y - cy) / fy * points3D[i].z;
        }
    });
}
} // namespace algorithm
} // namespace slmaster