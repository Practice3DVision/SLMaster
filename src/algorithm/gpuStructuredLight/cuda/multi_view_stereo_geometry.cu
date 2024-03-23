#include "common.hpp"

#include <Eigen/Eigen>

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {
namespace cuda {

const dim3 BLOCK(32, 8, 1);

// TODO@LiuYunhuang:测试水平相位情况
__global__ void multi_view_stereo_geometry_honrizon(
    const PtrStepSz<float> coarseDepthMap, const Eigen::Matrix3f M1,
    const PtrStep<float> wrappedMap1, const PtrStep<float> confidenceMap1,
    const Eigen::Matrix3f M2, const Eigen::Matrix3f R12,
    const Eigen::Vector3f T12, const PtrStep<float> wrappedMap2,
    const PtrStep<float> confidenceMap2, const Eigen::Matrix3f M3,
    const Eigen::Matrix3f R13, const Eigen::Vector3f T13,
    const PtrStep<float> wrappedMap3, const PtrStep<float> confidenceMap3,
    const Eigen::Matrix4f PL, const Eigen::Matrix4f PR,
    PtrStep<float> fineDepthMap, const float confidenceThreshold = 5.f,
    const float maxCost = 0.1f) {
    /*
    const int x1 = blockDim.x * blockIdx.x + threadIdx.x;
    const int y1 = blockDim.y * blockIdx.y + threadIdx.y;

    if (x1 > coarseDepthMap.cols - 1 || y1 > coarseDepthMap.rows - 1)
        return;

    const float zCam1 = coarseDepthMap.ptr(y1)[x1];

    if (zCam1 < 0.01f || confidenceMap1.ptr(y1)[x1] < confidenceThreshold) {
        fineDepthMap.ptr(y1)[x1] = 0.f;
        return;
    }

    // first camera
    Eigen::Vector3f cam1;
    cam1 << x1, y1, 1;
    cam1 = M1.inverse() * cam1 * zCam1;

    // second camera
    Eigen::Vector3f camElese = R12 * cam1 + T12;
    Eigen::Vector3f temp = M2 * camElese;
    int x2 = temp.x() / temp.z();
    int y2 = temp.y() / temp.z();

    if (x2 < 0 || x2 > coarseDepthMap.cols - 1 || y2 < 0 ||
        y2 > coarseDepthMap.rows - 1) {
        fineDepthMap.ptr(y1)[x1] = 0.f;
        return;
    }

    if (__ldg(&confidenceMap2.ptr(y2)[x2]) < confidenceThreshold) {
        fineDepthMap.ptr(y1)[x1] = 0.f;
        return;
    }

    // third camera
    camElese = R13 * cam1 + T13;
    temp = M3 * camElese;
    int x3 = temp.x() / temp.z();
    int y3 = temp.y() / temp.z();

    if (x3 < 0 || x3 > coarseDepthMap.cols - 1 || y3 < 0 ||
        y3 > coarseDepthMap.rows - 1) {
        fineDepthMap.ptr(y1)[x1] = 0.f;
        return;
    }

    if (__ldg(&confidenceMap3.ptr(y3)[x3]) < confidenceThreshold) {
        fineDepthMap.ptr(y1)[x1] = 0.f;
        return;
    }

    // remove depth or refine depth
    const float wrappedVal1 = __ldg(&wrappedMap1.ptr(y1)[x1]);
    const float wrappedVal2 = __ldg(&wrappedMap2.ptr(y2)[x2]);
    const float wrappedVal3 = __ldg(&wrappedMap3.ptr(y3)[x3]);

    const float diffWrappedVal12 = abs(wrappedVal1 - wrappedVal2);
    const float diffWrappedVal13 = abs(wrappedVal1 - wrappedVal3);
    const float SAME_FLOOR_DIFF_THRESHOLD = maxCost * 3;
    const float DIFF_FLOOR_DIFF_THRESHOLD = CV_2PI - SAME_FLOOR_DIFF_THRESHOLD;

    if ((diffWrappedVal12 < SAME_FLOOR_DIFF_THRESHOLD ||
        diffWrappedVal12 > DIFF_FLOOR_DIFF_THRESHOLD) &&
        (diffWrappedVal13 < SAME_FLOOR_DIFF_THRESHOLD ||
            diffWrappedVal13 > DIFF_FLOOR_DIFF_THRESHOLD)) {
        float finalX2 = x2, finalY2 = y2, finalX3 = x3, finalY3 = y3;
        // just stop when cost is smaller than maxCost
        if (diffWrappedVal12 > 0.01f) {
            int isBoundary = diffWrappedVal12 < DIFF_FLOOR_DIFF_THRESHOLD ? 1 :
-1; const int direction = (wrappedVal1 > wrappedVal2) ? isBoundary : -1 *
isBoundary; int bestX = x2; float minCost = diffWrappedVal12, preCost = FLT_MAX,
                aftCost = FLT_MAX;
            bool isUpdate = false;
#pragma unroll
            for (int i = 1; i < 12; ++i) {
                const int curLoc = x2 + (i * direction);
                const float stepMoveWrappedVal =
                    __ldg(&wrappedMap2.ptr(y2)[curLoc]);
                isBoundary = abs(wrappedVal1 - stepMoveWrappedVal) <
DIFF_FLOOR_DIFF_THRESHOLD ? 0 : 1; const int addOrSub2PI = wrappedVal1 >
stepMoveWrappedVal ? 1 : -1; const float cost = abs(wrappedVal1 -
stepMoveWrappedVal - CV_2PI * isBoundary * addOrSub2PI);

                if (cost < minCost) {
                    preCost = minCost;
                    minCost = cost;
                    bestX = curLoc;

                    isUpdate = true;
                }
                else if (isUpdate) {
                    aftCost = cost;
                    isUpdate = false;
                }
            }

            if(preCost > FLT_MAX - 10 || aftCost > FLT_MAX - 10) {
                finalX2 = bestX;
            }
            else {
                float denom = preCost + aftCost - 2.f * minCost;
                denom = abs(denom) < 0.0001f ? 0.0001f : denom;
                finalX2 = bestX + __fdividef(preCost - aftCost, denom * 2.f);
            }
        }

        Eigen::Matrix3f LC;
        Eigen::Vector3f RC;
        LC << PL(0, 0) - x1 * PL(2, 0), PL(0, 1) - x1 * PL(2, 1), PL(0, 2) - x1
* PL(2, 2), PL(1, 0) - y1 * PL(2, 0), PL(1, 1) - y1 * PL(2, 1), PL(1, 2) - y1 *
PL(2, 2), PR(0, 0) - finalX2 * PR(2, 0), PR(0, 1) - finalX2 * PR(2, 1), PR(0, 2)
- finalX2 * PR(2, 2); RC << finalX2 * PL(2, 3) - PL(0, 3), finalY2 * PL(2, 3) -
PL(1, 3), finalX3* PR(2, 3) - PR(0, 3); Eigen::Vector3f result = LC.inverse() *
RC; result = R12.inverse() * result - T12; fineDepthMap.ptr(y1)[x1] = result(2,
0);
    }
    else { // remove depth
        fineDepthMap.ptr(y1)[x1] = 0.f;
    }
    */
}

__global__ void multi_view_stereo_geometry_vertical(
    const PtrStepSz<float> coarseDepthMap, const Eigen::Matrix3f M1,
    const PtrStep<float> wrappedMap1, const PtrStep<float> confidenceMap1,
    const Eigen::Matrix3f M2, const Eigen::Matrix3f R12,
    const Eigen::Vector3f T12, const PtrStep<float> wrappedMap2,
    const PtrStep<float> confidenceMap2, const Eigen::Matrix3f M3,
    const Eigen::Matrix3f R13, const Eigen::Vector3f T13,
    const PtrStep<float> wrappedMap3, const PtrStep<float> confidenceMap3,
    const Eigen::Matrix4f PL, const Eigen::Matrix4f PR,
    PtrStep<float> fineDepthMap, const float confidenceThreshold = 5.f,
    const float maxCost = 0.1f) {
    const int x1 = blockDim.x * blockIdx.x + threadIdx.x;
    const int y1 = blockDim.y * blockIdx.y + threadIdx.y;

    if (x1 > coarseDepthMap.cols - 1 || y1 > coarseDepthMap.rows - 1)
        return;

    const float zCam1 = coarseDepthMap.ptr(y1)[x1];

    if (zCam1 < 0.01f || confidenceMap1.ptr(y1)[x1] < confidenceThreshold) {
        fineDepthMap.ptr(y1)[x1] = 0.f;
        return;
    }

    // first camera
    Eigen::Vector3f cam1;
    cam1 << x1, y1, 1;
    cam1 = M1.inverse() * cam1 * zCam1;

    // second camera
    Eigen::Vector3f camElese = R12 * cam1 + T12;
    Eigen::Vector3f temp = M2 * camElese;
    int x2 = temp.x() / temp.z();
    int y2 = temp.y() / temp.z();

    if (x2 < 0 || x2 > coarseDepthMap.cols - 1 || y2 < 0 ||
        y2 > coarseDepthMap.rows - 1) {
        fineDepthMap.ptr(y1)[x1] = 0.f;
        return;
    }

    if (__ldg(&confidenceMap2.ptr(y2)[x2]) < confidenceThreshold) {
        fineDepthMap.ptr(y1)[x1] = 0.f;
        return;
    }

    // third camera
    camElese = R13 * cam1 + T13;
    temp = M3 * camElese;
    int x3 = temp.x() / temp.z();
    int y3 = temp.y() / temp.z();

    if (x3 < 0 || x3 > coarseDepthMap.cols - 1 || y3 < 0 ||
        y3 > coarseDepthMap.rows - 1) {
        fineDepthMap.ptr(y1)[x1] = 0.f;
        return;
    }

    if (__ldg(&confidenceMap3.ptr(y3)[x3]) < confidenceThreshold) {
        fineDepthMap.ptr(y1)[x1] = 0.f;
        return;
    }

    // remove depth or refine depth
    const float wrappedVal1 = __ldg(&wrappedMap1.ptr(y1)[x1]);
    const float wrappedVal2 = __ldg(&wrappedMap2.ptr(y2)[x2]);
    const float wrappedVal3 = __ldg(&wrappedMap3.ptr(y3)[x3]);

    // refine depth
    /*             *
     *              *
     *   (point)x->  *
     *                *
     *                 *
     */
    const float diffWrappedVal12 = abs(wrappedVal1 - wrappedVal2);
    const float diffWrappedVal13 = abs(wrappedVal1 - wrappedVal3);

    if ((diffWrappedVal12 < maxCost || diffWrappedVal12 > CV_2PI - maxCost) &&
        (diffWrappedVal13 < maxCost * 3 ||
         diffWrappedVal13 > CV_2PI - maxCost * 3)) {
        float finalX2 = x2;
        // just stop when cost is smaller than maxCost
        if (diffWrappedVal12 > 0.01f) {
            int isBoundary = diffWrappedVal12 < CV_2PI - maxCost ? 1 : -1;
            const int direction =
                (wrappedVal1 > wrappedVal2) ? isBoundary : -1 * isBoundary;
            int bestX = x2;
            float minCost = diffWrappedVal12, preCost = FLT_MAX,
                  aftCost = FLT_MAX;
            bool isUpdate = false;
#pragma unroll
            for (int i = 1; i < 5; ++i) {
                const int curLoc = x2 + (i * direction);
                const float stepMoveWrappedVal =
                    (curLoc > coarseDepthMap.cols - 1 || curLoc < 0)
                        ? FLT_MAX
                        : __ldg(&wrappedMap2.ptr(y2)[curLoc]);
                isBoundary =
                    abs(wrappedVal1 - stepMoveWrappedVal) < CV_2PI - maxCost
                        ? 0
                        : 1;
                const int addOrSub2PI =
                    wrappedVal1 > stepMoveWrappedVal ? 1 : -1;
                const float cost = abs(wrappedVal1 - stepMoveWrappedVal -
                                       CV_2PI * isBoundary * addOrSub2PI);

                if (cost < minCost) {
                    preCost = minCost;
                    minCost = cost;
                    bestX = curLoc;

                    isUpdate = true;
                } else if (isUpdate) {
                    aftCost = cost;
                    isUpdate = false;
                }
            }

            if (preCost > FLT_MAX - 10 || aftCost > FLT_MAX - 10) {
                finalX2 = bestX;
            } else {
                float denom = preCost + aftCost - 2.f * minCost;
                denom = abs(denom) < 0.0001f ? 0.0001f : denom;
                finalX2 = bestX + __fdividef(preCost - aftCost, denom * 2.f);
            }
        }

        Eigen::Matrix3f LC;
        Eigen::Vector3f RC;
        LC << PL(0, 0) - x1 * PL(2, 0), PL(0, 1) - x1 * PL(2, 1),
            PL(0, 2) - x1 * PL(2, 2), PL(1, 0) - y1 * PL(2, 0),
            PL(1, 1) - y1 * PL(2, 1), PL(1, 2) - y1 * PL(2, 2),
            PR(0, 0) - finalX2 * PR(2, 0), PR(0, 1) - finalX2 * PR(2, 1),
            PR(0, 2) - finalX2 * PR(2, 2);
        RC << x1 * PL(2, 3) - PL(0, 3), y1 * PL(2, 3) - PL(1, 3),
            finalX2 * PR(2, 3) - PR(0, 3);
        Eigen::Vector3f result = LC.inverse() * RC;
        fineDepthMap.ptr(y1)[x1] = result(2, 0);
    } else { // remove depth
        fineDepthMap.ptr(y1)[x1] = 0.f;
    }
}

void multiViewStereoGeometry(
    const GpuMat &coarseDepthMap, const Eigen::Matrix3f &M1,
    const GpuMat &wrappedMap1, const GpuMat &confidenceMap1,
    const Eigen::Matrix3f &M2, const Eigen::Matrix3f &R12,
    const Eigen::Vector3f &T12, const GpuMat &wrappedMap2,
    const GpuMat &confidenceMap2, const Eigen::Matrix3f &M3,
    const Eigen::Matrix3f &R13, const Eigen::Vector3f &T13,
    const GpuMat &wrappedMap3, const GpuMat &confidenceMap3,
    const Eigen::Matrix4f &PL, const Eigen::Matrix4f &PR, GpuMat &fineDepthMap,
    const float confidenceThreshold = 5.f, const float maxCost = 0.01f,
    const bool isHonrizon = false, Stream &stream = Stream::Null()) {
    CV_Assert(!coarseDepthMap.empty() && !wrappedMap1.empty() &&
              !confidenceMap1.empty() && !wrappedMap2.empty() &&
              !confidenceMap2.empty() && !wrappedMap3.empty() &&
              !confidenceMap3.empty());

    const int rows = coarseDepthMap.rows;
    const int cols = coarseDepthMap.cols;

    if (fineDepthMap.empty()) {
        fineDepthMap.create(rows, cols, CV_32FC1);
    } else {
        fineDepthMap.setTo(0, stream);
    }

    const dim3 grid(divUp(cols, BLOCK.x), divUp(rows, BLOCK.y), 1);

    if (isHonrizon) {
        multi_view_stereo_geometry_honrizon<<<
            grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
            coarseDepthMap, M1, wrappedMap1, confidenceMap1, M2, R12, T12,
            wrappedMap2, confidenceMap2, M3, R13, T13, wrappedMap3,
            confidenceMap3, PL, PR, fineDepthMap, confidenceThreshold, maxCost);
    } else {
        multi_view_stereo_geometry_vertical<<<
            grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
            coarseDepthMap, M1, wrappedMap1, confidenceMap1, M2, R12, T12,
            wrappedMap2, confidenceMap2, M3, R13, T13, wrappedMap3,
            confidenceMap3, PL, PR, fineDepthMap, confidenceThreshold, maxCost);
    }
}

} // namespace cuda
} // namespace algorithm
} // namespace slmaster
