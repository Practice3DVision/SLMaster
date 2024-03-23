#include "common.hpp"

#include <Eigen/Eigen>

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {
namespace cuda {

const int ROW_PER_BLOCK = 8;
const int WARPS_PER_COL = 16;
const int THREADS_PER_GROUP = 32;
const dim3 BLOCK(THREADS_PER_GROUP, ROW_PER_BLOCK, 1);

struct CostDisp {
    __device__ CostDisp(const float &cost_, const int &disp_)
        : cost(cost_), disp(disp_) {}

    __device__ CostDisp() : cost(0.f), disp(0) {}

    float cost;
    int disp;
};

template <typename T, unsigned int GROUP_SIZE, unsigned int STEP>
struct subgroup_min_impl {
    static __device__ float2 call(const T &cost, const uint32_t &mask) {
        const T shflCost = __shfl_xor_sync(mask, cost, STEP / 2, GROUP_SIZE);
        const T minElement = shflCost > cost ? cost : shflCost;

        return subgroup_min_impl<T, GROUP_SIZE, STEP / 2>::call(minElement,
                                                                mask);
    }
};

template <typename T, unsigned int GROUP_SIZE>
struct subgroup_min_impl<T, GROUP_SIZE, 2u> {
    static __device__ float2 call(const T &cost, const uint32_t &mask) {
        float2 lastTwoMinElements;
        lastTwoMinElements.x = cost;
        lastTwoMinElements.y = __shfl_xor_sync(mask, cost, 1, GROUP_SIZE);

        return lastTwoMinElements;
    }
};

// never use it in current, because we apply different ratio in algorithm.
template <typename T, unsigned int GROUP_SIZE>
struct subgroup_min_impl<T, GROUP_SIZE, 1u> {
    static __device__ T call(const T &cost, const uint32_t &) { return cost; }
};

template <typename T, unsigned int GROUP_SIZE>
__device__ inline float2 subgroup_min(const T &cost, const uint32_t &mask) {
    return subgroup_min_impl<T, GROUP_SIZE, GROUP_SIZE>::call(cost, mask);
}

template <int DISP_PER_THREAD>
__global__ void stereo_match(const PtrStepSz<float> left,
                             const PtrStep<float> right, const int minDisp,
                             const int maxDisp, const float maxCost,
                             const float costMinDiff, PtrStep<float> dispMap) {
    const int dispBegin = threadIdx.x * DISP_PER_THREAD;
    // const int dispRange = maxDisp - minDisp;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int pixelsProcess = left.cols / gridDim.x;
    const int xBegin = pixelsProcess * blockIdx.x;
    const int xEnd = xBegin + pixelsProcess;

    if (y > left.rows - 1)
        return;

    __shared__ float unwrapVal[ROW_PER_BLOCK][DISP_PER_THREAD]
                              [THREADS_PER_GROUP + 1];

    // init shared memory
#pragma unroll
    for (int d = 0; d < DISP_PER_THREAD; ++d) {
        const int rightX = xBegin - 1 - (minDisp + dispBegin + d);

        if (rightX < 0 || rightX > left.cols - 1) {
            unwrapVal[threadIdx.y][d][1 + threadIdx.x] = FLT_MAX;
        } else {
            unwrapVal[threadIdx.y][d][1 + threadIdx.x] =
                __ldg(&right.ptr(y)[rightX]);
        }
    }

    if (threadIdx.x == 0) {
        const int rightX = xBegin - 1 - (minDisp - 1);
        if (rightX < 0 || rightX > left.cols - 1) {
            unwrapVal[threadIdx.y][DISP_PER_THREAD - 1][0] = FLT_MAX;
        } else {
            unwrapVal[threadIdx.y][DISP_PER_THREAD - 1][0] =
                __ldg(&right.ptr(y)[rightX]);
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = xBegin; j < xEnd; ++j) {
        // move to left
        const float rightLastVal =
            unwrapVal[threadIdx.y][DISP_PER_THREAD - 1][1 + threadIdx.x - 1];
        __syncthreads();
#pragma unroll
        for (int d = DISP_PER_THREAD - 1; d > 0; --d) {
            unwrapVal[threadIdx.y][d][1 + threadIdx.x] =
                unwrapVal[threadIdx.y][d - 1][1 + threadIdx.x];
        }

        unwrapVal[threadIdx.y][0][1 + threadIdx.x] = rightLastVal;

        if (threadIdx.x == 0) {
            const int rightX = j - (minDisp - 1);
            if (rightX < 0 || rightX > left.cols - 1) {
                unwrapVal[threadIdx.y][DISP_PER_THREAD - 1][0] = FLT_MAX;
            } else {
                unwrapVal[threadIdx.y][DISP_PER_THREAD - 1][0] =
                    __ldg(&right.ptr(y)[rightX]);
            }
        }

        __syncthreads();

        const float leftVal = __ldg(&left.ptr(y)[j]);

        if (abs(leftVal) < 0.001f) {
            dispMap.ptr(y)[j] = 0.f;
            continue;
        }

        CostDisp localBest(FLT_MAX, INT_MAX);
        float secondaryMinCost = FLT_MAX;

#pragma unroll
        for (int d = 0; d < DISP_PER_THREAD; ++d) {
            float cost =
                abs(leftVal - unwrapVal[threadIdx.y][d][1 + threadIdx.x]);

            if (localBest.cost > cost) {
                secondaryMinCost = localBest.cost;
                localBest.cost = cost;
                localBest.disp = d;
            }
        }
        __syncthreads();
        // find the minimum cost and best disparity
        const float2 lastTwoMinCosts =
            subgroup_min<float, 32>(localBest.cost, 0xffffffff);
        const float minCost = lastTwoMinCosts.x < lastTwoMinCosts.y
                                  ? lastTwoMinCosts.x
                                  : lastTwoMinCosts.y;
        const float diffCurPixel = abs(lastTwoMinCosts.x - lastTwoMinCosts.y);

        if (diffCurPixel < costMinDiff || minCost > maxCost)
            continue;

        if (abs(minCost - localBest.cost) < 0.001f &&
            abs(secondaryMinCost - minCost) > costMinDiff) {
            if ((threadIdx.x == 0 && localBest.disp == 0) ||
                (localBest.disp == DISP_PER_THREAD - 1 &&
                 threadIdx.x == blockDim.x - 1)) {
                dispMap.ptr(y)[j] = minDisp + dispBegin + localBest.disp;
            } else {
                float preCost, aftCost;
                if (localBest.disp == 0) {
                    preCost = abs(leftVal -
                                  unwrapVal[threadIdx.y][DISP_PER_THREAD - 1]
                                           [1 + threadIdx.x - 1]);
                    aftCost =
                        abs(leftVal - unwrapVal[threadIdx.y][localBest.disp + 1]
                                               [1 + threadIdx.x]);
                } else if (localBest.disp == DISP_PER_THREAD - 1) {
                    preCost =
                        abs(leftVal - unwrapVal[threadIdx.y][localBest.disp - 1]
                                               [1 + threadIdx.x]);
                    aftCost =
                        abs(leftVal -
                            unwrapVal[threadIdx.y][0][1 + threadIdx.x + 1]);
                } else {
                    preCost =
                        abs(leftVal - unwrapVal[threadIdx.y][localBest.disp - 1]
                                               [1 + threadIdx.x]);
                    aftCost =
                        abs(leftVal - unwrapVal[threadIdx.y][localBest.disp + 1]
                                               [1 + threadIdx.x]);
                }

                float denom = preCost + aftCost - 2.f * localBest.cost;
                denom = abs(denom) < 0.00001f ? 0.00001f : denom;
                dispMap.ptr(y)[j] = minDisp + dispBegin + localBest.disp +
                                    (preCost - aftCost) / (denom * 2.f);
            }
        }
    }
}

// TODO@LiuYunhuang: support for y axis stripe
void stereoMatch(const GpuMat &left, const GpuMat &right,
                 const StereoMatchParams &params, GpuMat &dispMap,
                 Stream &stream = Stream::Null()) {
    CV_Assert(!left.empty() && !right.empty());

    const int dispRange = params.maxDisp - params.minDisp;

    const int rows = left.rows;
    const int cols = left.cols;

    if (dispMap.empty()) {
        dispMap.create(rows, cols, CV_32FC1);
    } else {
        dispMap.setTo(0.f, stream);
    }

    const dim3 grid(WARPS_PER_COL, divUp(rows, BLOCK.y), 1);

    if (dispRange == 32) {
        stereo_match<1><<<grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
            left, right, params.minDisp, params.maxDisp, params.maxCost,
            params.costMinDiff, dispMap);
    }
    if (dispRange == 64) {
        stereo_match<2><<<grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
            left, right, params.minDisp, params.maxDisp, params.maxCost,
            params.costMinDiff, dispMap);
    } else if (dispRange == 128) {
        stereo_match<4><<<grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
            left, right, params.minDisp, params.maxDisp, params.maxCost,
            params.costMinDiff, dispMap);
    } else if (dispRange == 256) {
        stereo_match<8><<<grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
            left, right, params.minDisp, params.maxDisp, params.maxCost,
            params.costMinDiff, dispMap);
    } else if (dispRange == 512) {
        stereo_match<16>
            <<<grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
                left, right, params.minDisp, params.maxDisp, params.maxCost,
                params.costMinDiff, dispMap);
    } else {
        CV_LOG_INFO(nullptr,
                    "stereo_match function dosn't support current disparity.");
        CV_Assert(false);
    }
}
} // namespace cuda
} // namespace algorithm
} // namespace slmaster
