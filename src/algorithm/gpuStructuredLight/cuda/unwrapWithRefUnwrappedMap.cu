#include "common.hpp"

#include <math.h>

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {
namespace cuda {
const dim3 BLOCK(32, 8, 1);

__global__ void unwrap_with_ref_unwrapped_map(
    const PtrStepSz<float> wrappedMap, const PtrStep<float> confidenceMap,
    const PtrStep<float> refUnwrappedMap, PtrStep<float> unwrappedMap,
    const float confidenceThresholdVal = 5.f) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > wrappedMap.cols - 1 || y > wrappedMap.rows - 1)
        return;

    if (confidenceMap.ptr(y)[x] < confidenceThresholdVal) {
        unwrappedMap.ptr(y)[x] = 0.f;
        return;
    }

    const float wrappedVal = wrappedMap.ptr(y)[x];

    unwrappedMap.ptr(y)[x] =
        wrappedVal +
        floor(__fdividef(refUnwrappedMap.ptr(y)[x] - wrappedVal - CV_PI,
                         CV_2PI)) *
            CV_2PI +
        CV_PI;
}

void unwrapWithRefUnwrappedMap(const GpuMat &wrappedMap,
                               const GpuMat &confidenceMap,
                               const GpuMat &refUnwrappedMap, GpuMat &unwrapped,
                               const float confidenceThresholdVal = 5.f,
                               Stream &stream = Stream::Null()) {
    CV_Assert(!wrappedMap.empty() && !confidenceMap.empty() &&
              !refUnwrappedMap.empty());

    const int rows = wrappedMap.rows;
    const int cols = wrappedMap.cols;

    if (unwrapped.empty()) {
        unwrapped.create(rows, cols, CV_32FC1);
    } else {
        unwrapped.setTo(0.f, stream);
    }

    const dim3 grid(divUp(cols, BLOCK.x), divUp(rows, BLOCK.y), 1);
    unwrap_with_ref_unwrapped_map<<<grid, BLOCK, 0,
                                    StreamAccessor().getStream(stream)>>>(
        wrappedMap, confidenceMap, refUnwrappedMap, unwrapped,
        confidenceThresholdVal);
}
} // namespace cuda
} // namespace algorithm
} // namespace slmaster
