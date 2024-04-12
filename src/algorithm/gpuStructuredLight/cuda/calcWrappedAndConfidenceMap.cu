#include "common.hpp"

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {
namespace cuda {

const dim3 BLOCK(32, 8, 1);
const int PIXELS_PROCESS = 1;

template <typename SRC_TYPE>
__global__ void calc_wrapped_and_confidence_map_psp(
    const PtrStepSz<SRC_TYPE> phaseImgs, const int shiftSteps,
    PtrStep<float> wrappedMap, PtrStep<float> confidenceMap) {
    const int x = (blockDim.x * PIXELS_PROCESS) * blockIdx.x +
                  PIXELS_PROCESS * threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < phaseImgs.cols && y < phaseImgs.rows) {
        float stepShiftVal = __fdividef(CV_2PI, shiftSteps);
#pragma unroll
        for (int i = 0; i < PIXELS_PROCESS; ++i) {
            float sinusPart = 0.f, cosPart = 0.f, confidence = 0.f;
#pragma unroll
            for (int j = 0; j < shiftSteps; ++j) {
                SRC_TYPE curVal =
                    __ldg(&phaseImgs.ptr(y)[shiftSteps * (x + i) + j]);
                const float shiftVal = j * stepShiftVal;
                sinusPart += curVal * __sinf(shiftVal);
                cosPart += curVal * __cosf(shiftVal);
                confidence += curVal;
            }

            wrappedMap.ptr(y)[x + i] = -atan2(sinusPart, cosPart);
            confidenceMap.ptr(y)[x + i] = __fdividef(confidence, shiftSteps);
        }
    }
}

void calcPSPWrappedAndConfidenceMap(const GpuMat &phaseImgs, GpuMat &wrappedMap,
                                    GpuMat &confidenceMap,
                                    Stream &stream = Stream::Null()) {
    CV_Assert(phaseImgs.channels() > 2);

    const int rows = phaseImgs.rows;
    const int cols = phaseImgs.cols;

    if (wrappedMap.empty()) {
        wrappedMap.create(rows, cols, CV_32FC1);
    } else {
        wrappedMap.setTo(0.f, stream);
    }

    if (confidenceMap.empty()) {
        confidenceMap.create(rows, cols, CV_32FC1);
    } else {
        confidenceMap.setTo(0.f, stream);
    }

    const dim3 grid(divUp(cols, BLOCK.x * PIXELS_PROCESS), divUp(rows, BLOCK.y),
                    1);
    if (phaseImgs.type() == CV_8UC(phaseImgs.channels())) {
        calc_wrapped_and_confidence_map_psp<uchar>
            <<<grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
                phaseImgs, phaseImgs.channels(), wrappedMap, confidenceMap);
    } else if (phaseImgs.type() == CV_32FC(phaseImgs.channels())) {
        calc_wrapped_and_confidence_map_psp<float>
            <<<grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
                phaseImgs, phaseImgs.channels(), wrappedMap, confidenceMap);
    } else {
        CV_LOG_INFO(nullptr, "the calc_wrapped_and_confidence_map function "
                             "dosn't process it in current.\n");
        CV_Assert(false);
    }
}
} // namespace cuda
} // namespace algorithm
} // namespace slmaster
