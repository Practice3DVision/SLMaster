#include "common.hpp"

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {
namespace cuda {

const dim3 BLOCK(32, 8, 1);
const int PIXELS_PROCESS = 1;

template <typename SRC_TYPE>
__global__ void unwrap_sinus_comple_graycode(const PtrStepSz<SRC_TYPE> grayImgs,
                                             const PtrStep<float> wrappedMap,
                                             const PtrStep<float> confidenceMap,
                                             const int grayNums,
                                             const float confidenceThreshold,
                                             PtrStep<float> unwrappedMap) {
    const int x = (blockDim.x * PIXELS_PROCESS) * blockIdx.x +
                  PIXELS_PROCESS * threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < grayImgs.cols && y < grayImgs.rows) {
        const float PI_DIV_2 = __fdividef((float)CV_PI, 2.f);
#pragma unroll
        for (int i = 0; i < PIXELS_PROCESS; ++i) {
            const int curX = x + i;
            const float confidenceVal = __ldg(&confidenceMap.ptr(y)[curX]);
            const float wrapVal = __ldg(&wrappedMap.ptr(y)[curX]);

            if (confidenceVal < confidenceThreshold) {
                unwrappedMap.ptr(y)[curX] = 0.f;
                continue;
            }

            uint16_t bitVal = 0, K1 = 0, maxK = (1 << (grayNums - 1)) - 1;
#pragma unroll
            for (int j = 0; j < grayNums - 1; ++j) {
                SRC_TYPE imgVal = __ldg(&grayImgs.ptr(y)[grayNums * curX + j]);
                bitVal ^= (imgVal > confidenceVal);
                K1 = (K1 << 1) + bitVal;
            }

            if (K1 > maxK) {
                unwrappedMap.ptr(y)[curX] = 0.f;
                continue;
            }

            SRC_TYPE imgVal =
                __ldg(&grayImgs.ptr(y)[grayNums * (curX + 1) - 1]);
            bitVal ^= (imgVal > confidenceVal);
            uint16_t K2 = ((K1 << 1) + bitVal + 1) / 2;

            if (wrapVal >= -PI_DIV_2 && wrapVal <= PI_DIV_2) {
                unwrappedMap.ptr(y)[curX] = wrapVal + CV_2PI * K1 + CV_PI;
            } else if (wrapVal > PI_DIV_2) {
                unwrappedMap.ptr(y)[curX] = wrapVal + CV_2PI * (K2 - 1) + CV_PI;
            } else {
                unwrappedMap.ptr(y)[curX] = wrapVal + CV_2PI * K2 + CV_PI;
            }
        }
    }
}

void unwrapSinusCompleGraycodeMap(const GpuMat &grayImgs,
                                  const GpuMat &wrappedMap,
                                  const GpuMat &confidenceMap,
                                  GpuMat &unwrappedMap,
                                  const float confidenceThreshold = 0.f,
                                  Stream &stream = Stream::Null()) {
    CV_Assert(grayImgs.channels() > 2 && !confidenceMap.empty() &&
              confidenceMap.size() == grayImgs.size());

    const int rows = grayImgs.rows;
    const int cols = grayImgs.cols;

    if (unwrappedMap.empty()) {
        unwrappedMap.create(rows, cols, CV_32FC1);
    } else {
        unwrappedMap.setTo(0.f, stream);
    }

    const dim3 grid(divUp(cols, BLOCK.x * PIXELS_PROCESS), divUp(rows, BLOCK.y),
                    1);
    if (grayImgs.type() == CV_8UC(grayImgs.channels())) {
        unwrap_sinus_comple_graycode<uchar>
            <<<grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
                grayImgs, wrappedMap, confidenceMap, grayImgs.channels(),
                confidenceThreshold, unwrappedMap);
    } else if (grayImgs.type() == CV_32FC(grayImgs.channels())) {
        unwrap_sinus_comple_graycode<float>
            <<<grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
                grayImgs, wrappedMap, confidenceMap, grayImgs.channels(),
                confidenceThreshold, unwrappedMap);
    } else {
        CV_LOG_INFO(nullptr, "the calc_comple_graycode_floor_map function "
                             "dosn't process it in current.\n");
        CV_Assert(false);
    }
}
} // namespace cuda
} // namespace algorithm
} // namespace slmaster
