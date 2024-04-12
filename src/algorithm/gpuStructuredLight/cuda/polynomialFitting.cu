#include "common.hpp"

#include <Eigen/Eigen>

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {
namespace cuda {

const dim3 BLOCK(32, 8, 1);

__global__ void polynomialFittingDev(const PtrStepSz<float> phase,
                                     const Eigen::Matrix3f intrinsic,
                                     const Eigen::Vector<float, 8> params,
                                     const float minDepth, const float maxDepth,
                                     PtrStep<float> depth) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > phase.cols - 1 || y > phase.rows - 1)
        return;

    if (abs(phase.ptr(y)[x]) < 0.001f) {
        depth.ptr(y)[x] = 0.f;
        return;
    }

    const float phaseVal = phase.ptr(y)[x];

    Eigen::Matrix3f mapL;
    mapL(0, 0) = intrinsic(0, 0);
    mapL(0, 1) = 0.f;
    mapL(0, 2) = intrinsic(0, 2) - x;
    mapL(1, 0) = 0.f;
    mapL(1, 1) = intrinsic(1, 1);
    mapL(1, 2) = intrinsic(1, 2) - y;
    mapL(2, 0) = params(0, 0) - params(4, 0) * phaseVal;
    mapL(2, 1) = params(1, 0) - params(5, 0) * phaseVal;
    mapL(2, 2) = params(2, 0) - params(6, 0) * phaseVal;

    Eigen::Vector3f mapR, camPoint;

    mapR(0, 0) = 0.f;
    mapR(1, 0) = 0.f;
    mapR(2, 0) = params(7, 0) * phaseVal - params(3, 0);

    camPoint = mapL.inverse() * mapR;

    if (camPoint(2, 0) > minDepth && camPoint(2, 0) < maxDepth)
        depth.ptr(y)[x] = camPoint(2, 0);
}

void polynomialFitting(const GpuMat &phase, const Eigen::Matrix3f &intrinsic,
                       const Eigen::Vector<float, 8> &params,
                       const float minDepth, const float maxDepth,
                       GpuMat &depth, Stream &stream = Stream::Null()) {
    CV_Assert(!phase.empty());

    const int rows = phase.rows;
    const int cols = phase.cols;

    if (depth.empty()) {
        depth.create(rows, cols, CV_32FC1);
    } else {
        depth.setTo(0.f, stream);
    }

    dim3 grid(divUp(cols, BLOCK.x), divUp(rows, BLOCK.y), 1);
    polynomialFittingDev<<<grid, BLOCK, 0,
                           StreamAccessor().getStream(stream)>>>(
        phase, intrinsic, params, minDepth, maxDepth, depth);
}
} // namespace cuda
} // namespace algorithm
} // namespace slmaster
