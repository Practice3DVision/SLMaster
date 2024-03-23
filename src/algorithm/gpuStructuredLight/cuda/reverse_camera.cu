#include "common.hpp"

#include <Eigen/Eigen>

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {
namespace cuda {

const dim3 BLOCK(32, 8, 1);

__global__ void
reverseCameraVerticalDev(const PtrStepSz<float> phase, const Eigen::Matrix4f PL,
                         const Eigen::Matrix4f PR, const float minDepth,
                         const float maxDepth, const float pitch,
                         PtrStep<float> depth) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > phase.cols - 1 || y > phase.rows - 1)
        return;

    if (abs(phase.ptr(y)[x]) < 0.001f) {
        depth.ptr(y)[x] = 0.f;
        return;
    }

    float up = (phase.ptr(y)[x] / CV_2PI) * pitch;

    Eigen::Matrix3f mapL;
    mapL << PL(0, 0) - PL(2, 0) * x, PL(0, 1) - PL(2, 1) * x,
        PL(0, 2) - PL(2, 2) * x, PL(1, 0) - PL(2, 0) * y,
        PL(1, 1) - PL(2, 1) * y, PL(1, 2) - PL(2, 2) * y,
        PR(0, 0) - PR(2, 0) * up, PR(0, 1) - PR(2, 1) * up,
        PR(0, 2) - PR(2, 2) * up;

    Eigen::Vector3f mapR;
    mapR << PL(2, 3) * x - PL(0, 3), PL(2, 3) * y - PL(1, 3),
        PR(2, 3) * up - PR(0, 3);

    Eigen::Vector3f camPoint = mapL.inverse() * mapR;

    if (camPoint(2, 0) > minDepth && camPoint(2, 0) < maxDepth)
        depth.ptr(y)[x] = camPoint(2, 0);
}

__global__ void
reverseCameraHonrizonDev(const PtrStepSz<float> phase, const Eigen::Matrix4f PL,
                         const Eigen::Matrix4f PR, const float minDepth,
                         const float maxDepth, const float pitch,
                         PtrStep<float> depth) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > phase.cols - 1 || y > phase.rows - 1)
        return;

    if (abs(phase.ptr(y)[x]) < 0.001f) {
        depth.ptr(y)[x] = 0.f;
        return;
    }

    float up = (phase.ptr(y)[x] / CV_2PI) * pitch;

    Eigen::Matrix3f mapL;
    mapL << PL(0, 0) - PL(2, 0) * x, PL(0, 1) - PL(2, 1) * x,
        PL(0, 2) - PL(2, 2) * x, PL(1, 0) - PL(2, 0) * y,
        PL(1, 1) - PL(2, 1) * y, PL(1, 2) - PL(2, 2) * y,
        PR(1, 0) - PR(2, 0) * up, PR(1, 1) - PR(2, 1) * up,
        PR(1, 2) - PR(2, 2) * up;

    Eigen::Vector3f mapR;
    mapR << PL(2, 3) * x - PL(0, 3), PL(2, 3) * y - PL(1, 3),
        PR(2, 3) * up - PR(1, 3);

    Eigen::Vector3f camPoint = mapL.inverse() * mapR;

    if (camPoint(2, 0) > minDepth && camPoint(2, 0) < maxDepth)
        depth.ptr(y)[x] = camPoint(2, 0);
}

void reverseCamera(const GpuMat &phase, const Eigen::Matrix4f &PL,
                   const Eigen::Matrix4f &PR, const float minDepth,
                   const float maxDepth, const float pitch, GpuMat &depth,
                   const bool isHonrizon = false,
                   Stream &stream = Stream::Null()) {
    CV_Assert(!phase.empty());

    const int rows = phase.rows;
    const int cols = phase.cols;

    if (depth.empty()) {
        depth.create(rows, cols, CV_32FC1);
    } else {
        depth.setTo(0.f, stream);
    }

    dim3 grid(divUp(cols, BLOCK.x), divUp(rows, BLOCK.y), 1);

    if (isHonrizon) {
        reverseCameraHonrizonDev<<<grid, BLOCK, 0,
                                   StreamAccessor().getStream(stream)>>>(
            phase, PL, PR, minDepth, maxDepth, pitch, depth);
    } else {
        reverseCameraVerticalDev<<<grid, BLOCK, 0,
                                   StreamAccessor().getStream(stream)>>>(
            phase, PL, PR, minDepth, maxDepth, pitch, depth);
    }
}
} // namespace cuda
} // namespace algorithm
} // namespace slmaster
