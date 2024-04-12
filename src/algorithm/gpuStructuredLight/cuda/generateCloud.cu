#include "common.hpp"

#include <Eigen/Eigen>

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {
namespace cuda {

const dim3 BLOCK(32, 8, 1);

__global__ void generate_cloud(const PtrStepSz<float> depth, Eigen::Matrix3f M,
                               const float minDepth, const float maxDepth,
                               float3 *cloud) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = depth.cols * y + x;

    if (x > depth.cols - 1 || y > depth.rows - 1)
        return;

    if (abs(depth.ptr(y)[x]) < 0.001f) {
        cloud[id] = make_float3(NAN, NAN, NAN);
        return;
    }

    // get camera point
    const float depthCamZ = depth.ptr(y)[x];
    const float depthCamX = (x - M(0, 2)) / M(0, 0) * depthCamZ;
    const float depthCamY = (y - M(1, 2)) / M(1, 1) * depthCamZ;

    cloud[id] = make_float3(depthCamX, depthCamY, depthCamZ);
}

void generateCloud(const GpuMat &depth, const Eigen::Matrix3f &M,
                   const float minDepth, const float maxDepth, float3 *cloud,
                   Stream &stream = Stream::Null()) {
    CV_Assert(!depth.empty());

    const int rows = depth.rows;
    const int cols = depth.cols;

    if (cloud == nullptr) {
        cudaMallocAsync(&cloud, sizeof(float3) * rows * cols,
                        StreamAccessor().getStream(stream));
    }

    const dim3 grid(divUp(cols, BLOCK.x), divUp(rows, BLOCK.y), 1);
    generate_cloud<<<grid, BLOCK, 0, StreamAccessor().getStream(stream)>>>(
        depth, M, minDepth, maxDepth, cloud);
}
} // namespace cuda
} // namespace algorithm
} // namespace slmaster