#include "common.hpp"

#include <Eigen/Eigen>

using namespace cv;
using namespace cv::cuda;

namespace slmaster {
namespace algorithm {
namespace cuda {
const dim3 BLOCK(32, 8, 1);

__global__ void from_depth_get_texture(const PtrStepSz<float> depth,
                                       const PtrStep<uchar3> &texture,
                                       const Eigen::Matrix3f M,
                                       const Eigen::Matrix3f R,
                                       const Eigen::Vector3f T,
                                       PtrStep<uchar3> mappedTexture) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > depth.cols - 1 || y > depth.rows - 1)
        return;

    if (abs(depth.ptr(y)[x]) < 0.001f) {
        mappedTexture.ptr(y)[x] = make_uchar3(0, 0, 0);
        return;
    }
    /*
    //no undistor because distort is small and texture isn't need so accuracy.
    float imgX = (x - M[2]) / M[0];
    float imgY = (y - M[5]) / M[1];
    const float rr = x * x + y * y;
    const float firstPart = 1 + D[0] * rr + D[1] * rr * rr + D[4] * rr * rr *
    rr; const float secondPart = imgX * imgY;
    //undistor it
    imgX = imgX * firstPart + 2 * D[2] * secondPart + D[3] * (rr + 2 * imgX *
    imgX); imgY = imgY * firstPart + D[2] * (rr + 2 * imgY * imgY) + 2 * D[3] *
    imgX * imgY;
    */
    // get camera point
    const float depthCamZ = depth.ptr(y)[x];
    const float depthCamX = (x - M(0, 2)) / M(0, 0) * depthCamZ;
    const float depthCamY = (y - M(2, 0)) / M(0, 1) * depthCamZ;
    const float textureCamX = R(0, 0) * depthCamX + R(0, 1) * depthCamY +
                              R(0, 2) * depthCamZ + T(0, 0);
    const float textureCamY = R(1, 0) * depthCamX + R(1, 1) * depthCamY +
                              R(1, 2) * depthCamZ + T(1, 0);
    const float textureCamZ = R(2, 0) * depthCamX + R(2, 1) * depthCamY +
                              R(2, 2) * depthCamZ + T(2, 0);
    const int texturePixelX =
        __fdividef(M(0, 0) * textureCamX + M(0, 2) * textureCamZ, textureCamZ);
    const int texturePixelY =
        __fdividef(M(1, 1) * textureCamY + M(1, 2) * textureCamZ, textureCamZ);

    if (texturePixelX < 0 || texturePixelX > depth.cols - 1 ||
        texturePixelY < 0 || texturePixelY > depth.rows - 1) {
        mappedTexture.ptr(y)[x] = make_uchar3(0, 0, 0);
    } else {
        mappedTexture.ptr(y)[x] = texture.ptr(texturePixelY)[texturePixelX];
    }
}

void fromDepthGetTexture(const GpuMat &depth, const GpuMat &texture,
                         const Eigen::Matrix3f &M, const Eigen::Matrix3f &R,
                         const Eigen::Vector3f &T, GpuMat &mappedTexture,
                         Stream &stream = Stream::Null()) {
    CV_Assert(!depth.empty() && !texture.empty());

    const int rows = depth.rows;
    const int cols = depth.cols;

    const dim3 grid(divUp(cols, BLOCK.x), divUp(rows, BLOCK.y), 1);
    from_depth_get_texture<<<grid, BLOCK, 0,
                             StreamAccessor().getStream(stream)>>>(
        depth, texture, M, R, T, mappedTexture);
}
} // namespace cuda
} // namespace algorithm
} // namespace slmaster