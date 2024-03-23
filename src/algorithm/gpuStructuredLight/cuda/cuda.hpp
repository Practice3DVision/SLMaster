/**
 * @file cuda.hpp
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __CUDASTRUCTUREDLIGHT_CUDA_HPP__
#define __CUDASTRUCTUREDLIGHT_CUDA_HPP__

#include "common.hpp"

#include <Eigen/Eigen>

namespace slmaster {
namespace algorithm {
namespace cuda {
void calcPSPWrappedAndConfidenceMap(
    const cv::cuda::GpuMat &phaseImgs, cv::cuda::GpuMat &wrappedMap,
    cv::cuda::GpuMat &confidenceMap,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null());

void unwrapSinusCompleGraycodeMap(
    const cv::cuda::GpuMat &grayImgs, const cv::cuda::GpuMat &wrappedMap,
    const cv::cuda::GpuMat &confidenceMap, cv::cuda::GpuMat &unwrappedMap,
    const float confidenceThreshold = 0.f,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null());

void stereoMatch(const cv::cuda::GpuMat &left, const cv::cuda::GpuMat &right,
                 const StereoMatchParams &params, cv::cuda::GpuMat &dispMap,
                 cv::cuda::Stream &stream = cv::cuda::Stream::Null());

void polynomialFitting(const cv::cuda::GpuMat &phase,
                       const Eigen::Matrix3f &intrinsic,
                       const Eigen::Vector<float, 8> &params,
                       const float minDepth, const float maxDepth,
                       cv::cuda::GpuMat &depth,
                       cv::cuda::Stream &stream = cv::cuda::Stream::Null());

void unwrapWithRefUnwrappedMap(
    const cv::cuda::GpuMat &wrappedMap, const cv::cuda::GpuMat &confidenceMap,
    const cv::cuda::GpuMat &refUnwrappedMap, cv::cuda::GpuMat &unwrapped,
    const float confidenceThresholdVal = 5.f,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null());

void fromDepthGetTexture(const cv::cuda::GpuMat &depth,
                         const cv::cuda::GpuMat &texture,
                         const Eigen::Matrix3f &M, const Eigen::Matrix3f &R,
                         const Eigen::Vector3f &T,
                         cv::cuda::GpuMat &mappedTexture,
                         cv::cuda::Stream &stream = cv::cuda::Stream::Null());

void generateCloud(const cv::cuda::GpuMat &depth, const Eigen::Matrix3f &M,
                   const float minDepth, const float maxDepth, float3 *cloud,
                   cv::cuda::Stream &stream = cv::cuda::Stream::Null());

void reverseCamera(const cv::cuda::GpuMat &phase, const Eigen::Matrix4f &PL,
                   const Eigen::Matrix4f &PR, const float minDepth,
                   const float maxDepth, const float pitch,
                   cv::cuda::GpuMat &depth, const bool isHonrizon = false,
                   cv::cuda::Stream &stream = cv::cuda::Stream::Null());

void multiViewStereoGeometry(
    const cv::cuda::GpuMat &coarseDepthMap, const const Eigen::Matrix3f &M1,
    const cv::cuda::GpuMat &wrappedMap1, const cv::cuda::GpuMat &confidenceMap1,
    const Eigen::Matrix3f &M2, const Eigen::Matrix3f &R12,
    const Eigen::Vector3f &T12, const cv::cuda::GpuMat &wrappedMap2,
    const cv::cuda::GpuMat &confidenceMap2, const Eigen::Matrix3f &M3,
    const Eigen::Matrix3f &R13, const Eigen::Vector3f &T13,
    const cv::cuda::GpuMat &wrappedMap3, const cv::cuda::GpuMat &confidenceMap3,
    const Eigen::Matrix4f &PL, const Eigen::Matrix4f &PR,
    cv::cuda::GpuMat &fineDepthMap, const float confidenceThreshold = 5.f,
    const float maxCost = 0.01f, const bool isHonrizon = false,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null());
} // namespace cuda
} // namespace algorithm
} // namespace slmaster

#endif // !__CUDASTRUCTUREDLIGHT_CUDA_HPP__
