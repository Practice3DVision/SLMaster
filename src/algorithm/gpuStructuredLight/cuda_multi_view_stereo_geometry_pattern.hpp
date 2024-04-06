/**
 * @file cuda_multi_view_stereo_geometry_pattern.hpp
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __OPENCV_CUDA_MUTIL_VIEW_STEREO_GEOMETRY_GPU_HPP__
#define __OPENCV_CUDA_MUTIL_VIEW_STEREO_GEOMETRY_GPU_HPP__

#include "../../common.h"
#include "cudastructuredlight.hpp"
#include "opencv2/core/cuda.hpp"

#include <Eigen/Eigen>

namespace slmaster {
namespace algorithm {
/** @brief Class implementing the Multi-view Stereo Geometry pattern,
 based on @cite Willomitzer.
 *
 *  The resulting pattern consists of a sinusoidal pattern.
 *
 *  The equipment is mainly composed of a camera 1 with a very small angle
 with the projector and two other cameras 2 and 3 with a large angle with the
 projector.
 *
 *  The entire projection sequence contains the sine fringe sequence.
 For an image with a format (WIDTH, HEIGHT), a vertical sinusoidal fringe
 with a period of N has a period width of w = WIDTH / N.Firstly, the algorithm
 uses the reference phase solution method of An, Y et al. to quickly dephase
 the wrapping phase, then uses the polynomial fitting method to quickly recover
 the depth, and finally uses the multi-view stereo geometric constraint to
 optimize the coarser depth again: removing the background or improving the
 accuracy.
 *
 */
class SLMASTER_API MultiViewStereoGeometryPatternGPU
    : public StructuredLightPatternGPU {
  public:
    /** @brief Parameters of StructuredLightPattern constructor.
     *  @param width Projector's width. Default value is 1280.
     *  @param height Projector's height. Default value is 720.
     */
    struct SLMASTER_API Params {
        Params();
        bool horizontal;
        int width;
        int height;
        int nbrOfPeriods;
        int shiftTime;
        int minDisparity;
        int maxDisparity;
        float confidenceThreshold;
        float maxCost;
        float costMinDiff;
        float costMaxDiff;
        float minDepth;
        float maxDepth;
        cv::cuda::GpuMat refUnwrappedMap;
        Eigen::Matrix4f PL1;
        Eigen::Matrix4f PR4;
        Eigen::Matrix4f PR2;
        Eigen::Matrix3f M1;
        Eigen::Matrix3f M2;
        Eigen::Matrix3f M3;
        Eigen::Matrix3f R12;
        Eigen::Matrix3f R13;
        Eigen::Vector3f T12;
        Eigen::Vector3f T13;
    };

    /** @brief Constructor
     @param parameters MultiViewStereoGeometryPattern parameters
     MultiViewStereoGeometryPattern::Params: the width and the height of the
     projector.
     */
    static cv::Ptr<MultiViewStereoGeometryPatternGPU>
    create(const MultiViewStereoGeometryPatternGPU::Params &parameters =
               MultiViewStereoGeometryPatternGPU::Params());
    /**
     * @brief Compute a wrapped phase map from sinusoidal patterns.
     * @param patternImages Input data to compute the wrapped phase map.
     * @param wrappedPhaseMap Wrapped phase map obtained through PSP.
     * @param confidenceMap Phase modulation diagram.
     * @param stream CUDA asynchronous streams.
     */
    virtual void computeWrappedAndConfidenceMap(
        const cv::cuda::GpuMat &patternImages,
        cv::cuda::GpuMat &wrappedPhaseMap, cv::cuda::GpuMat &confidenceMap,
        cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const = 0;
    /**
     * @brief Unwrap the wrapped phase map to remove phase ambiguities.
     * @param refUnwrapMap Refer to the absolute phase
     * @param wrappedPhaseMap The wrapped phase map computed from the pattern.
     * @param confidenceMap Phase modulation diagram.
     * @param unwrappedPhaseMap Noisy and coarse absolute phase map.
     * @param stream CUDA asynchronous streams.
     */
    virtual void unwrapPhaseMap(
        const cv::cuda::GpuMat &refUnwapMap,
        const cv::cuda::GpuMat &wrappedPhaseMap,
        const cv::cuda::GpuMat &confidenceMap,
        cv::cuda::GpuMat &unwrappedPhaseMap,
        cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const = 0;
    /**
     * @brief Use polynomials to fit the structured light system model and
     * recover the rough depth from the absolute phase map.
     *
     * @param unwrappedPhaseMap Noisy and coarse absolute phase map.
     * @param coarseDepthMap Noisy and coarse depth map.
     * @param stream CUDA asynchronous streams.
     */
    virtual void polynomialFitting(
        const cv::cuda::GpuMat &unwrappedPhaseMap,
        cv::cuda::GpuMat &coarseDepthMap,
        cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const = 0;

    /**
     * @brief Use multi-view stereo geometry constraints to remove rough depth
     * map noise or optimize accuracy.
     * @param coarseDepthMap Noisy and coarse depth map.
     * @param wrappedMap1 Wrap phase of camera 1.
     * @param wrappedMap2 Wrap phase of camera 2.
     * @param wrappedMap3 Wrap phase of camera 3.
     * @param refineDepthMap Extremely low-noise and accuracy-optimized depth
     * maps.
     * @param stream CUDA asynchronous streams.
     */
    CV_WRAP
    virtual void multiViewStereoRefineDepth(
        const cv::cuda::GpuMat &coarseDepthMap,
        const cv::cuda::GpuMat &wrappedMap1,
        const cv::cuda::GpuMat &confidenceMap1,
        const cv::cuda::GpuMat &wrappedMap2,
        const cv::cuda::GpuMat &confidenceMap2,
        const cv::cuda::GpuMat &wrappedMap3,
        const cv::cuda::GpuMat &confidenceMap3,
        cv::cuda::GpuMat &refineDepthMap,
        cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const = 0;
};

} // namespace algorithm
} // namespace slmaster
#endif //!__OPENCV_CUDA_MUTIL_VIEW_STEREO_GEOMETRY_GPU_HPP__
