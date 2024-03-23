/**
 * @file cuda_sinus_comple_graycode_pattern.hpp
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __OPENCV_CUDA_SINUSOIDAL_COMPLEMENTARY_GRAYCODE_HPP__
#define __OPENCV_CUDA_SINUSOIDAL_COMPLEMENTARY_GRAYCODE_HPP__

#include "cudastructuredlight.hpp"

namespace slmaster {
namespace algorithm {
/** @brief Class implementing the Sinusoidal Complementary Gray-code pattern,
 based on @cite Zhang Q.
 *
 *  The resulting pattern consists of a sinusoidal pattern and a Gray code
 pattern corresponding to the period width.
 *
 *  The entire projection sequence contains the sine fringe sequence and the
 Gray code fringe sequence.
 *  For an image with a format (WIDTH, HEIGHT), a vertical sinusoidal fringe
 with a period of N has a period width of
 *  w = WIDTH / N.The algorithm uses log 2 (N) vertical Gray code patterns and
 an additional sequence Gray code pattern to
 *  avoid edge dephasing errors caused by the sampling theorem.The same goes for
 horizontal stripes.

 *
 *  For an image in 1280*720 format, a phase-shifted fringe pattern needs to be
 generated for 32 periods, with a period width of 1280 / 32 = 40 and
 *  the required number of Gray code patterns being log 2 (32) = 5 + 1 = 6.the
 algorithm can be applied to sinusoidal complementary Gray codes of any number
 of steps and any bits,
 *  as long as their period widths satisfy the above principle.
 */
class SLMASTER_API SinusCompleGrayCodePatternGPU : public StructuredLightPatternGPU {
  public:
    /** @brief Parameters of StructuredLightPattern constructor.
     *  @param width Projector's width. Default value is 1280.
     *  @param height Projector's height. Default value is 720.
     */
    struct SLMASTER_API Params {
        Params();
        int width;
        int height;
        int nbrOfPeriods;
        int shiftTime;
        int minDisparity;
        int maxDisparity;
        bool horizontal;
        float confidenceThreshold;
        float maxCost;
        float costMinDiff;
    };

    /** @brief Constructor
     @param parameters SinusCompleGrayCodePattern parameters
     SinusCompleGrayCodePattern::Params: the width and the height of the
     projector.
     */
    static cv::Ptr<SinusCompleGrayCodePatternGPU>
    create(const SinusCompleGrayCodePatternGPU::Params &parameters =
               SinusCompleGrayCodePatternGPU::Params());
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
     * @param grayImgs The Gray code imgs
     * @param wrappedPhaseMap The wrapped phase map computed from the pattern.
     * @param confidenceMap Phase modulation diagram.
     * @param unwrappedPhaseMap The unwrapped phase map used to find
     * correspondences between the two devices.
     * @param confidenceThreshod confidence threshod to discard invalid data.
     * @param stream CUDA asynchronous streams.
     */
    virtual void unwrapPhaseMap(
        const cv::cuda::GpuMat &grayImgs,
        const cv::cuda::GpuMat &wrappedPhaseMap,
        const cv::cuda::GpuMat &confidenceMap,
        cv::cuda::GpuMat &unwrappedPhaseMap,
        const float confidenceThreshold = 0.f,
        cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const = 0;
    /**
     * @brief compute disparity from left unwrap map and right unwrap map.
     * @param lhsUnwrapMap left unwrap map.
     * @param rhsUnwrapMap right unwrap map.
     * @param disparityMap dispairty map that computed.
     * @param stream CUDA asynchronous streams.
     */
    virtual void computeDisparity(
        const cv::cuda::GpuMat &lhsUnwrapMap,
        const cv::cuda::GpuMat &rhsUnwrapMap, cv::cuda::GpuMat &disparityMap,
        cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const = 0;
};

} // namespace algorithm
} // namespace slmaster
#endif //!__OPENCV_CUDA_SINUSOIDAL_COMPLEMENTARY_GRAYCODE_HPP__
