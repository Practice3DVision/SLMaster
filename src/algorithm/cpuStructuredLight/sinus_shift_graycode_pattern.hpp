/**
 * @file sinus_shift_graycode_pattern.hpp
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-04-11
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __OPENCV_SINUSOIDAL_SHIFT_GRAYCODE_HPP__
#define __OPENCV_SINUSOIDAL_SHIFT_GRAYCODE_HPP__

#include <opencv2/core.hpp>

#include "structured_light.hpp"

namespace slmaster {
namespace algorithm {
/** @brief Class implementing the Sinusoidal Shift Gray-code pattern,
 based on @cite Wu, Z., et al.
 *
 *  The resulting pattern consists of a sinusoidal pattern and a shift Gray code
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
 *  the required number of Gray code patterns being log 2 (32) = 5. The
 algorithm can be applied to sinusoidal complementary Gray codes of any number
 of steps and any bits,
 *  as long as their period widths satisfy the above principle.
 */
class SLMASTER_API SinusShiftGrayCodePattern : public StructuredLightPattern {
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
    };

    /** @brief Constructor
     @param parameters SinusCompleGrayCodePattern parameters
     SinusCompleGrayCodePattern::Params: the width and the height of the
     projector.
     */
    static cv::Ptr<SinusShiftGrayCodePattern>
    create(const SinusShiftGrayCodePattern::Params &parameters =
               SinusShiftGrayCodePattern::Params());
    /**
     * @brief Compute a confidence map from sinusoidal patterns.
     * @param patternImages Input data to compute the confidence map.
     * @param confidenceMap confidence map obtained through PSP.
     */
    virtual void computeConfidenceMap(cv::InputArrayOfArrays patternImages,
                                      cv::OutputArray confidenceMap) const = 0;
    /**
     * @brief Compute a wrapped phase map from sinusoidal patterns.
     * @param patternImages Input data to compute the wrapped phase map.
     * @param wrappedPhaseMap Wrapped phase map obtained through PSP.
     */
    virtual void computePhaseMap(cv::InputArrayOfArrays patternImages,
                                 cv::OutputArray wrappedPhaseMap) const = 0;
    /**
     * @brief Compute a floor map from complementary graycode patterns and
     * wrappedPhaseMap.
     * @param patternImages Input data to compute the floor map.
     * @param confidenceMap Input data to threshold gray code img, we use
     * confidence map because that we set confidence map is same as texture map,
     * A = B.
     * @param wrappedPhaseMap Input data to help us select K1 or K2.
     * @param floorMap Floor map obtained through complementary graycode and
     * wrappedPhaseMap.
     */
    virtual void computeFloorMap(cv::InputArrayOfArrays patternImages,
                                 cv::InputArray confidenceMap,
                                 cv::InputArray wrappedPhaseMap,
                                 cv::OutputArray floorMap) const = 0;
    /**
     * @brief Unwrap the wrapped phase map to remove phase ambiguities.
     * @param wrappedPhaseMap The wrapped phase map computed from the pattern.
     * @param unwrappedPhaseMap The unwrapped phase map used to find
     * correspondences between the two devices.
     * @param shadowMask Mask used to discard shadow regions.
     * @param confidenceThreshod confidence threshod to discard invalid data.
     */
    virtual void
    unwrapPhaseMap(cv::InputArray wrappedPhaseMap, cv::InputArray floorMap,
                   cv::OutputArray unwrappedPhaseMap,
                   cv::InputArray shadowMask = cv::noArray()) const = 0;
    /**
     * @brief compute disparity from left unwrap map and right unwrap map.
     * @param lhsUnwrapMap left unwrap map.
     * @param rhsUnwrapMap right unwrap map.
     * @param disparityMap dispairty map that computed.
     */
    virtual void computeDisparity(cv::InputArray lhsUnwrapMap,
                                  cv::InputArray rhsUnwrapMap,
                                  cv::OutputArray disparityMap) const = 0;
};
} // namespace algorithm
} // namespace slmaster
#endif
