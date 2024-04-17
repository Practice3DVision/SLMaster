/**
 * @file threeFrequencyHeterodyne.hpp
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __THREE_FREQUENCY_HETERODYNE_PATTERN_HPP_
#define __THREE_FREQUENCY_HETERODYNE_PATTERN_HPP_

#include <opencv2/core.hpp>

#include "structuredLight.hpp"

namespace slmaster {
namespace algorithm {
/** @brief Class implementing the three frequency heterodyne pattern,
 based on @cite Carsten Reich, et al.
 *
 *  The resulting pattern consists of three sinusoidal pattern with different frequency.
 *
 *  The entire projection sequence contains the three sinusoidal pattern sequence.
 *  For an image with a format (WIDTH, HEIGHT), a vertical sinusoidal fringe
 with a period of N has a period width of w = WIDTH / N. In order to obtain a smooth absolute phase, 
 once the base frequency is determined, the algorithm will automatically generate the other 
 two frequencies, Tbase-6 and Tbase-11, respectively.
 *
 *  For an image in 1280*720 format, a phase-shifted fringe pattern which 64 basic cycles have been selected,
  a period width of base fringe is equal 1280 / 64 = 20 and other two frequencies fringe's cycles are 64-6=58,
  64-11=53.
 * 
 */
class SLMASTER_API ThreeFrequencyHeterodynePattern : public StructuredLightPattern {
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
    static cv::Ptr<ThreeFrequencyHeterodynePattern>
    create(const ThreeFrequencyHeterodynePattern::Params &parameters =
               ThreeFrequencyHeterodynePattern::Params());
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
     * @param wrappedMaps wrappedMap of three frequency.
     * @param confidenceMap Input data to threshold gray code img, we use
     * confidence map because that we set confidence map is same as texture map,
     * A = B.
     * @param floorMap Floor map obtained through complementary graycode and
     * wrappedPhaseMap.
     */
    virtual void computeFloorMap(cv::InputArrayOfArrays wrappedMaps,
                                 cv::InputArray confidenceMap,
                                 cv::OutputArray floorMap) const = 0;
    /**
     * @brief Unwrap the wrapped phase map to remove phase ambiguities.
     * @param wrappedPhaseMap The wrapped phase map computed from the pattern.
     * @param floorMap floor map.
     * @param confidenceMap confidence map.
     * @param unwrappedPhaseMap The unwrapped phase map used to find
     * correspondences between the two devices.
     */
    virtual void unwrapPhaseMap(cv::InputArray wrappedPhaseMap, cv::InputArray floorMap,
                                cv::InputArray confidenceMap, cv::OutputArray unwrappedPhaseMap) const = 0;
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
#endif // !__THREE_FREQUENCY_HETERODYNE_PATTERN_HPP_
