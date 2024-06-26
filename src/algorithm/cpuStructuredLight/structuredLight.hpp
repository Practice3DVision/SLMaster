/**
 * @file structured_light.hpp
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __STRUCTURED_LIGHT_PATTERN_HPP_
#define __STRUCTURED_LIGHT_PATTERN_HPP_

#include <opencv2/core.hpp>

#include "../../common.h"

namespace slmaster {
namespace algorithm {
//! Type of the decoding algorithm
// other algorithms can be implemented
enum {
    SINUSOIDAL_COMPLEMENTARY_GRAY_CODE =
        0, //!< Zhang Q, Su X, Xiang L, et al. 3-D shape measurement based on
           //!< complementary Gray-code light[J]. Optics and Lasers in
           //!< Engineering, 2012, 50(4): 574-579.>
    SINUSOIDAL_SHIFT_GRAY_CODE, // Wu, Z., et al. (2019). "High-speed
                                // three-dimensional shape measurement based on
                                // shifting Gray-code light." Opt Express
                                // 27(16): 22631-22644.
    INTERZONE_SINUS_FOUR_GRAYSCALE, // Wu, Z., et al. (2021). "Generalized phase
                                    // unwrapping method that avoids jump errors
                                    // for fringe projection profilometry." Opt
                                    // Express 29(17): 27181-27192.
    // He, X., et al. (2019). "Quaternary gray-code phase unwrapping for binary
    // fringe projection profilometry." Optics and Lasers in Engineering 121:
    // 358-368.
    THREE_FREQUENCY_HETERODYNE, // Carsten Reich, Reinhold Ritter, and Jan
                                // Thesing "3-D shape measurement of complex
                                // objects by combining photogrammetry and
                                // fringe projection," Optical Engineering
                                // 39(1), 2000.

};

/** @brief Abstract base class for generating and decoding structured light
 * patterns.
 */
class SLMASTER_API StructuredLightPattern : public virtual cv::Algorithm {
  public:
    /** @brief Generates the structured light pattern to project.

     @param patternImages The generated pattern: a vector<Mat>, in which each
     image is a CV_8U Mat at projector's resolution.
     */
    virtual bool generate(cv::OutputArrayOfArrays patternImages) = 0;

    /** @brief Decodes the structured light pattern, generating a disparity map

     @param patternImages The acquired pattern images to decode
     (vector<vector<Mat>>), loaded as grayscale and previously rectified.
     @param disparityMap The decoding result: a CV_64F Mat at image resolution,
     storing the computed disparity map.
     @param flags Flags setting decoding algorithms. Default:
     DECODE_3D_UNDERWORLD.
     @note All the images must be at the same resolution.
     */
    virtual bool
    decode(const std::vector<std::vector<cv::Mat>> &patternImages,
           cv::OutputArray disparityMap,
           int flags = SINUSOIDAL_COMPLEMENTARY_GRAY_CODE) const = 0;
};

//! @}

} // namespace algorithm
} // namespace slmaster
#endif
