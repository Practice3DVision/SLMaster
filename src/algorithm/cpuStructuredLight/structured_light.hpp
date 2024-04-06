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

#ifndef __OPENCV_STRUCTURED_LIGHT_HPP__
#define __OPENCV_STRUCTURED_LIGHT_HPP__

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
     @param blackImages The all-black images needed for shadowMasks computation.
     @param whiteImages The all-white images needed for shadowMasks computation.
     @param flags Flags setting decoding algorithms. Default:
     DECODE_3D_UNDERWORLD.
     @note All the images must be at the same resolution.
     */
    virtual bool
    decode(const std::vector<std::vector<cv::Mat>> &patternImages,
           cv::OutputArray disparityMap,
           cv::InputArrayOfArrays blackImages = cv::noArray(),
           cv::InputArrayOfArrays whiteImages = cv::noArray(),
           int flags = SINUSOIDAL_COMPLEMENTARY_GRAY_CODE) const = 0;
};

//! @}

} // namespace algorithm
} // namespace slmaster
#endif
