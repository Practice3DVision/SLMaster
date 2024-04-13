/**
 * @file cudastructuredlight.hpp
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __CUDA_STRUCTURED_LIGHT_HPP_
#define __CUDA_STRUCTURED_LIGHT_HPP_

#ifndef __cplusplus
#error cudastructuredlight.hpp header must be complier as C++
#endif

#include <opencv2/core/cuda.hpp>

#include "../../common.h"
#include "cuda/cuda.hpp"

namespace slmaster {
namespace algorithm {
//! Type of the decoding algorithm
// other algorithms can be implemented
enum {
    SINUSOIDAL_COMPLEMENTARY_GRAY_CODE_GPU =
        0, //!< Zhang Q, Su X, Xiang L, et al. 3-D shape measurement based on
           //!< complementary Gray-code light[J]. Optics and Lasers in
           //!< Engineering, 2012, 50(4): 574-579.>
    MULTI_VIEW_STEREO_GEOMETRY_GPU, // Willomitzer, F. and G. Häusler (2017).
                                // "Single-shot 3D motion picture camera with a
                                // dense point cloud(Article)." Optics Express
                                // Vol.25(No.19): 23451-23464. An, Y., et al.
                                // (2016). "Pixel-wise absolute phase unwrapping
                                // using geometric constraints of structured
                                // light system." Opt Express 24(16):
                                // 18445-18459. Li, Z., et al. (2013).
                                // "Multiview phase shifting: a full-resolution
                                // and high-speed 3D measurement framework for
                                // arbitrary shape dynamic objects." Opt Lett
                                // 38(9): 1389-1391. Willomitzer, F., et al.
                                // (2015). "Single-shot three-dimensional
                                // sensing with improved data density." Applied
                                // Optics 54(3).

};

/** @brief Abstract base class for generating and decoding structured light
 * patterns.
 */
class SLMASTER_API StructuredLightPatternGPU : public virtual cv::Algorithm {
  public:
    /** @brief Generates the structured light pattern to project.

     @param patternImages The generated pattern: a vector<Mat>, in which each
     image is a CV_8U Mat at projector's resolution.
     */
    virtual bool generate(std::vector<cv::Mat> &patternImages) = 0;

    /** @brief Decodes the structured light pattern, generating a disparity map

     @param patternImages The acquired pattern images to decode
     (vector<vector<GpuMat>>), loaded as grayscale and previously rectified.
     @param disparityMap The decoding result: a CV_64F Mat at image resolution,
     storing the computed disparity map.
     @param flags Flags setting decoding algorithms. Default:
     SINUSOIDAL_COMPLEMENTARY_GRAY_CODE.
     @note All the images must be at the same resolution.
     @param stream CUDA asynchronous streams.
     */
    virtual bool
    decode(const std::vector<std::vector<cv::Mat>> &patternImages,
           cv::Mat &disparityMap,
           const int flags = SINUSOIDAL_COMPLEMENTARY_GRAY_CODE_GPU,
           cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const = 0;
};

} // namespace algorithm
} // namespace slmaster

#endif //!__CUDA_STRUCTURED_LIGHT_HPP__
