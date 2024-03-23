/**
 * @file common.hpp
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __CUDASTRUCTUREDLIGHT_COMMON_H__
#define __CUDASTRUCTUREDLIGHT_COMMON_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace slmaster {
namespace algorithm {
namespace cuda {
/**
 * @brief cuda stereo match controll params
 *
 */
struct StereoMatchParams {
    StereoMatchParams()
        : minDisp(0), maxDisp(256), maxCost(0.1f), costMinDiff(0.02f) {}
    int minDisp;
    int maxDisp;
    float maxCost;
    float costMinDiff;
};
} // namespace cuda
} // namespace algorithm
} // namespace slmaster

#endif // !__CUDASTRUCTUREDLIGHT_COMMON_H__