/**
 * @file algorithm.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __SLMASTER_ALGORITHM_H_
#define __SLMASTER_ALGORITHM_H_

#include "cpuStructuredLight/defocusMethod.h"
#include "cpuStructuredLight/sinus_comple_graycode_pattern.hpp"
#include "cpuStructuredLight/structured_light.hpp"
#include "cpuStructuredLight/recoverDepth.h"

#ifdef OPENCV_WITH_CUDA_MODULE
#include "gpuStructuredLight/cuda_multi_view_stereo_geometry_pattern.hpp"
#include "gpuStructuredLight/cuda_sinus_comple_graycode_pattern.hpp"
#include "gpuStructuredLight/cudastructuredlight.hpp"
#endif

#endif //! __SLMASTER_ALGORITHM_H_