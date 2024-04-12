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
#include "cpuStructuredLight/sinusCompleGraycodePattern.hpp"
#include "cpuStructuredLight/sinusShiftGraycodePattern.hpp"
#include "cpuStructuredLight/interzonePhaseUnwrappingPattern.hpp"
#include "cpuStructuredLight/structuredLight.hpp"
#include "cpuStructuredLight/recoverDepth.h"
#include "cpuStructuredLight/lasterLine.h"

#ifdef OPENCV_WITH_CUDA_MODULE
#include "gpuStructuredLight/cudaMultiViewStereoGeometryPattern.hpp"
#include "gpuStructuredLight/cudaSinusCompleGraycodePattern.hpp"
#include "gpuStructuredLight/cudaStructuredLight.hpp"
#endif

#endif //! __SLMASTER_ALGORITHM_H_