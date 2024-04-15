/**
 * @file cameras.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-23
 * 
 * @copyright Copyright (c) 2024
 * 
 */

 #ifndef __SLMASTER_CAMERAS_H_
 #define __SLMASTER_CAMERAS_H_

 #include "caliInfo.h"
 #include "tool.h"
 #include "pattern.h"
 #include "slCamera.h"
 #include "slCameraFactory.h"

 #include "monocular/monocularCamera.h"
 #include "monocular/monoSinusCompleGrayCodePattern.h"
 #include "monocular/monoSinusShiftGrayCodePattern.h"
 #include "monocular/monoInterzoneSinusFourGrayscalePattern.h"

 #include "binocular/binoocularCamera.h"
 #include "binocular/binosSinusCompleGrayCodePattern.h"
 #include "binocular/binosSinusShiftGrayCodePattern.h"
 #include "binocular/binosInterzoneSinusFourGrayscalePattern.h"

 #include "trinocular/trinocularCamera.h"
 #include "trinocular/trinocularMultiViewStereoGeometryPattern.h"

 #endif //!__SLMASTER_CAMERAS_H_