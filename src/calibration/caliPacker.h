/**
 * @file caliPacker.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef CALIPACKER_H
#define CALIPACKER_H

#include "../cameras/caliInfo.h"

namespace slmaster {
namespace calibration {
class SLMASTER_API CaliPacker {
  public:
    CaliPacker();
    CaliPacker(slmaster::cameras::CaliInfo *bundleInfo) { bundleInfo_ = bundleInfo; }
    ~CaliPacker() = default;
    void bundleCaliInfo(slmaster::cameras::CaliInfo &info);
    void readIntrinsic(const std::string &filePath);
    void writeCaliInfo(const int caliType,
                       const std::string &filePath);

  private:
    slmaster::cameras::CaliInfo *bundleInfo_;
};
} // namespace calibration
} // namespace slmaster

#endif // CALIPACKER_H
