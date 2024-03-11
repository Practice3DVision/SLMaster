#ifndef CALIPACKER_H
#define CALIPACKER_H

#include <caliInfo.h>

#include "AppType.h"

class CaliPacker {
  public:
    CaliPacker();
    CaliPacker(slmaster::CaliInfo* bundleInfo) { bundleInfo_ = bundleInfo; }
    ~CaliPacker() = default;
    void bundleCaliInfo(slmaster::CaliInfo &info);
    void readIntrinsic(const std::string& filePath);
    void writeCaliInfo(const AppType::CaliType caliType, const std::string& filePath);

  private:
    slmaster::CaliInfo *bundleInfo_;
};

#endif // CALIPACKER_H
