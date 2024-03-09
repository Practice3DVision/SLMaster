#include "caliPacker.h"

using namespace slmaster;

CaliPacker::CaliPacker() : bundleInfo_(nullptr) {}

void CaliPacker::bundleCaliInfo(CaliInfo &info) { bundleInfo_ = &info; }

void CaliPacker::readIntrinsic(const std::string& filePath) {
    cv::FileStorage readYml(filePath, cv::FileStorage::READ);
    readYml["M1"] >> bundleInfo_->info_.M1_;
    readYml["D1"] >> bundleInfo_->info_.D1_;
    readYml["M2"] >> bundleInfo_->info_.M2_;
    readYml["D2"] >> bundleInfo_->info_.D2_;
    readYml.release();
}

void CaliPacker::writeCaliInfo(const AppType::CaliType caliType, const std::string& filePath) {
    cv::FileStorage caliInfoOutPut(filePath, cv::FileStorage::WRITE | cv::FileStorage::APPEND);
    caliInfoOutPut << "M1" << bundleInfo_->info_.M1_;
    caliInfoOutPut << "D1" << bundleInfo_->info_.D1_;
    caliInfoOutPut << "S" << bundleInfo_->info_.S_;

    if (caliType == AppType::CaliType::Stereo) {
        caliInfoOutPut << "M2" << bundleInfo_->info_.M2_;
        caliInfoOutPut << "D2" << bundleInfo_->info_.D2_;
        caliInfoOutPut << "R1" << bundleInfo_->info_.R1_;
        caliInfoOutPut << "P1" << bundleInfo_->info_.P1_;
        caliInfoOutPut << "R2" << bundleInfo_->info_.R2_;
        caliInfoOutPut << "P2" << bundleInfo_->info_.P2_;
        caliInfoOutPut << "Q" << bundleInfo_->info_.Q_;
        caliInfoOutPut << "Rlr" << bundleInfo_->info_.Rlr_;
        caliInfoOutPut << "Tlr" << bundleInfo_->info_.Tlr_;
        caliInfoOutPut << "E" << bundleInfo_->info_.E_;
        caliInfoOutPut << "F" << bundleInfo_->info_.F_;
        caliInfoOutPut << "epilines31" << bundleInfo_->info_.epilines31_;
        caliInfoOutPut << "epilines32" << bundleInfo_->info_.epilines32_;
        caliInfoOutPut << "epilines12" << bundleInfo_->info_.epilines12_;
    }

    if(caliType == AppType::CaliType::Projector) {
        caliInfoOutPut << "M4" << bundleInfo_->info_.M4_;
        caliInfoOutPut << "D4" << bundleInfo_->info_.D4_;
        caliInfoOutPut << "Rlp" << bundleInfo_->info_.Rlp_;
        caliInfoOutPut << "Tlp" << bundleInfo_->info_.Tlp_;
        caliInfoOutPut << "K1" << bundleInfo_->info_.K1_;
    }

    caliInfoOutPut.release();
}
