#include <caliInfo.h>

namespace slmaster {
namespace cameras {
CaliInfo::CaliInfo(const std::string filePath) {
    cv::FileStorage readYml(filePath, cv::FileStorage::READ);
    readYml["M1"] >> info_.M1_;
    readYml["D1"] >> info_.D1_;
    readYml["M2"] >> info_.M2_;
    readYml["D2"] >> info_.D2_;
    readYml["M3"] >> info_.M3_;
    readYml["D3"] >> info_.D3_;
    readYml["M4"] >> info_.M4_;
    readYml["D4"] >> info_.D4_;
    readYml["K1"] >> info_.K1_;
    readYml["R1"] >> info_.R1_;
    readYml["P1"] >> info_.P1_;
    readYml["R2"] >> info_.R2_;
    readYml["P2"] >> info_.P2_;
    readYml["Q"] >> info_.Q_;
    readYml["Rlr"] >> info_.Rlr_;
    readYml["Tlr"] >> info_.Tlr_;
    readYml["Rlp"] >> info_.Rlp_;
    readYml["Tlp"] >> info_.Tlp_;
    readYml["Rlc"] >> info_.Rlc_;
    readYml["Tlc"] >> info_.Tlc_;
    readYml["epilines12"] >> info_.epilines12_;
    readYml["epilines31"] >> info_.epilines31_;
    readYml["epilines32"] >> info_.epilines32_;
    readYml["lightPlaneEq"] >> info_.lightPlaneEq_;
    readYml["S"] >> info_.S_;
    readYml.release();
}
} // namespace cameras
} // namespace slmaster
