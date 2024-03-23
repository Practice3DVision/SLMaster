/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this
 license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without
 modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright
 notice,
 //     this list of conditions and the following disclaimer in the
 documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote
 products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is"
 and
 // any express or implied warranties, including, but not limited to, the
 implied
 // warranties of merchantability and fitness for a particular purpose are
 disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any
 direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include <string>

#include "test_precomp.hpp"

#include <cuda_runtime.h>

namespace opencv_test {
namespace {

using namespace std;

const string STRUCTURED_LIGHT_DIR = "structured_light";
const string CALIBRATION_PATH =
    "data/sinus_complementary_graycode/test_decode/cali.yml";
const string LEFT_CAM_FOLDER_DATA =
    "data/sinus_complementary_graycode/test_decode/left";
const string RIGHT_CAM_FOLDER_DATA =
    "data/sinus_complementary_graycode/test_decode/right";
const string TEST_GENERATE_FOLDER_DATA =
    "data/sinus_complementary_graycode/test_generate";
const string TEST_UNWRAP_FOLDER_DATA =
    "data/sinus_complementary_graycode/test_unwrap";

class SinusoidalComplementaryGrayCodeFixSuit : public testing::Test {
  public:
    void SetUp() override {
        params.height = 720;
        params.width = 1280;
        params.horizontal = false;
        params.nbrOfPeriods = 32;
        params.shiftTime = 4;
        params.confidenceThreshold = 10.f;
        params.minDisparity = 0;
        params.maxDisparity = 256;
        params.maxCost = 0.1f;
        params.costMinDiff = 0.0001f;

        pattern = cv::cuda::SinusCompleGrayCodePattern::create(params);
    }

    void TearDown() override {}

    cv::cuda::SinusCompleGrayCodePattern::Params params;
    cv::Ptr<cv::cuda::SinusCompleGrayCodePattern> pattern;
};

TEST_F(SinusoidalComplementaryGrayCodeFixSuit, calc_wrappedMap) {
    std::vector<cv::Mat> imgs;
    for (int i = 0; i < 4; ++i) {
        cv::Mat tempImg = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" +
                                         STRUCTURED_LIGHT_DIR + "/" +
                                         TEST_GENERATE_FOLDER_DATA + "/im" +
                                         std::to_string(i) + ".bmp",
                                     0);
        imgs.push_back(tempImg);
    }

    cv::Mat mergeImg;
    cv::merge(imgs, mergeImg);
    cv::cuda::GpuMat mergeImgDev(mergeImg);

    cv::cuda::GpuMat wrappedPhaseMap, confidenceMap;
    pattern->computeWrappedAndConfidenceMap(mergeImgDev, wrappedPhaseMap, confidenceMap);

    cv::Mat wrappedPhaseMapHost, confidenceMapHost;
    wrappedPhaseMap.download(wrappedPhaseMapHost);
    confidenceMap.download(confidenceMapHost);

    EXPECT_LE(std::abs(wrappedPhaseMapHost.ptr<float>(380)[245] + 2.3562), 0.01f);
    EXPECT_LE(std::abs(confidenceMapHost.ptr<float>(259)[233] - 127), 0.6f);
}

TEST_F(SinusoidalComplementaryGrayCodeFixSuit, unwrapMap) {
    std::vector<cv::Mat> phaseImgs, grayImgs;
    for (int i = 0; i < 10; ++i) {
        cv::Mat tempImg = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" +
                                         STRUCTURED_LIGHT_DIR + "/" +
                                         TEST_GENERATE_FOLDER_DATA + "/im" +
                                         std::to_string(i) + ".bmp",
                                     0);
        i < 4 ? phaseImgs.push_back(tempImg) : grayImgs.push_back(tempImg);
    }

    cv::Mat mergePhaseImgs, mergeGrayImgs;
    cv::merge(phaseImgs, mergePhaseImgs);
    cv::merge(grayImgs, mergeGrayImgs);
    cv::cuda::GpuMat mergePhaseImgsDev(mergePhaseImgs);
    cv::cuda::GpuMat mergeGrayImgsDev(mergeGrayImgs);

    cv::cuda::GpuMat wrappedPhaseMap, confidenceMap, unwrappedPhaseMap;
    pattern->computeWrappedAndConfidenceMap(mergePhaseImgsDev, wrappedPhaseMap, confidenceMap);
    pattern->unwrapPhaseMap(mergeGrayImgsDev, wrappedPhaseMap, confidenceMap, unwrappedPhaseMap);

    cv::Mat unwrappedPhaseMapHost;
    unwrappedPhaseMap.download(unwrappedPhaseMapHost);

    EXPECT_LE(std::abs(unwrappedPhaseMapHost.ptr<float>(240)[550] - 86.394), 0.01f);
}

TEST_F(SinusoidalComplementaryGrayCodeFixSuit, computeDisparity) {
    cv::FileStorage caliReader(cvtest::TS::ptr()->get_data_path() + "/" +
                                   STRUCTURED_LIGHT_DIR + "/" +
                                   CALIBRATION_PATH,
                               cv::FileStorage::READ);

    cv::Mat M1, M2, D1, D2, R, T, R1, R2, P1, P2, Q, S;
    caliReader["M1"] >> M1;
    caliReader["M2"] >> M2;
    caliReader["D1"] >> D1;
    caliReader["D2"] >> D2;
    caliReader["R"] >> R;
    caliReader["T"] >> T;
    caliReader["R1"] >> R1;
    caliReader["R2"] >> R2;
    caliReader["P1"] >> P1;
    caliReader["P2"] >> P2;
    caliReader["Q"] >> Q;
    caliReader["S"] >> S;

    cv::Size imgSize(S.at<int>(0, 0), S.at<int>(0, 1));
    cv::Mat mapLX, mapLY, mapRX, mapRY;
    cv::initUndistortRectifyMap(M1, D1, R1, P1, imgSize, CV_32FC1, mapLX,
                                mapLY);
    cv::initUndistortRectifyMap(M2, D2, R2, P2, imgSize, CV_32FC1, mapRX,
                                mapRY);

    std::vector<cv::Mat> leftPhaseImgs, leftGrayImgs, rightPhaseImgs, rightGrayImgs;
    for (int i = 1; i < 20; ++i) {
        cv::Mat tempImg = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" +
                                         STRUCTURED_LIGHT_DIR + "/" +
                                         LEFT_CAM_FOLDER_DATA + "/im" +
                                         std::to_string(i) + ".bmp",
                                     0);
        //tempImg.convertTo(tempImg, CV_32FC1);
        cv::remap(tempImg, tempImg, mapLX, mapLY, cv::INTER_LINEAR);
        i < 13 ? leftPhaseImgs.push_back(tempImg) : leftGrayImgs.push_back(tempImg);

        tempImg = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" +
                                         STRUCTURED_LIGHT_DIR + "/" +
                                         RIGHT_CAM_FOLDER_DATA + "/im" +
                                         std::to_string(i) + ".bmp",
                                     0);
        //tempImg.convertTo(tempImg, CV_32FC1);
        cv::remap(tempImg, tempImg, mapRX, mapRY, cv::INTER_LINEAR);
        i < 13 ? rightPhaseImgs.push_back(tempImg) : rightGrayImgs.push_back(tempImg);
    }

    cv::Mat leftMergePhaseImgs, leftMergeGrayImgs, rightMergePhaseImgs, rightMergeGrayImgs;

    cv::merge(leftPhaseImgs, leftMergePhaseImgs);
    cv::merge(leftGrayImgs, leftMergeGrayImgs);
    cv::merge(rightPhaseImgs, rightMergePhaseImgs);
    cv::merge(rightGrayImgs, rightMergeGrayImgs);

    cv::cuda::GpuMat leftMergePhaseImgsDev(leftMergePhaseImgs);
    cv::cuda::GpuMat leftMergeGrayImgsDev(leftMergeGrayImgs);
    cv::cuda::GpuMat rightMergePhaseImgsDev(rightMergePhaseImgs);
    cv::cuda::GpuMat rightMergeGrayImgsDev(rightMergeGrayImgs);

    cv::cuda::GpuMat leftWrappedPhaseMap, leftConfidenceMap, leftUnwrappedPhaseMap;
    cv::cuda::GpuMat rightWrappedPhaseMap, rightConfidenceMap, rightUnwrappedPhaseMap;
    pattern->computeWrappedAndConfidenceMap(leftMergePhaseImgsDev, leftWrappedPhaseMap, leftConfidenceMap);
    pattern->unwrapPhaseMap(leftMergeGrayImgsDev, leftWrappedPhaseMap, leftConfidenceMap, leftUnwrappedPhaseMap, params.confidenceThreshold);
    pattern->computeWrappedAndConfidenceMap(rightMergePhaseImgsDev, rightWrappedPhaseMap, rightConfidenceMap);
    pattern->unwrapPhaseMap(rightMergeGrayImgsDev, rightWrappedPhaseMap, rightConfidenceMap, rightUnwrappedPhaseMap, params.confidenceThreshold);

    cv::cuda::GpuMat disparityMapDev;
    pattern->computeDisparity(leftUnwrappedPhaseMap, rightUnwrappedPhaseMap, disparityMapDev);

    cv::Mat disparityMapHost;
    disparityMapDev.download(disparityMapHost);

    EXPECT_LE(std::abs(disparityMapHost.ptr<float>(530)[865] - 237.f), 1.f);
}

TEST_F(SinusoidalComplementaryGrayCodeFixSuit, decode) {
    cv::FileStorage caliReader(cvtest::TS::ptr()->get_data_path() + "/" +
                                   STRUCTURED_LIGHT_DIR + "/" +
                                   CALIBRATION_PATH,
                               cv::FileStorage::READ);

    cv::Mat M1, M2, D1, D2, R, T, R1, R2, P1, P2, Q, S;
    caliReader["M1"] >> M1;
    caliReader["M2"] >> M2;
    caliReader["D1"] >> D1;
    caliReader["D2"] >> D2;
    caliReader["R"] >> R;
    caliReader["T"] >> T;
    caliReader["R1"] >> R1;
    caliReader["R2"] >> R2;
    caliReader["P1"] >> P1;
    caliReader["P2"] >> P2;
    caliReader["Q"] >> Q;
    caliReader["S"] >> S;

    cv::Size imgSize(S.at<int>(0, 0), S.at<int>(0, 1));
    cv::Mat mapLX, mapLY, mapRX, mapRY;
    cv::initUndistortRectifyMap(M1, D1, R1, P1, imgSize, CV_32FC1, mapLX,
                                mapLY);
    cv::initUndistortRectifyMap(M2, D2, R2, P2, imgSize, CV_32FC1, mapRX,
                                mapRY);

    std::vector<std::vector<cv::Mat>> imgs(2);
    for (int i = 1; i < 20; ++i) {
        cv::Mat tempImg = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" +
                                         STRUCTURED_LIGHT_DIR + "/" +
                                         LEFT_CAM_FOLDER_DATA + "/im" +
                                         std::to_string(i) + ".bmp",
                                     0);
        //tempImg.convertTo(tempImg, CV_32FC1);
        cv::remap(tempImg, tempImg, mapLX, mapLY, cv::INTER_LINEAR);
        imgs[0].push_back(tempImg);

        tempImg = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" +
                                         STRUCTURED_LIGHT_DIR + "/" +
                                         RIGHT_CAM_FOLDER_DATA + "/im" +
                                         std::to_string(i) + ".bmp",
                                     0);
        //tempImg.convertTo(tempImg, CV_32FC1);
        cv::remap(tempImg, tempImg, mapRX, mapRY, cv::INTER_LINEAR);
        imgs[1].push_back(tempImg);
    }

    cv::cuda::SinusCompleGrayCodePattern::Params curParams;
    curParams.height = 1080;
    curParams.width = 1920;
    curParams.horizontal = false;
    curParams.nbrOfPeriods = 64;
    curParams.shiftTime = 12;
    curParams.confidenceThreshold = 10.f;
    curParams.minDisparity = -128;
    curParams.maxDisparity = 384;
    curParams.maxCost = 0.1f;
    curParams.costMinDiff = 0.0001f;

    auto curPattern = cv::cuda::SinusCompleGrayCodePattern::create(curParams);

    cv::Mat disparityMapHost;
    curPattern->decode(imgs, disparityMapHost);

    EXPECT_LE(std::abs(disparityMapHost.ptr<float>(530)[865] - 237.f), 1.f);
}

} // namespace
} // namespace opencv_test
