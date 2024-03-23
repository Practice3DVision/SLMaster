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

const string STRUCTURED_LIGHT_DIR = "../gpu/structured_light/multi_view_stereo_geometry";
const string CALIBRATION_PATH = STRUCTURED_LIGHT_DIR + "/caliInfo.yml";
const string LEFT_CAM_FOLDER_DATA = STRUCTURED_LIGHT_DIR + "/L/";
const string RIGHT_CAM_FOLDER_DATA = STRUCTURED_LIGHT_DIR + "/R/";
const string COLOR_CAM_FOLDER_DATA = STRUCTURED_LIGHT_DIR + "/C/";
const string REF_UNWRAP_IMG_DATA = STRUCTURED_LIGHT_DIR + "/refUnwrappedMap.tiff";
const string TRUTH_UNWRAPPED_IMG_DATA = STRUCTURED_LIGHT_DIR + "/VPhase.tiff";

class MultiViewStereoGeometryPatternFixSuit : public testing::Test {
  public:
    void SetUp() override {
        cv::FileStorage caliParamsReader(cvtest::TS::ptr()->get_data_path() + "/" + CALIBRATION_PATH, cv::FileStorage::READ);
        cv::Mat M1, M2, M3, M4, D1, D2, D3, Rlr, Tlr , Rlc, Tlc, Rlp, Tlp;
        caliParamsReader["M1"] >> M1;
        caliParamsReader["M2"] >> M2;
        caliParamsReader["M3"] >> M3;
        caliParamsReader["M4"] >> M4;
        caliParamsReader["D1"] >> D1;
        caliParamsReader["D2"] >> D2;
        caliParamsReader["D3"] >> D3;
        caliParamsReader["Rlr"] >> Rlr;
        caliParamsReader["Tlr"] >> Tlr;
        caliParamsReader["Rlc"] >> Rlc;
        caliParamsReader["Tlc"] >> Tlc;
        caliParamsReader["Rlp"] >> Rlp;
        caliParamsReader["Tlp"] >> Tlp;

        M1.convertTo(M1, CV_32FC1);
        M2.convertTo(M2, CV_32FC1);
        M3.convertTo(M3, CV_32FC1);
        M4.convertTo(M4, CV_32FC1);
        D1.convertTo(D1, CV_32FC1);
        D2.convertTo(D2, CV_32FC1);
        D3.convertTo(D3, CV_32FC1);
        Rlr.convertTo(Rlr, CV_32FC1);
        Tlr.convertTo(Tlr, CV_32FC1);
        Rlc.convertTo(Rlc, CV_32FC1);
        Tlc.convertTo(Tlc, CV_32FC1);
        Rlp.convertTo(Rlp, CV_32FC1);
        Tlp.convertTo(Tlp, CV_32FC1);

        cv::Mat refUnwrapImg = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" + REF_UNWRAP_IMG_DATA, cv::IMREAD_UNCHANGED);
        cv::Mat PL1 = cv::Mat::eye(4, 4, CV_32FC1);
        M1.copyTo(PL1(cv::Rect(0, 0, 3, 3)));
        cv::Mat PR2 = cv::Mat::eye(4, 4, CV_32FC1);
        Rlr.copyTo(PR2(cv::Rect(0, 0, 3, 3)));
        Tlr.copyTo(PR2(cv::Rect(3, 0, 1, 3)));
        cv::Mat M2Normal = cv::Mat::eye(4, 4, CV_32FC1);
        M2.copyTo(M2Normal(cv::Rect(0, 0, 3, 3)));
        PR2 = M2Normal * PR2;
        cv::Mat PR4 = cv::Mat::eye(4, 4, CV_32FC1);
        Rlp.copyTo(PR4(cv::Rect(0, 0, 3, 3)));
        Tlp.copyTo(PR4(cv::Rect(3, 0, 1, 3)));
        cv::Mat M4Normal = cv::Mat::eye(4, 4, CV_32FC1);
        M4.copyTo(M4Normal(cv::Rect(0, 0, 3, 3)));
        PR4 = M4Normal * PR4;

        params.height = 1024;
        params.width = 1920;
        params.horizontal = false;
        params.nbrOfPeriods = 32;
        params.shiftTime = 3;
        params.confidenceThreshold = 10.f;
        params.minDisparity = 0;
        params.maxDisparity = 256;
        params.minDepth = -FLT_MAX;
        params.maxDepth = FLT_MAX;
        params.maxCost = 0.5f;
        params.costMinDiff = 0.0001f;
        params.costMaxDiff = 1.f;
        cv::cv2eigen(M1, params.M1);
        cv::cv2eigen(M2, params.M2);
        cv::cv2eigen(M3, params.M3);
        cv::cv2eigen(Rlr, params.R12);
        cv::cv2eigen(Tlr, params.T12);
        cv::cv2eigen(Rlc, params.R13);
        cv::cv2eigen(Tlc, params.T13);
        cv::cv2eigen(PL1, params.PL1);
        cv::cv2eigen(PR2, params.PR2);
        cv::cv2eigen(PR4, params.PR4);
        params.refUnwrappedMap.upload(refUnwrapImg);

        pattern = cv::cuda::MultiViewStereoGeometryPattern::create(params);

        imgs.resize(3, std::vector<cv::Mat>(3));
        for (size_t i = 0; i < 3; ++i) {
            imgs[0][i] = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" + LEFT_CAM_FOLDER_DATA + std::to_string(i) + ".bmp", 0);
            imgs[1][i] = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" + RIGHT_CAM_FOLDER_DATA + std::to_string(i) + ".bmp", 0);
            imgs[2][i] = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" + COLOR_CAM_FOLDER_DATA + std::to_string(i) + ".bmp", 0);
        }
    }

    void TearDown() override {}

    cv::cuda::MultiViewStereoGeometryPattern::Params params;
    cv::Ptr<cv::cuda::MultiViewStereoGeometryPattern> pattern;
    std::vector<std::vector<cv::Mat>> imgs; //color left right
};

TEST_F(MultiViewStereoGeometryPatternFixSuit, test_generate) {
    std::vector<cv::Mat> stripImgs;
    pattern->generate(stripImgs);

    EXPECT_LE(abs(stripImgs[0].ptr(399)[738] - 166), 1);
}

TEST_F(MultiViewStereoGeometryPatternFixSuit, test_calcWrappedMap) {
    std::vector<cv::Mat> imgs;
    pattern->generate(imgs);

    cv::Mat mergeImgHost;
    cv::merge(imgs, mergeImgHost);

    cv::cuda::GpuMat mergeImgDev(mergeImgHost), wrappedMapDev, confidenceMapDev;
    pattern->computeWrappedAndConfidenceMap(mergeImgDev, wrappedMapDev, confidenceMapDev);

    cv::Mat wrappedMapHost, confidenceMapHost;
    wrappedMapDev.download(wrappedMapHost);
    confidenceMapDev.download(confidenceMapHost);

    EXPECT_LE(abs(wrappedMapHost.ptr<float>(344)[266] - (-0.4211)), 0.01f);
    EXPECT_LE(abs(confidenceMapHost.ptr<float>(344)[266] - 127), 1.f);
}

TEST_F(MultiViewStereoGeometryPatternFixSuit, test_unwrappedMapWithRefMap) {
    cv::Mat mergePhaseImg;
    cv::merge(imgs[0], mergePhaseImg);
    cv::cuda::GpuMat mergePhaseImgDev(mergePhaseImg);
    cv::cuda::GpuMat wrappedPhaseMapDev, confidenceMapDev, unwrappedMapDev;
    pattern->computeWrappedAndConfidenceMap(mergePhaseImgDev, wrappedPhaseMapDev, confidenceMapDev);
    pattern->unwrapPhaseMap(params.refUnwrappedMap, wrappedPhaseMapDev, confidenceMapDev, unwrappedMapDev);

    cv::Mat truthUnwrappedMap = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" + TRUTH_UNWRAPPED_IMG_DATA, cv::IMREAD_UNCHANGED);

    cv::Mat unwrappedMap, wrapMap, confidenceMap;
    unwrappedMapDev.download(unwrappedMap);
    wrappedPhaseMapDev.download(wrapMap);
    confidenceMapDev.download(confidenceMap);

    cv::Mat foreground = cv::Mat(unwrappedMap.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    for (int i = 0; i < unwrappedMap.rows; ++i) {
        auto ptrFore = foreground.ptr<cv::Vec3b>(i);
        auto ptrTruth = truthUnwrappedMap.ptr<float>(i);
        auto ptrUnwrapMap = unwrappedMap.ptr<float>(i);
        for (int j = 0; j < unwrappedMap.cols; ++j) {
            if (abs(ptrUnwrapMap[j] - ptrTruth[j]) > 0.5f && abs(ptrTruth[j]) > 0.001f && ptrUnwrapMap[j] > 0.001f && confidenceMap.ptr<float>(i)[j] > 20.f) {
                ptrFore[j] = cv::Vec3b(0, 0, 255);
            }
        }
    }

    EXPECT_LE(std::abs(unwrappedMap.ptr<float>(443)[735] - truthUnwrappedMap.ptr<float>(443)[735]), 0.1f);
}

TEST_F(MultiViewStereoGeometryPatternFixSuit, test_polynomialFitting) {
    cv::Mat mergePhaseImg;
    cv::merge(imgs[0], mergePhaseImg);
    cv::cuda::GpuMat mergePhaseImgDev(mergePhaseImg);
    cv::cuda::GpuMat wrappedPhaseMapDev, confidenceMapDev, unwrappedMapDev;
    pattern->computeWrappedAndConfidenceMap(mergePhaseImgDev, wrappedPhaseMapDev, confidenceMapDev);
    pattern->unwrapPhaseMap(params.refUnwrappedMap, wrappedPhaseMapDev, confidenceMapDev, unwrappedMapDev);
    cv::cuda::GpuMat coarseDepthMapDev;
    pattern->polynomialFitting(unwrappedMapDev, coarseDepthMapDev);
    cv::Mat coarseDepthMap;
    coarseDepthMapDev.download(coarseDepthMap);

    EXPECT_LE(abs(coarseDepthMap.ptr<float>(443)[735] - 618.73), 1.f);
}

TEST_F(MultiViewStereoGeometryPatternFixSuit, test_multiViewStereoRefineDepth) {
    cv::Mat mergePhaseImgCam1, mergePhaseImgCam2, mergePhaseImgCam3;
    cv::merge(imgs[0], mergePhaseImgCam1);
    cv::merge(imgs[1], mergePhaseImgCam2);
    cv::merge(imgs[2], mergePhaseImgCam3);
    cv::cuda::GpuMat mergePhaseImgDevCam1(mergePhaseImgCam1), mergePhaseImgDevCam2(mergePhaseImgCam2), mergePhaseImgDevCam3(mergePhaseImgCam3);
    cv::cuda::GpuMat wrappedPhaseMapDevCam1, confidenceMapDevCam1, unwrappedMapDevCam1, wrappedPhaseMapDevCam2, confidenceMapDevCam2, wrappedPhaseMapDevCam3, confidenceMapDevCam3;
    pattern->computeWrappedAndConfidenceMap(mergePhaseImgDevCam1, wrappedPhaseMapDevCam1, confidenceMapDevCam1);
    pattern->unwrapPhaseMap(params.refUnwrappedMap, wrappedPhaseMapDevCam1, confidenceMapDevCam1, unwrappedMapDevCam1);
    cv::cuda::GpuMat coarseDepthMapDev, refineDepthMapDev;
    pattern->polynomialFitting(unwrappedMapDevCam1, coarseDepthMapDev);
    pattern->computeWrappedAndConfidenceMap(mergePhaseImgDevCam2, wrappedPhaseMapDevCam2, confidenceMapDevCam2);
    pattern->computeWrappedAndConfidenceMap(mergePhaseImgDevCam3, wrappedPhaseMapDevCam3, confidenceMapDevCam3);
    pattern->multiViewStereoRefineDepth(coarseDepthMapDev, wrappedPhaseMapDevCam1, confidenceMapDevCam1, wrappedPhaseMapDevCam2, confidenceMapDevCam2, wrappedPhaseMapDevCam3, confidenceMapDevCam3, refineDepthMapDev);

    cv::Mat hostDepth;
    refineDepthMapDev.download(hostDepth);
    EXPECT_LE(abs(hostDepth.ptr<float>(470)[680] - 630), 1.f);
}

TEST_F(MultiViewStereoGeometryPatternFixSuit, test_decode) {
    cv::Mat depthMap;
    pattern->decode(imgs, depthMap, cuda::MULTI_VIEW_STEREO_GEOMETRY);
    EXPECT_LE(abs(depthMap.ptr<float>(470)[680] - 630), 1.f);
}

}
}
