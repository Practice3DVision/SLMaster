#include <nStepNGrayCodeMasterCpu.h>
#include <rectifierCpu.h>
#include <matrixsInfo.h>
#include <wrapCreatorCpu.h>
#include <restructorCpu.h>
#include <tool.h>

#include <gtest/gtest.h>

TEST(SLCamera, rectifier) {
    std::vector<cv::Mat> imgs;
    cv::Mat img = cv::imread("../../test/data/phaseGray/L/" + std::to_string(0) + ".bmp", 0);
    imgs.emplace_back(img);

    sl::tool::MatrixsInfo* matrixInfo = new sl::tool::MatrixsInfo("../../test/data/cali/intrinsic.yml", "../../test/data/cali/extrinsic.yml");
    sl::rectifier::Rectifier* rectifier = new sl::rectifier::RectifierCpu(matrixInfo->getInfo());
    cv::Mat preRectify = imgs[0].clone();
    rectifier->remapImg(imgs[0], imgs[0]);
}

TEST(SLCamera, wrapCreator) {
    sl::tool::MatrixsInfo* matrixInfo = new sl::tool::MatrixsInfo("../../test/data/cali/intrinsic.yml", "../../test/data/cali/extrinsic.yml");
    sl::rectifier::Rectifier* rectifier = new sl::rectifier::RectifierCpu(matrixInfo->getInfo());

    std::vector<cv::Mat> imgs;
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat img = cv::imread("../../test/data/phaseGray/L/" + std::to_string(i) + ".bmp", 0);
        rectifier->remapImg(img, img);
        imgs.emplace_back(img);
    }

    sl::wrapCreator::WrapCreator* wrapCreator = new sl::wrapCreator::WrapCreatorCpu();

    cv::Mat wrapImg, conditionImg;
    wrapCreator->getWrapImg(std::vector<cv::Mat>(imgs.begin(), imgs.begin() + 4), wrapImg, conditionImg);
}

TEST(SLCamera, phaseSolver) {
    sl::tool::MatrixsInfo* matrixInfo = new sl::tool::MatrixsInfo("../../test/data/cali/intrinsic.yml", "../../test/data/cali/extrinsic.yml");
    sl::rectifier::Rectifier* rectifier = new sl::rectifier::RectifierCpu(matrixInfo->getInfo());

    std::vector<cv::Mat> imgs;
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat img = cv::imread("../../test/data/phaseGray/L/" + std::to_string(i) + ".bmp", 0);
        img.convertTo(img, CV_32FC1);
        imgs.emplace_back(img);
        rectifier->remapImg(imgs[i], imgs[i]);
    }

    sl::phaseSolver::PhaseSolver* phaseSolver = new sl::phaseSolver::NStepNGrayCodeMasterCpu();

    sl::phaseSolver::PhaseSolverGroupDataHost groupData;
    phaseSolver->solve(imgs, groupData, 4);
}

TEST(SLCamera, filterDepth) {
    sl::tool::MatrixsInfo* matrixInfo = new sl::tool::MatrixsInfo("../../test/data/cali/intrinsic.yml", "../../test/data/cali/extrinsic.yml");
    sl::rectifier::Rectifier* rectifier = new sl::rectifier::RectifierCpu(matrixInfo->getInfo());

    std::vector<cv::Mat> imgs;
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat img = cv::imread("../../test/data/phaseGray/L/" + std::to_string(i) + ".bmp", 0);
        img.convertTo(img, CV_32FC1);
        imgs.emplace_back(img);
        rectifier->remapImg(imgs[i], imgs[i]);
    }

    sl::phaseSolver::PhaseSolver* phaseSolver = new sl::phaseSolver::NStepNGrayCodeMasterCpu();

    sl::phaseSolver::PhaseSolverGroupDataHost groupData;
    phaseSolver->solve(imgs, groupData, 4);

    cv::Mat outMap;
    sl::tool::filterPhase(groupData.__unwrapMap, outMap, 0.5, 5);
}

TEST(SLCamera, restructor) {
    sl::tool::MatrixsInfo* matrixInfo = new sl::tool::MatrixsInfo("../../test/data/cali/intrinsic.yml", "../../test/data/cali/extrinsic.yml");
    sl::rectifier::Rectifier* rectifier = new sl::rectifier::RectifierCpu(matrixInfo->getInfo());

    std::vector<cv::Mat> imgsL, imgsR;
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat imgL = cv::imread("../../test/data/phaseGray/L/" + std::to_string(i) + ".bmp", 0);
        cv::Mat imgR = cv::imread("../../test/data/phaseGray/R/" + std::to_string(i) + ".bmp", 0);
        rectifier->remapImg(imgL, imgL, true);
        rectifier->remapImg(imgR, imgR, false);
        imgsL.emplace_back(imgL);
        imgsR.emplace_back(imgR);
    }

    sl::phaseSolver::PhaseSolver* phaseSolver = new sl::phaseSolver::NStepNGrayCodeMasterCpu();

    sl::phaseSolver::PhaseSolverGroupDataHost groupDataL, groupDataR;
    phaseSolver->solve(imgsL, groupDataL, 0, 4);
    phaseSolver->solve(imgsR, groupDataR, 0, 4);

    sl::restructor::Restructor* restructor = new sl::restructor::RestructorCpu(matrixInfo->getInfo());
    cv::Mat depthImg;
    sl::restructor::RestructParamater param(-200, 200, 0, 2000);
    param.__isMapToPreDepthAxes = true;
    param.__isMapToColorCamera = false;
    param.__maximumCost = 0.1;
    restructor->restruction(groupDataL.__unwrapMap, groupDataR.__unwrapMap, param, depthImg);
}

TEST(SLCamera, restructorApplyFilter) {
    sl::tool::MatrixsInfo* matrixInfo = new sl::tool::MatrixsInfo("../../test/data/cali/intrinsic.yml", "../../test/data/cali/extrinsic.yml");
    sl::rectifier::Rectifier* rectifier = new sl::rectifier::RectifierCpu(matrixInfo->getInfo());

    std::vector<cv::Mat> imgsL, imgsR;
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat imgL = cv::imread("../../test/data/phaseGray/L/" + std::to_string(i) + ".bmp", 0);
        cv::Mat imgR = cv::imread("../../test/data/phaseGray/R/" + std::to_string(i) + ".bmp", 0);
        rectifier->remapImg(imgL, imgL, true);
        rectifier->remapImg(imgR, imgR, false);
        imgsL.emplace_back(imgL);
        imgsR.emplace_back(imgR);
    }

    sl::phaseSolver::PhaseSolver* phaseSolver = new sl::phaseSolver::NStepNGrayCodeMasterCpu();

    sl::phaseSolver::PhaseSolverGroupDataHost groupDataL, groupDataR;
    phaseSolver->solve(imgsL, groupDataL, 0, 4);
    phaseSolver->solve(imgsR, groupDataR, 0, 4);

    cv::Mat outMapL, outMapR;
    sl::tool::filterPhase(groupDataL.__unwrapMap, outMapL, 0.5f, 7);
    sl::tool::filterPhase(groupDataR.__unwrapMap, outMapR, 0.5f, 7);

    sl::restructor::Restructor* restructor = new sl::restructor::RestructorCpu(matrixInfo->getInfo());
    cv::Mat depthImg;
    sl::restructor::RestructParamater param(-200, 200, 0, 2000);
    param.__isMapToPreDepthAxes = true;
    param.__isMapToColorCamera = false;
    param.__maximumCost = 0.1;
    restructor->restruction(outMapL, outMapR, param, depthImg);
}

TEST(SLCamera, remapDepth) {
    sl::tool::MatrixsInfo* matrixInfo = new sl::tool::MatrixsInfo("../../test/data/cali/intrinsic.yml", "../../test/data/cali/extrinsic.yml");
    sl::rectifier::Rectifier* rectifier = new sl::rectifier::RectifierCpu(matrixInfo->getInfo());

    std::vector<cv::Mat> imgsL, imgsR;
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat imgL = cv::imread("../../test/data/phaseGray/L/" + std::to_string(i) + ".bmp", 0);
        cv::Mat imgR = cv::imread("../../test/data/phaseGray/R/" + std::to_string(i) + ".bmp", 0);
        rectifier->remapImg(imgL, imgL, true);
        rectifier->remapImg(imgR, imgR, false);
        imgsL.emplace_back(imgL);
        imgsR.emplace_back(imgR);
    }

    sl::phaseSolver::PhaseSolver* phaseSolver = new sl::phaseSolver::NStepNGrayCodeMasterCpu();

    sl::phaseSolver::PhaseSolverGroupDataHost groupDataL, groupDataR;
    phaseSolver->solve(imgsL, groupDataL, 50, 4);
    phaseSolver->solve(imgsR, groupDataR, 50, 4);

    sl::restructor::Restructor* restructor = new sl::restructor::RestructorCpu(matrixInfo->getInfo());
    cv::Mat depthImg;
    sl::restructor::RestructParamater param(-200, 200, 0, 2000);
    param.__isMapToPreDepthAxes = true;
    param.__isMapToColorCamera = false;
    param.__maximumCost = 0.1;
    restructor->restruction(groupDataL.__unwrapMap, groupDataR.__unwrapMap, param, depthImg);
}