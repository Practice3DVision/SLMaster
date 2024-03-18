#include <nStepNGrayCodeMasterGpu.h>
#include <rectifierGpu.h>
#include <rectifierCpu.h>
#include <matrixsInfo.h>
#include <wrapCreatorGpu.h>
#include <phaseSolver.h>
#include <restructor.h>
#include <restructorGpu.h>
#include <tool.h>

#include <gtest/gtest.h>

TEST(SLCamera, rectifier) {
    std::vector<cv::Mat> imgs;
    cv::Mat img = cv::imread("../../test/data/phaseGray/L/" + std::to_string(0) + ".bmp", 0);
    imgs.emplace_back(img);

    sl::tool::MatrixsInfo* matrixInfo = new sl::tool::MatrixsInfo("../../test/data/cali/intrinsic.yml", "../../test/data/cali/extrinsic.yml");
    sl::rectifier::Rectifier* rectifier = new sl::rectifier::RectifierGpu(matrixInfo->getInfo());
    cv::Mat preRectify = imgs[0].clone();
    cv::cuda::GpuMat imgRected;
    cv::Mat result;
    rectifier->remapImg(imgs[0], imgRected);
    imgRected.download(result);
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

    sl::wrapCreator::WrapCreator* wrapCreator = new sl::wrapCreator::WrapCreatorGpu();

    cv::cuda::GpuMat wrapImgDevice, conditionImgDevice;
    cv::Mat wrapImg, conditionImg;
    wrapCreator->getWrapImg(std::vector<cv::Mat>(imgs.begin(), imgs.begin() + 4), wrapImgDevice, conditionImgDevice);
    wrapImgDevice.download(wrapImg);
    conditionImgDevice.download(conditionImg);
}

TEST(SLCamera, phaseSolver) {
    sl::tool::MatrixsInfo* matrixInfo = new sl::tool::MatrixsInfo("../../test/data/cali/intrinsic.yml", "../../test/data/cali/extrinsic.yml");
    sl::rectifier::Rectifier* rectifier = new sl::rectifier::RectifierCpu(matrixInfo->getInfo());

    std::vector<cv::Mat> imgs;
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat img = cv::imread("../../test/data/phaseGray/L/" + std::to_string(i) + ".bmp", 0);
        rectifier->remapImg(img, img);
        imgs.emplace_back(img);
    }

    sl::phaseSolver::PhaseSolver* phaseSolver = new sl::phaseSolver::NStepNGrayCodeMasterGpu();

    sl::phaseSolver::PhaseSolverGroupDataDevice groupData;
    phaseSolver->solve(imgs, groupData, 20, 4, cv::cuda::Stream::Null(), dim3(32, 8));
    sl::phaseSolver::PhaseSolverGroupDataHost groupDataHost;
    groupData.__wrapMap.download(groupDataHost.__wrapMap);
    groupData.__textureMap.download(groupDataHost.__textureMap);
    groupData.__unwrapMap.download(groupDataHost.__unwrapMap);
}

TEST(SLCamera, phaseFilter) {
    sl::tool::MatrixsInfo* matrixInfo = new sl::tool::MatrixsInfo("../../test/data/cali/intrinsic.yml", "../../test/data/cali/extrinsic.yml");
    sl::rectifier::Rectifier* rectifier = new sl::rectifier::RectifierCpu(matrixInfo->getInfo());

    std::vector<cv::Mat> imgs;
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat img = cv::imread("../../test/data/phaseGray/L/" + std::to_string(i) + ".bmp", 0);
        rectifier->remapImg(img, img);
        imgs.emplace_back(img);
    }

    sl::phaseSolver::PhaseSolver* phaseSolver = new sl::phaseSolver::NStepNGrayCodeMasterGpu();

    sl::phaseSolver::PhaseSolverGroupDataDevice groupData;
    phaseSolver->solve(imgs, groupData, 20, 4, cv::cuda::Stream::Null(), dim3(32, 8));
    cv::cuda::GpuMat phaseFiltered;
    sl::tool::cudaFunc::filterPhase(groupData.__unwrapMap, phaseFiltered, 0.5, 5);
    cv::Mat imgPre, imgAft;
    groupData.__unwrapMap.download(imgPre);
    phaseFiltered.download(imgAft);
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

    sl::phaseSolver::PhaseSolver* phaseSolver = new sl::phaseSolver::NStepNGrayCodeMasterGpu();

    sl::phaseSolver::PhaseSolverGroupDataDevice groupDataL, groupDataR;
    phaseSolver->solve(imgsL, groupDataL, 20, 4, cv::cuda::Stream::Null(), dim3(32, 8));
    phaseSolver->solve(imgsR, groupDataR, 20, 4, cv::cuda::Stream::Null(), dim3(32, 8));

    sl::restructor::RestructorGpu* restructor = new sl::restructor::RestructorGpu(matrixInfo->getInfo());
    sl::restructor::RestructParamater param(-200, 200, 0, 2000);
    param.__isMapToPreDepthAxes = true;
    param.__isMapToColorCamera = false;
    param.__maximumCost = 0.1;
    cv::Mat depthImg;
    cv::cuda::GpuMat  depthImgDevice;
    restructor->restruction(groupDataL.__unwrapMap, groupDataR.__unwrapMap, param, depthImgDevice);
    depthImgDevice.download(depthImg);
}