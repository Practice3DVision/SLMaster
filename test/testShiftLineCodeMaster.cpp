#include <gtest/gtest.h>

#include <matrixsInfo.h>
#include <rectifier.h>
#include <rectifierCpu.h>
#include <nShiftLineNGrayCodeMasterCpu.h>
#include <restructorShiftLineCpu.h>

const std::string imgPath = "../../test/data/shiftline1";
const std::string intrinsicPath = "../../test/data/shiftline1/intrinsic.yml";
const std::string extrinsicPath = "../../test/data/shiftline1/extrinsic.yml";

TEST(ShiftLinePhaseSolver, solve) {
    std::vector<cv::Mat> leftImgs;
    for (size_t i = 0; i < 17; ++i) {
        cv::Mat img = cv::imread(imgPath + "/L/" + std::to_string(i) + ".bmp", 0);
        leftImgs.emplace_back(img);
    }

    auto solver = new sl::phaseSolver::NShiftLineNGrayCodeMasterCpu;

    sl::phaseSolver::PhaseSolverGroupDataHost groupData;
    solver->solve(leftImgs, groupData, 0.5, 9);
} 

TEST(RestructorShiftLineCpu, restruction) {
    sl::tool::MatrixsInfo info(intrinsicPath, extrinsicPath);
    sl::rectifier::RectifierCpu rectifier(info.getInfo());

    std::vector<cv::Mat> leftImgs, rightImgs;
    for (size_t i = 0; i < 17; ++i) {
        cv::Mat leftImg = cv::imread(imgPath + "/L/" + std::to_string(i) + ".bmp", 0);
        cv::Mat rightImg = cv::imread(imgPath + "/R/" + std::to_string(i) + ".bmp", 0);
        rectifier.remapImg(leftImg, leftImg);
        rectifier.remapImg(rightImg, rightImg, false);
        leftImgs.emplace_back(leftImg);
        rightImgs.emplace_back(rightImg);
    }

    auto solver = new sl::phaseSolver::NShiftLineNGrayCodeMasterCpu;

    sl::phaseSolver::PhaseSolverGroupDataHost leftGroupData, rightGroupData;
    solver->solve(leftImgs, leftGroupData, 0.5, 9);
    solver->solve(rightImgs, rightGroupData, 0.5, 9);

    sl::restructor::RestructParamater params;
    params.__minDisparity = -500;
    params.__maxDisparity = 500;
    params.__minDepth = 0;
    params.__maxDepth = 2000;
    params.__isMapToColorCamera = false;
    params.__isMapToPreDepthAxes = false;
    
    sl::restructor::RestructorShiftLineCpu restructor(info.getInfo());
    cv::Mat depthImg;
    restructor.restruction(leftGroupData.__unwrapMap, rightGroupData.__unwrapMap, params, depthImg);
}