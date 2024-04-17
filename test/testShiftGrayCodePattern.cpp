#include <gtest/gtest.h>

#include <slmaster.h>

using namespace slmaster;
using namespace slmaster::algorithm;
using namespace std;
using namespace cv;

const string imgsPath = "../../data/shiftGraycode/";

class ShiftGrayCodePatternSuit : public testing::Test {
  public:
    void SetUp() override {
        for (int i = 0; i < 9; ++i) {
            Mat temp = imread(imgsPath + to_string(i) + ".bmp", 0);
            imgs.emplace_back(temp);
        }
    }

    vector<Mat> imgs;
};

TEST_F(ShiftGrayCodePatternSuit, testGenerate) {
    auto params = SinusShiftGrayCodePattern::Params();
    params.shiftTime = 4;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 32;

    auto pattern = SinusShiftGrayCodePattern::create(params);

    vector<Mat> imgs;
    pattern->generate(imgs);

    ASSERT_EQ(imgs[6].ptr<uchar>(400)[215], 255);
}

TEST_F(ShiftGrayCodePatternSuit, testGenerateUnwrap) {
    auto params = SinusShiftGrayCodePattern::Params();
    params.shiftTime = 4;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 32;

    auto pattern = SinusShiftGrayCodePattern::create(params);

    vector<Mat> imgs;
    pattern->generate(imgs);

    Mat wrapPhaseMap, floorMap, confidenceMap, unwrapPhaseMap;
    pattern->computePhaseMap(imgs, wrapPhaseMap);
    pattern->computeConfidenceMap(imgs, confidenceMap);
    pattern->computeFloorMap(imgs, confidenceMap, wrapPhaseMap, floorMap);
    pattern->unwrapPhaseMap(wrapPhaseMap, floorMap, confidenceMap, unwrapPhaseMap);

    ASSERT_EQ(floorMap.ptr<uint16_t>(444)[780], 12);
    ASSERT_EQ(floorMap.ptr<uint16_t>(444)[781], 13);
}

TEST_F(ShiftGrayCodePatternSuit, testUnwrap) {
    auto params = SinusShiftGrayCodePattern::Params();
    params.shiftTime = 4;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 32;
    params.confidenceThreshold = 70.f;

    auto pattern = SinusShiftGrayCodePattern::create(params);

    Mat wrapPhaseMap, floorMap, confidenceMap, unwrapPhaseMap;
    pattern->computePhaseMap(imgs, wrapPhaseMap);
    pattern->computeConfidenceMap(imgs, confidenceMap);
    pattern->computeFloorMap(imgs, confidenceMap, wrapPhaseMap, floorMap);
    pattern->unwrapPhaseMap(wrapPhaseMap, floorMap, confidenceMap, unwrapPhaseMap);

    ASSERT_EQ(floorMap.ptr<uint16_t>(453)[700], 17);
    ASSERT_LE(abs(unwrapPhaseMap.ptr<float>(460)[653] - 103.75), 0.1f);
}