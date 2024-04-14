#include <gtest/gtest.h>

#include <slmaster.h>

using namespace slmaster;
using namespace slmaster::algorithm;
using namespace std;
using namespace cv;

class ShiftGrayCodePatternSuit : public testing::Test {
  public:
    void SetUp() override {}
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

TEST_F(ShiftGrayCodePatternSuit, testUnwrap) {
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
    pattern->unwrapPhaseMap(wrapPhaseMap, floorMap, unwrapPhaseMap);

    ASSERT_EQ(floorMap.ptr<uint16_t>(444)[780], 12);
    ASSERT_EQ(floorMap.ptr<uint16_t>(444)[781], 13);
}