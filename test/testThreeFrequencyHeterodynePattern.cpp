#include <gtest/gtest.h>

#include <slmaster.h>

using namespace slmaster;
using namespace slmaster::algorithm;
using namespace std;
using namespace cv;

const string imgsPath = "../../data/threeFrequencyHeterodyne/";

class ThreeFrequencyHeterodynePatternSuit : public testing::Test {
  public:
    void SetUp() override {
        /*
        for (int i = 0; i < 5; ++i) {
            Mat temp = imread(imgsPath + to_string(i) + ".bmp", 0);
            imgs.emplace_back(temp);
        }
        */
    }

    vector<Mat> imgs;
};

TEST_F(ThreeFrequencyHeterodynePatternSuit, testGenerate) {
    auto params = ThreeFrequencyHeterodynePattern::Params();
    params.shiftTime = 4;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 64;

    auto pattern = ThreeFrequencyHeterodynePattern::create(params);

    vector<Mat> imgs;
    pattern->generate(imgs);

    ASSERT_EQ(imgs[0].ptr<uchar>(420)[730], 191);
    ASSERT_EQ(imgs[3].ptr<uchar>(420)[730], 17);
    ASSERT_EQ(imgs[6].ptr<uchar>(420)[730], 176);
}

TEST_F(ThreeFrequencyHeterodynePatternSuit, testGenerateUnwrap) {
    auto params = ThreeFrequencyHeterodynePattern::Params();
    params.shiftTime = 4;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 70;

    auto pattern = ThreeFrequencyHeterodynePattern::create(params);

    vector<Mat> imgs;
    pattern->generate(imgs);

    vector<Mat> wrappedPhaseMaps;
    Mat floorMap, confidenceMap, unwrapPhaseMap;
    pattern->computePhaseMap(imgs, wrappedPhaseMaps);
    pattern->computeConfidenceMap(imgs, confidenceMap);
    pattern->computeFloorMap(wrappedPhaseMaps, confidenceMap, floorMap);
    pattern->unwrapPhaseMap(wrappedPhaseMaps, confidenceMap, floorMap,
                            unwrapPhaseMap);

    ASSERT_EQ(floorMap.ptr<uint16_t>(420)[730], 191);
    ASSERT_LE(abs(unwrapPhaseMap.ptr<float>(410)[685] - 35.866), 0.1f);
}

TEST_F(ThreeFrequencyHeterodynePatternSuit, testUnwrap) {
    auto params = ThreeFrequencyHeterodynePattern::Params();
    params.shiftTime = 4;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 64;
    params.confidenceThreshold = 30.f;

    auto pattern = ThreeFrequencyHeterodynePattern::create(params);

    vector<Mat> wrappedPhaseMaps;
    Mat floorMap, confidenceMap, unwrapPhaseMap;
    pattern->computePhaseMap(imgs, wrappedPhaseMaps);
    pattern->computeConfidenceMap(imgs, confidenceMap);
    pattern->computeFloorMap(imgs, confidenceMap, floorMap);
    pattern->unwrapPhaseMap(wrappedPhaseMaps, confidenceMap, floorMap,
                            unwrapPhaseMap);
    
    ASSERT_EQ(floorMap.ptr<uint16_t>(486)[660], 8);
    ASSERT_LE(abs(unwrapPhaseMap.ptr<float>(384)[656] - 51.8f), 0.1f);
}