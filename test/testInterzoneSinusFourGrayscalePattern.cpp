#include <gtest/gtest.h>

#include <slmaster.h>

using namespace slmaster;
using namespace slmaster::algorithm;
using namespace std;
using namespace cv;

const string imgsPath = "../../data/interzoneFourGrayscale/";

class InterzoneSinusFourGrayscalePatternSuit : public testing::Test {
  public:
    void SetUp() override {
        for (int i = 0; i < 5; ++i) {
            Mat temp = imread(imgsPath + to_string(i) + ".bmp", 0);
            imgs.emplace_back(temp);
        }
    }

    vector<Mat> imgs;
};

TEST_F(InterzoneSinusFourGrayscalePatternSuit, testGenerate) {
    auto params = InterzoneSinusFourGrayscalePattern::Params();
    params.shiftTime = 3;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 16;

    auto pattern = InterzoneSinusFourGrayscalePattern::create(params);

    vector<Mat> imgs;
    pattern->generate(imgs);

    ASSERT_EQ(imgs[4].ptr<uchar>(340)[660], 170);
    ASSERT_EQ(imgs[4].ptr<uchar>(515)[790], 85);
}

TEST_F(InterzoneSinusFourGrayscalePatternSuit, testGenerateUnwrap) {
    auto params = InterzoneSinusFourGrayscalePattern::Params();
    params.shiftTime = 3;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 16;

    auto pattern = InterzoneSinusFourGrayscalePattern::create(params);

    vector<Mat> imgs;
    pattern->generate(imgs);

    Mat wrappedPhaseMap, floorMap, confidenceMap, unwrapPhaseMap;
    pattern->computePhaseMap(imgs, wrappedPhaseMap);
    pattern->computeConfidenceMap(imgs, confidenceMap);
    pattern->computeFloorMap(imgs, confidenceMap, floorMap);
    pattern->unwrapPhaseMap(wrappedPhaseMap, confidenceMap, floorMap,
                            unwrapPhaseMap);

    ASSERT_EQ(floorMap.ptr<uint16_t>(330)[515], 4);
    ASSERT_LE(abs(unwrapPhaseMap.ptr<float>(410)[685] - 35.866), 0.1f);
}

TEST_F(InterzoneSinusFourGrayscalePatternSuit, testUnwrap) {
    auto params = InterzoneSinusFourGrayscalePattern::Params();
    params.shiftTime = 3;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 16;
    params.confidenceThreshold = 30.f;

    auto pattern = InterzoneSinusFourGrayscalePattern::create(params);

    Mat wrappedPhaseMap, floorMap, confidenceMap, unwrapPhaseMap;
    pattern->computePhaseMap(imgs, wrappedPhaseMap);
    pattern->computeConfidenceMap(imgs, confidenceMap);
    pattern->computeFloorMap(imgs, confidenceMap, floorMap);
    pattern->unwrapPhaseMap(wrappedPhaseMap, confidenceMap, floorMap,
                            unwrapPhaseMap);
    
    ASSERT_EQ(floorMap.ptr<uint16_t>(486)[660], 8);
    ASSERT_LE(abs(unwrapPhaseMap.ptr<float>(384)[656] - 51.8f), 0.1f);
}