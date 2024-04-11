#include <gtest/gtest.h>

#include <slmaster.h>

using namespace slmaster;
using namespace slmaster::algorithm;
using namespace std;
using namespace cv;

class ShiftGrayCodePatternSuit : public testing::Test {
    public:
        void SetUp() override {

        }
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
}