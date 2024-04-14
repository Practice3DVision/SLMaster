#include <benchmark/benchmark.h>

#include <slmaster.h>

using namespace slmaster;
using namespace slmaster::algorithm;
using namespace std;
using namespace cv;

class ShiftGrayCodePatternSuit : public benchmark::Fixture {
  public:
    void SetUp(const benchmark::State &) override {}
};

BENCHMARK_DEFINE_F(ShiftGrayCodePatternSuit, testGenerate)
(benchmark::State &state) {
    auto params = SinusShiftGrayCodePattern::Params();
    params.shiftTime = 4;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 32;

    auto pattern = SinusShiftGrayCodePattern::create(params);

    vector<Mat> imgs;

    for (auto _ : state) {
        pattern->generate(imgs);
    }
}

BENCHMARK_DEFINE_F(ShiftGrayCodePatternSuit, testUnwrap)
(benchmark::State &state) {
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

    for (auto _ : state) {
        pattern->computePhaseMap(imgs, wrapPhaseMap);
        pattern->computeConfidenceMap(imgs, confidenceMap);
        pattern->computeFloorMap(imgs, confidenceMap, wrapPhaseMap, floorMap);
        pattern->unwrapPhaseMap(wrapPhaseMap, floorMap, unwrapPhaseMap);
    }
}

BENCHMARK_REGISTER_F(ShiftGrayCodePatternSuit, testGenerate)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK_REGISTER_F(ShiftGrayCodePatternSuit, testUnwrap)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK_MAIN();