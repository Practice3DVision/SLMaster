#include <benchmark/benchmark.h>

#include <slmaster.h>

using namespace slmaster;
using namespace slmaster::algorithm;
using namespace std;
using namespace cv;

class InterzoneSinusFourGrayscalePatternSuit : public benchmark::Fixture {
  public:
    void SetUp(const benchmark::State &) override {}
};

BENCHMARK_DEFINE_F(InterzoneSinusFourGrayscalePatternSuit, testGenerate)
(benchmark::State &state) {
    auto params = InterzoneSinusFourGrayscalePattern::Params();
    params.shiftTime = 3;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 16;

    auto pattern = InterzoneSinusFourGrayscalePattern::create(params);

    vector<Mat> imgs;

    for (auto _ : state) {
        pattern->generate(imgs);
    }
}

BENCHMARK_DEFINE_F(InterzoneSinusFourGrayscalePatternSuit, testUnwrap)
(benchmark::State &state) {
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

    for (auto _ : state) {
        pattern->computePhaseMap(imgs, wrappedPhaseMap);
        pattern->computeConfidenceMap(imgs, confidenceMap);
        pattern->computeFloorMap(imgs, confidenceMap, floorMap);
        pattern->unwrapPhaseMap(wrappedPhaseMap, confidenceMap, floorMap,
                                unwrapPhaseMap);
    }
}

BENCHMARK_REGISTER_F(InterzoneSinusFourGrayscalePatternSuit, testGenerate)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK_REGISTER_F(InterzoneSinusFourGrayscalePatternSuit, testUnwrap)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK_MAIN();