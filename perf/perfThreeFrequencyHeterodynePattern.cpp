#include <benchmark/benchmark.h>

#include <slmaster.h>

using namespace slmaster;
using namespace slmaster::algorithm;
using namespace std;
using namespace cv;

const string imgsPath = "../../data/threeFrequencyHeterodyne/";

class ThreeFrequencyHeterodynePatternSuit : public benchmark::Fixture {
  public:
    void SetUp(const benchmark::State &) {
        for (int i = 0; i < 12; ++i) {
            Mat temp = imread(imgsPath + to_string(i) + ".bmp", 0);
            imgs.emplace_back(temp);
        }
    }

    vector<Mat> imgs;
};

BENCHMARK_DEFINE_F(ThreeFrequencyHeterodynePatternSuit, testGenerate)(benchmark::State& state) {
    auto params = ThreeFrequencyHeterodynePattern::Params();
    params.shiftTime = 4;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 64;

    auto pattern = ThreeFrequencyHeterodynePattern::create(params);

    vector<Mat> imgs;

    for (auto _ : state) {
        pattern->generate(imgs);
    }
}

BENCHMARK_DEFINE_F(ThreeFrequencyHeterodynePatternSuit, testGenerateUnwrap)(benchmark::State& state) {
    auto params = ThreeFrequencyHeterodynePattern::Params();
    params.shiftTime = 4;
    params.height = 1080;
    params.width = 1920;
    params.horizontal = false;
    params.nbrOfPeriods = 70;

    auto pattern = ThreeFrequencyHeterodynePattern::create(params);

    vector<Mat> imgs;
    pattern->generate(imgs);

    
    for (auto _ : state) {
        vector<Mat> wrappedPhaseMaps;
        Mat floorMap, confidenceMap, unwrapPhaseMap;
        pattern->computePhaseMap(imgs, wrappedPhaseMaps);
        pattern->computeConfidenceMap(imgs, confidenceMap);
        pattern->computeFloorMap(wrappedPhaseMaps, confidenceMap, floorMap);
        pattern->unwrapPhaseMap(wrappedPhaseMaps[0], floorMap, confidenceMap,
                                unwrapPhaseMap);
    }
}

BENCHMARK_REGISTER_F(ThreeFrequencyHeterodynePatternSuit, testGenerate)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK_REGISTER_F(ThreeFrequencyHeterodynePatternSuit, testGenerateUnwrap)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK_MAIN();