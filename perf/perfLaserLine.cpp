#include <benchmark/benchmark.h>

#include <slmaster.h>

using namespace cv;
using namespace slmaster;
using namespace cameras;
using namespace calibration;
using namespace algorithm;
using namespace std;

const string laserLineImgsPath = "../../data/laserLine/";

class LaserLineSuit : public benchmark::Fixture {
    protected:
        void SetUp(const benchmark::State &) override {
            auto background = imread(laserLineImgsPath + "21.bmp", 0);
            auto front = imread(laserLineImgsPath + "21-laser.bmp", 0);
            mask = front - background;
            threshold(mask, mask, 0, 255, THRESH_OTSU);
            img = front.clone();

            int xBegin = INT_MAX, xEnd = -INT_MAX, yBegin = INT_MAX, yEnd = -INT_MAX;
            for (int i = 0; i < background.rows; ++i) {
                auto ptrBackground = background.ptr<uchar>(i);
                auto ptrFront = front.ptr<uchar>(i);
                auto ptrMask = mask.ptr<uchar>(i);

                for (int j = 0; j < background.cols; ++j) {
                    if(ptrMask[j] != 0) {
                        xBegin = xBegin > j ? j : xBegin;
                        xEnd = xEnd < j ? j : xEnd;
                        yBegin = yBegin > i ? i : yBegin;
                        yEnd = yEnd < i ? i : yEnd;
                    }
                }
            }

            GaussianBlur(img, img, Size(9, 9), 0, 0);

            calibrateImgs.emplace_back(imread(laserLineImgsPath + "21.bmp", 0));
            calibrateImgs.emplace_back(imread(laserLineImgsPath + "21-laser.bmp", 0));
            calibrateImgs.emplace_back(imread(laserLineImgsPath + "22.bmp", 0));
            calibrateImgs.emplace_back(imread(laserLineImgsPath + "22-laser.bmp", 0));
            calibrateImgs.emplace_back(imread(laserLineImgsPath + "23.bmp", 0));
            calibrateImgs.emplace_back(imread(laserLineImgsPath + "23-laser.bmp", 0));
            calibrateImgs.emplace_back(imread(laserLineImgsPath + "24.bmp", 0));
            calibrateImgs.emplace_back(imread(laserLineImgsPath + "24-laser.bmp", 0));
        }
    
        Mat img;
        Mat mask;
        vector<Mat> calibrateImgs;
};

BENCHMARK_DEFINE_F(LaserLineSuit, testStegerExtract)(benchmark::State& state) {
    vector<Point2f> outPoints;

    for (auto _ : state) {
        stegerExtract(img, outPoints, mask);
    }
}

BENCHMARK_DEFINE_F(LaserLineSuit, testCalibration)(benchmark::State& state){
    CaliInfo info(laserLineImgsPath + "caliInfo.yml");

    HiddenMethodCalibrator calibrator;
    Calibrator* chessCalibrator = new ChessBoardCalibrator();
    calibrator.setCalibrator(chessCalibrator);

    for (auto _ : state) {
        calibrator.calibration(calibrateImgs, info.info_.M1_, info.info_.D1_, Size(11, 8), 20.f, info.info_.lightPlaneEq_, false);
    }
}

BENCHMARK_DEFINE_F(LaserLineSuit, testRecoverDepth)(benchmark::State& state){
    CaliInfo info(laserLineImgsPath + "caliInfo.yml");

    vector<Point2f> points2D;
    stegerExtract(img, points2D, mask);
    
    vector<Point3f> points3D;

    for(auto _ : state) {
        recoverDepthFromLightPlane(points2D, info.info_.lightPlaneEq_, info.info_.M1_, points3D);
    }
}

BENCHMARK_REGISTER_F(LaserLineSuit, testStegerExtract)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK_REGISTER_F(LaserLineSuit, testCalibration)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK_REGISTER_F(LaserLineSuit, testRecoverDepth)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK_MAIN();