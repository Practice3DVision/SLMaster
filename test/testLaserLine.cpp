#include <gtest/gtest.h>

#include <slmaster.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

using namespace cv;
using namespace std;
using namespace slmaster;
using namespace algorithm;
using namespace calibration;
using namespace cameras;

const string laserLineImgsPath = "../../data/laserLine/";

class LaserLineSuit : public testing::Test {
    public:
        void SetUp() override {
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

TEST_F(LaserLineSuit, testStegerExtract) {
    vector<Point2f> outPoints;
    stegerExtract(img, outPoints, mask);
    
    Mat colorImg;
    cvtColor(img, colorImg, COLOR_GRAY2BGR);

    FileStorage pointFile("point.yml", FileStorage::WRITE);

    int index = 0;
    for (auto point : outPoints) {
        circle(colorImg, Point2i(point.x, point.y), 1, Scalar(0, 0, 255));
        Mat pointMat(point);
        pointFile << "Point" + to_string(index++) << pointMat;
    }

    imshow("test", colorImg);
    waitKey(0);
}

TEST_F(LaserLineSuit, testCalibration) {
    CaliInfo info(laserLineImgsPath + "caliInfo.yml");

    HiddenMethodCalibrator calibrator;
    Calibrator* chessCalibrator = new ChessBoardCalibrator();
    calibrator.setCalibrator(chessCalibrator);
    auto error = calibrator.calibration(calibrateImgs, info.info_.M1_, info.info_.D1_, Size(11, 8), 20.f, info.info_.lightPlaneEq_, false);
    std::cout << error << std::endl;

    CaliPacker packer(&info);
    packer.writeCaliInfo(3, laserLineImgsPath + "caliInfo.yml");
}

TEST_F(LaserLineSuit, testRecoverDepth) {
    CaliInfo info(laserLineImgsPath + "caliInfo.yml");

    vector<Point2f> points2D;
    stegerExtract(img, points2D, mask);
    
    vector<Point3f> points3D;
    recoverDepthFromLightPlane(points2D, info.info_.lightPlaneEq_, info.info_.M1_, points3D);

    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (auto point : points3D) {
        cloud.points.emplace_back(pcl::PointXYZ(point.x, point.y, point.z));
    }

    pcl::io::savePLYFile("line.ply", cloud);
}

TEST_F(LaserLineSuit, rotationAxisCalibration) {

}