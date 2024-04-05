#include <gtest/gtest.h>

#include <slmaster.h>

using namespace cv;
using namespace std;
using namespace slmaster;
using namespace algorithm;

const string laserLineImgsPath = "../../test/data/laserLine/";

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
        }
    
        Mat img;
        Mat mask;
};

TEST_F(LaserLineSuit, testStegerExtract) {
    vector<Point2f> outPoints;
    stegerExtract(img, outPoints, mask);
    
    Mat colorImg;
    cvtColor(img, colorImg, COLOR_GRAY2BGR);

    FileStorage pointFile("point.yml", FileStorage::WRITE);

    int index = 0;
    for (auto point : outPoints) {
        circle(colorImg, Point(point.x, point.y), 1, Scalar(0, 0, 255));
        Mat pointMat(point);
        pointFile << "Point" + to_string(index++) << pointMat;
    }

    imshow("test", colorImg);
    waitKey(0);
}