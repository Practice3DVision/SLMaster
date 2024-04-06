#include "hiddenMethodCalibrator.h"

#include "../algorithm/cpuStructuredLight/lasterLine.h"

using namespace std;
using namespace cv;

namespace slmaster {

using namespace algorithm;

namespace calibration {
double HiddenMethodCalibrator::calibration(
    const vector<Mat> &imgs, const Mat &intrinsic, const Mat &distort,
    const Size chessSize, const float distance, Mat &lightPlaneEq,
    const bool useCurrentFeaturePoints) {
    CV_Assert_N(imgs.size() % 2 == 0, !intrinsic.empty());
    if (!useCurrentFeaturePoints) {
        // 提取棋盘格点和光条
        chessPoints_.clear();
        laserStrips_.clear();
        chessPoints_.resize(imgs.size() / 2);
        laserStrips_.resize(imgs.size() / 2);

        for (int i = 0; i < imgs.size(); i += 2) {
            // 查找角点
            if (!calibrator_->findFeaturePoints(imgs[i], chessSize,
                                                chessPoints_[i / 2])) {
                return i;
            }
            // 查找棋盘格ROI
            auto rect = minAreaRect(chessPoints_[i / 2]);
            vector<Point2f> vertices;
            rect.points(vertices);
            // 提取激光条纹
            Mat diff = imgs[i + 1] - imgs[i];
            threshold(diff, diff, 0, 255, THRESH_OTSU);

            Mat stripesROI(diff.size(), CV_8UC1, Scalar(0));
            vector<Point2f> pointsStripesROI;
            for (int i = 0; i < diff.rows; ++i) {
                auto ptrStripesROI = stripesROI.ptr<uchar>(i);
                auto ptrDiff = diff.ptr<uchar>(i);
                for (int j = 0; j < diff.cols; ++j) {
                    if (ptrDiff[j] == 0) {
                        continue;
                    }

                    auto dist = pointPolygonTest(vertices, Point2f(j, i), true);
                    if (dist > 0) {
                        ptrStripesROI[j] = 255;
                        pointsStripesROI.emplace_back(Point2f(j, i));
                    }
                }
            }

            // 计算激光条纹宽度
            rect = minAreaRect(pointsStripesROI);
            int laserWidth = rect.size.width > rect.size.height
                                 ? rect.size.height
                                 : rect.size.width;
            if (laserWidth % 2 == 0)
                laserWidth += 1;
            // 高斯滤波
            Mat imgGaussianed;
            GaussianBlur(imgs[i + 1], imgGaussianed,
                         Size(laserWidth, laserWidth), 0, 0);
            // Steger条纹中心提取
            stegerExtract(imgGaussianed, laserStrips_[i / 2], stripesROI);
        }
    }

    // 空间点坐标
    vector<Point3f> worldPoints;
    for (int i = 0; i < chessSize.height; ++i) {
        for (int j = 0; j < chessSize.width; ++j) {
            worldPoints.emplace_back(Point3f(j * distance, i * distance, 0));
        }
    }

    // 光平面方程计算
    Mat intrinsicInv = intrinsic.inv();
    vector<Vec3f> laserLine3D(laserStrips_.size());
    vector<Vec3f> laserLine3DPoints(laserStrips_.size());
    for (int i = 0; i < laserStrips_.size(); ++i) {
        Vec4f tempLine;
        fitLine(laserStrips_[i], tempLine, DIST_HUBER, 0, 0.01, 0.01);

        Vec3f laserLine;
        laserLine[0] = tempLine[1];
        laserLine[1] = -tempLine[0];
        laserLine[2] = tempLine[0] * tempLine[3] - tempLine[1] * tempLine[2];

        Vec3f laserCamPlaneN;
        laserCamPlaneN[0] = laserLine[0] * intrinsic.ptr<double>(0)[0];
        laserCamPlaneN[1] = laserLine[1] * intrinsic.ptr<double>(1)[1],
        laserCamPlaneN[2] = laserLine[0] * intrinsic.ptr<double>(0)[2] +
                            laserLine[1] * intrinsic.ptr<double>(1)[2] +
                            laserLine[2];
        Vec3f laserCamPlanePoint(0.f, 0.f, 0.f);

        Mat r, t;
        solvePnP(worldPoints, chessPoints_[i], intrinsic, distort, r, t);
        Rodrigues(r, r);

        Vec3f targetPlaneN(r.ptr<double>(0)[2], r.ptr<double>(1)[2],
                           r.ptr<double>(2)[2]);
        Vec3f targetPlanePoint(-t.ptr<double>(0)[0], -t.ptr<double>(1)[0],
                               -t.ptr<double>(2)[0]);

        laserLine3D[i] = laserCamPlaneN.cross(targetPlaneN);
        float tempT =
            ((laserCamPlanePoint - targetPlanePoint).dot(targetPlaneN)) /
            (laserCamPlaneN.dot(targetPlaneN));
        laserLine3DPoints[i] = laserCamPlanePoint + tempT * laserLine3D[i];
    }

    Vec3f lightPlaneN(0.f, 0.f, 0.f);
    for (size_t i = 0; i < laserLine3D.size(); ++i) {
        lightPlaneN +=
            laserLine3D[i].cross(laserLine3D[(i + 1) % laserLine3D.size()]);
    }

    lightPlaneN /= norm(lightPlaneN);

    float sumD = 0.f;
    for (const auto &point : laserLine3DPoints) {
        float D = -lightPlaneN.dot(point);
        sumD += D;
    }
    float averageD = sumD / laserLine3DPoints.size();

    lightPlaneEq = Mat(4, 1, CV_64FC1, Scalar(0.));
    lightPlaneEq.ptr<double>(0)[0] = lightPlaneN[0];
    lightPlaneEq.ptr<double>(1)[0] = lightPlaneN[1];
    lightPlaneEq.ptr<double>(2)[0] = lightPlaneN[2];
    lightPlaneEq.ptr<double>(3)[0] = averageD;

    float mod = sqrtf(lightPlaneN[0] * lightPlaneN[0] +
                      lightPlaneN[1] * lightPlaneN[1] +
                      lightPlaneN[2] * lightPlaneN[2]);
    double error = 0.f;
    for (auto point : laserLine3DPoints) {
        error += abs(point[0] * lightPlaneN[0] + point[1] * lightPlaneN[1] +
                     point[2] * lightPlaneN[2] + averageD) /
                 mod;
    }
    error /= laserLine3DPoints.size();

    return error;
}
} // namespace calibration
} // namespace slmaster