#include "rotationAxisCalibration.h"

using namespace std;
using namespace cv;

namespace slmaster {
namespace calibration {
double RotationAxisCalibration::calibration(
    const vector<cv::Mat> &imgs, const Mat &intrinsic, const Mat &distort,
    const Size chessSize, const float distance, Mat &rotatorAxis,
    const bool useCurrentFeaturePoints) {
    vector<Point3f> worldPoints;
    for (int i = 0; i < chessSize.height; ++i) {
        for (int j = 0; j < chessSize.width; ++j) {
            worldPoints.emplace_back(Point3f(j * distance, i * distance, 0));
        }
    }
    // 查找特征点
    if (!useCurrentFeaturePoints) {
        chessPoints_.clear();
        chessPoints_.resize(imgs.size());

        for (int i = 0; i < imgs.size(); ++i) {
            if (!calibrator_->findFeaturePoints(imgs[i], chessSize,
                                                chessPoints_[i])) {
                return i;
            }
        }
    }
    // 计算特征点在相机坐标系下的坐标
    vector<vector<Point3f>> points3D(chessPoints_[0].size());
    for (int i = 0; i < imgs.size(); ++i) {
        Mat rvec, tvec;
        solvePnP(worldPoints, chessPoints_[i], intrinsic, distort, rvec, tvec);
        Rodrigues(rvec, rvec);

        for (int j = 0; j < worldPoints.size(); ++j) {
            Mat pt = (Mat_<double>(3, 1) << worldPoints[j].x, worldPoints[j].y,
                      worldPoints[j].z);
            Mat camPoint = rvec * pt + tvec;
            Vec3f camPointVec(camPoint.ptr<double>(0)[0],
                              camPoint.ptr<double>(1)[0],
                              camPoint.ptr<double>(2)[0]);
            points3D[j].emplace_back(camPointVec);
        }
    }
    // 根据特征点逐个拟合圆心点
    vector<Vec3f> circleCenterPoints(points3D.size());
    for (int i = 0; i < points3D.size(); ++i) {
        // 计算平面方程
        Mat x;
        {
            Mat A;

            for (int j = 0; j < points3D[i].size(); ++j) {
                Mat data = (Mat_<float>(1, 4) << points3D[i][j].x,
                            points3D[i][j].y, points3D[i][j].z, 1);
                A.push_back(data);
            }

            SVD svd(A);
            x = svd.vt.row(svd.vt.rows - 1).t();
        }
        // 计算圆心点
        Mat A, B;
        for (int j = 0; j < points3D[i].size(); ++j) {
            auto point2 = points3D[i][(j + 1) % points3D[i].size()];
            auto point1 = points3D[i][j];
            Vec3f diffPoint = point2 - point1;

            Mat dataA =
                (Mat_<float>(1, 3) << diffPoint[0], diffPoint[1], diffPoint[2]);
            Mat dataB = (Mat_<float>(1, 1)
                         << (point2.x * point2.x - point1.x * point1.x +
                             point2.y * point2.y - point1.y * point1.y +
                             point2.z * point2.z - point1.z * point1.z) /
                                2.f);

            A.push_back(dataA);
            B.push_back(dataB);
        }
        // 约束条件，圆心在平面上
        Mat n = (Mat_<float>(1, 3) << x.ptr<float>(0)[0], x.ptr<float>(1)[0],
                 x.ptr<float>(2)[0]);
        Mat d = (Mat_<float>(1, 1) << x.ptr<float>(3)[0]);
        A.push_back(n);
        B.push_back(d);
        // 求解圆心
        Mat circleCenter;
        solve(A, B, circleCenter, DECOMP_SVD);

        circleCenterPoints.emplace_back(Point3f(circleCenter.ptr<float>(0)[0],
                                                circleCenter.ptr<float>(1)[0],
                                                circleCenter.ptr<float>(2)[0]));
    }

    Vec6f lineParams;
    fitLine(circleCenterPoints, lineParams, DIST_HUBER, 0, 0.01, 0.01);

    rotatorAxis = (Mat_<double>(6, 1) << lineParams[0], lineParams[1],
                   lineParams[2], lineParams[3], lineParams[4], lineParams[5]);

    double error = 0.f;
    Vec3f axisDirection(lineParams[0], lineParams[1], lineParams[2]);
    Vec3f axisCenter(lineParams[3], lineParams[4], lineParams[5]);
    for (auto point : circleCenterPoints) {
        Vec3f pq = point - axisCenter;
        Vec3f n = pq.cross(axisDirection);
        double pqLength = norm(pq);
        double distance = norm(n) / norm(axisDirection);
        error += distance;
    }
    error /= circleCenterPoints.size();

    return error;
}
} // namespace calibration
} // namespace slmaster