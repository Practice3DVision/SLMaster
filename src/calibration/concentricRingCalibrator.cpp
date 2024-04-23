#include "concentricRingCalibrator.h"

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>

namespace slmaster {
namespace calibration {

void recursePath(cv::Mat &img, const int x, const int y, const int color) {
    if (x < 0 || x > img.cols - 1 || y < 0 || y > img.rows - 1 ||
        img.ptr<uchar>(y)[x] != color)
        return;

    if (img.ptr<uchar>(y)[x] == color)
        img.ptr<uchar>(y)[x] = 255 - color;

    recursePath(img, x - 1, y, color);
    recursePath(img, x + 1, y, color);
    recursePath(img, x, y - 1, color);
    recursePath(img, x, y + 1, color);
}

cv::Mat fillHole(const cv::Mat &threshodImg,
                 const std::vector<cv::Point2f> &holeCenter) {
    const int rows = threshodImg.rows;
    const int cols = threshodImg.cols;

    cv::Mat img = threshodImg.clone();

    for (auto pt : holeCenter) {
        const int y = pt.y, x = pt.x;
        const int color = img.ptr<uchar>(pt.y)[x];
        recursePath(img, x, y, color);
    }

    return img;
}

int solveQuadratic(double a, double b, double c, std::vector<double> &roots) {
    double discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
        return 0; // No real roots
    double sqrtD = std::sqrt(discriminant);
    roots.push_back((-b + sqrtD) / (2 * a));
    if (discriminant != 0)
        roots.push_back((-b - sqrtD) / (2 * a));
    return discriminant == 0 ? 1 : 2; // One or two roots
}

std::vector<cv::Point2f> findEllipseLineIntersections(cv::RotatedRect ellipse,
                                                      cv::Vec4f line) {
    std::vector<cv::Point2f> intersections;

    // Decompose the elements of the rotated rectangle (ellipse)
    cv::Point2f center = ellipse.center;
    cv::Size2f size = ellipse.size;
    float angle = ellipse.angle;
    float a = size.width / 2.0;
    float b = size.height / 2.0;

    // Line points
    cv::Point2f pt1(line[0], line[1]);
    cv::Point2f pt2(line[2], line[3]);

    // Calculate the direction vector of the line
    cv::Point2f dir = pt2 - pt1;

    // Angle in radians
    float theta = angle * CV_PI / 180.0;
    float cos_t = std::cos(theta);
    float sin_t = std::sin(theta);

    // Transform line to ellipse coordinate system
    cv::Point2f dir_rot(cos_t * dir.x + sin_t * dir.y,
                        -sin_t * dir.x + cos_t * dir.y);
    cv::Point2f pt1_rot(cos_t * (pt1.x - center.x) + sin_t * (pt1.y - center.y),
                        -sin_t * (pt1.x - center.x) +
                            cos_t * (pt1.y - center.y));

    // Ellipse equation in the form (X/a)^2 + (Y/b)^2 = 1
    // Line equation Y = mX + c in transformed coordinates
    float m = dir_rot.y / dir_rot.x;
    float c = pt1_rot.y - m * pt1_rot.x;

    // Substitute line equation into ellipse equation and solve for X
    double A = (b * b + a * a * m * m);
    double B = 2 * a * a * m * c;
    double C = a * a * (c * c - b * b);

    std::vector<double> roots;
    int numRoots = solveQuadratic(A, B, C, roots);

    for (double x : roots) {
        double y = m * x + c;
        // Transform back to original coordinate system
        double X = cos_t * x - sin_t * y + center.x;
        double Y = sin_t * x + cos_t * y + center.y;
        intersections.push_back(cv::Point2f(X, Y));
    }

    return intersections;
}

bool getIntersection(const cv::Vec4f &line1, const cv::Vec4f &line2,
                     cv::Point2f &intersection) {
    float x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
    float x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];

    float A1 = y2 - y1;
    float B1 = x1 - x2;
    float C1 = A1 * x1 + B1 * y1;

    float A2 = y4 - y3;
    float B2 = x3 - x4;
    float C2 = A2 * x3 + B2 * y3;

    float det = A1 * B2 - A2 * B1;
    if (std::abs(det) < 1e-6) {
        return false;
    }

    intersection.x = (B2 * C1 - B1 * C2) / det;
    intersection.y = (A1 * C2 - A2 * C1) / det;

    return true;
}

cv::Point2f estimateCenter(const cv::Point2f &pt1, const cv::Point2f &pt2,
                           const cv::Point2f &pt3, const float lambda) {
    cv::Point2f center;

    float numerator =
        lambda * pt2.x * (pt3.x - pt1.x) - pt1.x * (pt3.x - pt2.x);
    float denominator = lambda * (pt3.x - pt1.x) - (pt3.x - pt2.x);

    if (std::abs(denominator) < 0.000001f) {
        std::cerr << "Error: Denominator is zero, which may indicate parallel "
                     "lines or a calculation error."
                  << std::endl;
        return cv::Point2f(0.f, 0.f);
    }

    center.x = numerator / denominator;

    numerator = lambda * pt2.y * (pt3.y - pt1.y) - pt1.y * (pt3.y - pt2.y);
    denominator = lambda * (pt3.y - pt1.y) - (pt3.y - pt2.y);

    if (std::abs(denominator) < 0.000001f) {
        std::cerr << "Error: Denominator is zero, which may indicate parallel "
                     "lines or a calculation error."
                  << std::endl;
        return cv::Point2f(0.f, 0.f);
    }

    center.y = numerator / denominator;

    return center;
}

ConcentricRingCalibrator::ConcentricRingCalibrator() {}

ConcentricRingCalibrator::~ConcentricRingCalibrator() {}

bool ConcentricRingCalibrator::findFeaturePoints(
    const cv::Mat &img, const cv::Size &featureNums,
    std::vector<cv::Point2f> &points, const ThreshodMethod threshodType) {
    CV_Assert(!img.empty());
    points.clear();

    return findConcentricRingGrid(img, featureNums, radius_, points);
}

double ConcentricRingCalibrator::calibrate(const std::vector<cv::Mat> &imgs,
                                           cv::Mat &intrinsic, cv::Mat &distort,
                                           const cv::Size &featureNums,
                                           float &process,
                                           const ThreshodMethod threshodType,
                                           const bool blobBlack) {
    imgPoints_.clear();
    worldPoints_.clear();

    std::vector<cv::Point3f> worldPointsCell;
    for (int i = 0; i < featureNums.height; ++i) {
        for (int j = 0; j < featureNums.width; ++j) {
            worldPointsCell.emplace_back(
                cv::Point3f(j * distance_, i * distance_, 0));
        }
    }
    for (int i = 0; i < imgs.size(); ++i)
        worldPoints_.emplace_back(worldPointsCell);

    for (int i = 0; i < imgs.size(); ++i) {
        std::vector<cv::Point2f> imgPointCell;
        if (!findFeaturePoints(imgs[i], featureNums, imgPointCell)) {
            return i;
        } else {
            imgPoints_.emplace_back(imgPointCell);

            cv::Mat imgWithFeature = imgs[i].clone();
            if (imgWithFeature.type() == CV_8UC1) {
                cv::cvtColor(imgWithFeature, imgWithFeature,
                             cv::COLOR_GRAY2BGR);
            }

            cv::drawChessboardCorners(imgWithFeature, featureNums, imgPointCell,
                                      true);
            drawedFeaturesImgs_.push_back(imgWithFeature);

            process = static_cast<float>(i + 1) / imgs.size();
        }
    }

    std::vector<cv::Mat> rvecs, tvecs;
    double error = cv::calibrateCamera(worldPoints_, imgPoints_, imgs[0].size(),
                                       intrinsic, distort, rvecs, tvecs);

    for (int i = 0; i < worldPoints_.size(); ++i) {
        std::vector<cv::Point2f> reprojectPoints;
        std::vector<cv::Point2f> curErrorsDistribute;
        cv::projectPoints(worldPoints_[i], rvecs[i], tvecs[i], intrinsic,
                          distort, reprojectPoints);
        for (int j = 0; j < reprojectPoints.size(); ++j) {
            curErrorsDistribute.emplace_back(
                cv::Point2f(reprojectPoints[j].x - imgPoints_[i][j].x,
                            reprojectPoints[j].y - imgPoints_[i][j].y));
        }
        errors_.emplace_back(curErrorsDistribute);
    }

    return error;
}

void ConcentricRingCalibrator::sortElipse(
    const std::vector<cv::RotatedRect> &rects,
    const std::vector<std::vector<cv::Point2f>> &rectsPoints,
    const std::vector<cv::Point2f> &sortedCenters,
    std::vector<std::vector<cv::RotatedRect>> &sortedRects) {
    sortedRects.clear();

    auto diffNearestPoint = sortedCenters[0] - sortedCenters[1];
    const float distanceFeat =
        std::sqrtf(diffNearestPoint.x * diffNearestPoint.x +
                   diffNearestPoint.y * diffNearestPoint.y) /
        2.f;

    for (size_t i = 0; i < sortedCenters.size(); ++i) {
        std::vector<cv::RotatedRect> rectCurCluster;
        std::vector<std::vector<cv::Point2f>> rectPointsCurCluster;
        for (size_t j = 0; j < rects.size(); ++j) {
            cv::Point2f dist = rects[j].center - sortedCenters[i];
            float distance = std::sqrt(dist.x * dist.x + dist.y * dist.y);

            if (distance < distanceFeat) {
                bool isNeedMerged = false;
                for (size_t d = 0; d < rectCurCluster.size(); ++d) {
                    if (std::abs(rects[j].size.width -
                                 rectCurCluster[d].size.width) < 5.f) {
                        std::vector<cv::Point2f> mergePoints;
                        mergePoints.insert(mergePoints.end(),
                                           rectsPoints[j].begin(),
                                           rectsPoints[j].end());
                        mergePoints.insert(mergePoints.end(),
                                           rectPointsCurCluster[d].begin(),
                                           rectPointsCurCluster[d].end());

                        cv::RotatedRect ellipseFited =
                            cv::fitEllipseDirect(mergePoints);

                        rectCurCluster[d] = ellipseFited;
                        rectPointsCurCluster[d] = mergePoints;

                        isNeedMerged = true;
                        break;
                    }
                }

                if (!isNeedMerged) {
                    rectCurCluster.emplace_back(rects[j]);
                    rectPointsCurCluster.emplace_back(rectsPoints[j]);
                }
            }
        }
        /*
        static int circleIndex = 0;
        static cv::FileStorage writeFile("circle.yml", cv::FileStorage::WRITE);
        for (int s = 0; s < rectPointsCurCluster.size(); ++s) {
            for (int j = 0; j < rectPointsCurCluster[s].size(); ++j) {
                cv::Mat pt =
                    (cv::Mat_<float>(2, 1) << rectPointsCurCluster[s][j].x,
                     rectPointsCurCluster[s][j].y);
                writeFile << "pt_" + std::to_string(circleIndex++) << pt;
            }

            cv::Mat pt = (cv::Mat_<float>(2, 1) << rectCurCluster[s].center.x,
                          rectCurCluster[s].center.y);
            writeFile << "pt_" + std::to_string(circleIndex++) << pt;
        }
        */
        sortedRects.emplace_back(rectCurCluster);
    }
}

bool ConcentricRingCalibrator::findConcentricRingGrid(
    const cv::Mat &inputImg, const cv::Size patternSize,
    const std::vector<float> radius, std::vector<cv::Point2f> &centerPoints) {

    std::vector<cv::Point2f> ellipseCenter;
    cv::Mat threshodFindCircle = inputImg.clone();

    if (inputImg.type() == CV_8UC3) {
        cv::cvtColor(inputImg, threshodFindCircle, cv::COLOR_BGR2GRAY);
    }

    cv::adaptiveThreshold(threshodFindCircle, threshodFindCircle, 255,
                          cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 61, 0);

    cv::SimpleBlobDetector::Params params;
    params.blobColor = 255;
    params.filterByCircularity = true;
    cv::Ptr<cv::SimpleBlobDetector> detector =
        cv::SimpleBlobDetector::create(params);

    std::vector<cv::Point2f> pointsOfCell;
    bool isFind = cv::findCirclesGrid(
        threshodFindCircle, cv::Size(patternSize.width, patternSize.height),
        pointsOfCell, cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING,
        detector);

    if (!isFind) {
        return false;
    }

    auto diffNearestPoint = pointsOfCell[0] - pointsOfCell[1];
    const float distanceFeat =
        std::sqrtf(diffNearestPoint.x * diffNearestPoint.x +
                   diffNearestPoint.y * diffNearestPoint.y) /
        2.f;

    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::RotatedRect> ellipses;
    std::vector<std::vector<cv::Point2f>> ellipsesPoints;
    std::vector<Contour> contours;
    std::vector<std::vector<cv::Point2i>> contours2Draw;

    cv::Mat inputImgClone = inputImg.clone();
    if (inputImg.type() == CV_8UC3) {
        cv::cvtColor(inputImg, inputImgClone, cv::COLOR_BGR2GRAY);
    }

    EdgesSubPix(inputImgClone, 1.5, 20, 40, contours, hierarchy,
                cv::RETR_CCOMP);
    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].points.size() < 20 ||
            contours[i].points.size() > 1000) {
            continue;
        }

        std::vector<cv::Point2f> polyContour;
        cv::approxPolyDP(contours[i].points, polyContour, 0.01, true);
        float area = cv::contourArea(polyContour, false);
        float length = cv::arcLength(polyContour, false);
        float circleRatio = 4.0 * CV_PI * area / (length * length);
        if (circleRatio < 0.6f) {
            continue;
        }

        cv::RotatedRect ellipseFited = cv::fitEllipseDirect(contours[i].points);

        bool isEfficient = false;
        for (size_t j = 0; j < pointsOfCell.size(); ++j) {
            cv::Point dist = pointsOfCell[j] - ellipseFited.center;
            float distance = std::sqrt(dist.x * dist.x + dist.y * dist.y);
            if (distance < distanceFeat) {
                isEfficient = true;
                break;
            }
        }

        if (!isEfficient) {
            continue;
        }

        ellipses.emplace_back(ellipseFited);
        ellipsesPoints.emplace_back(contours[i].points);
    }

    std::vector<std::vector<cv::RotatedRect>> sortedRects;
    std::vector<std::vector<std::vector<cv::Point2f>>> sortedPoints;
    sortElipse(ellipses, ellipsesPoints, pointsOfCell, sortedRects);

    cv::Mat testMat = threshodFindCircle.clone();
    cv::cvtColor(testMat, testMat, cv::COLOR_GRAY2BGR);
    for (int j = 0; j < sortedRects.size(); ++j) {
        for (int k = 0; k < sortedRects[j].size(); ++k) {
            cv::ellipse(testMat, sortedRects[j][k], cv::Scalar(0, 255, 0));
        }
    }

    cv::drawChessboardCorners(testMat,
                              cv::Size(patternSize.width, patternSize.height),
                              pointsOfCell, true);

    if (sortedRects.size() != patternSize.width * patternSize.height) {
        return false;
    }

    std::vector<std::vector<cv::Mat>> circleNormalMat;
    for (int i = 0; i < sortedRects.size(); i++) {
        if (sortedRects[i].size() < 4) {
            return false;
        }

        std::vector<cv::Mat> normalMatsCluster(4);
        getEllipseNormalQuation(sortedRects[i][0], normalMatsCluster[0]);
        getEllipseNormalQuation(sortedRects[i][1], normalMatsCluster[1]);
        getEllipseNormalQuation(sortedRects[i][2], normalMatsCluster[2]);
        getEllipseNormalQuation(sortedRects[i][3], normalMatsCluster[3]);
        circleNormalMat.emplace_back(normalMatsCluster);
    }

    getRingCenters(circleNormalMat, sortedRects, radius, centerPoints);

    return true;
}

void ConcentricRingCalibrator::getRingCenters(
    const std::vector<std::vector<cv::Mat>> &normalMats,
    const std::vector<std::vector<cv::RotatedRect>> &rects,
    const std::vector<float> &radius, std::vector<cv::Point2f> &points) {
    points.clear();

    for (int d = 0; d < normalMats.size(); ++d) {
        std::vector<Eigen::Matrix3f> matchEllipse(4);
        cv::cv2eigen(normalMats[d][0], matchEllipse[0]);
        cv::cv2eigen(normalMats[d][1], matchEllipse[1]);
        cv::cv2eigen(normalMats[d][2], matchEllipse[2]);
        cv::cv2eigen(normalMats[d][3], matchEllipse[3]);

        float minimunEignValueDistance = FLT_MAX;
        float eignValueDistance = FLT_MAX;

        std::vector<cv::Point2f> centerns;
        cv::Vec4f vanishingLinePts;

        for (int i = 3; i > 1; i--) {
            for (int j = i - 1; j > 0; j--) {
                Eigen::Matrix3f polarCircleMat =
                    matchEllipse[i].inverse() * matchEllipse[j];

                Eigen::EigenSolver<Eigen::Matrix3f> eignSolver(polarCircleMat);
                Eigen::Matrix3f value = eignSolver.pseudoEigenvalueMatrix();
                value = value * std::pow(radius[j] / radius[i], 2);
                Eigen::Matrix3f vector = eignSolver.pseudoEigenvectors();
                vector = vector * std::pow(radius[j] / radius[i], 2);

                if (std::abs(value(0, 0) - value(1, 1)) < 0.2f) {
                    eignValueDistance = std::pow(value(0, 0) - 1.f, 2) +
                                        std::pow(value(1, 1) - 1.f, 2);

                    Eigen::Vector3f v1, v2, vanishingLine;
                    v1 << vector(0, 0), vector(1, 0), vector(2, 0);
                    v2 << vector(0, 1), vector(1, 1), vector(2, 1);

                    vanishingLine = v2.cross(v1);

                    for (int k = 0; k < 4; ++k) {
                        Eigen::Vector3f pt =
                            matchEllipse[k].inverse() * vanishingLine;
                        centerns.push_back(
                            cv::Point2f(pt(0) / pt(2), pt(1) / pt(2)));
                    }

                    if (eignValueDistance < minimunEignValueDistance) {
                        minimunEignValueDistance = eignValueDistance;

                        vanishingLinePts[0] = v1(0) / v1(2);
                        vanishingLinePts[1] = v1(1) / v1(2);
                        vanishingLinePts[2] = v2(0) / v2(2);
                        vanishingLinePts[3] = v2(1) / v2(2);
                    }
                } else if (std::abs(value(0, 0) - value(2, 2)) < 0.2f) {
                    eignValueDistance = std::pow(value(0, 0) - 1.f, 2) +
                                        std::pow(value(2, 2) - 1.f, 2);

                    Eigen::Vector3f v1, v2, vanishingLine;
                    v1 << vector(0, 0), vector(1, 0), vector(2, 0);
                    v2 << vector(0, 2), vector(1, 2), vector(2, 2);

                    vanishingLine = v2.cross(v1);

                    for (int k = 0; k < 4; ++k) {
                        Eigen::Vector3f pt =
                            matchEllipse[k].inverse() * vanishingLine;
                        centerns.push_back(
                            cv::Point2f(pt(0) / pt(2), pt(1) / pt(2)));
                    }

                    if (eignValueDistance < minimunEignValueDistance) {
                        minimunEignValueDistance = eignValueDistance;

                        vanishingLinePts[0] = v1(0) / v1(2);
                        vanishingLinePts[1] = v1(1) / v1(2);
                        vanishingLinePts[2] = v2(0) / v2(2);
                        vanishingLinePts[3] = v2(1) / v2(2);
                    }
                } else {
                    eignValueDistance = std::pow(value(1, 1) - 1.f, 2) +
                                        std::pow(value(2, 2) - 1.f, 2);

                    Eigen::Vector3f v1, v2, vanishingLine;
                    v1 << vector(0, 1), vector(1, 1), vector(2, 1);
                    v2 << vector(0, 2), vector(1, 2), vector(2, 2);

                    vanishingLine = v2.cross(v1);

                    for (int k = 0; k < 4; ++k) {
                        Eigen::Vector3f pt =
                            matchEllipse[k].inverse() * vanishingLine;
                        centerns.push_back(
                            cv::Point2f(pt(0) / pt(2), pt(1) / pt(2)));
                    }

                    if (eignValueDistance < minimunEignValueDistance) {
                        minimunEignValueDistance = eignValueDistance;

                        vanishingLinePts[0] = v1(0) / v1(2);
                        vanishingLinePts[1] = v1(1) / v1(2);
                        vanishingLinePts[2] = v2(0) / v2(2);
                        vanishingLinePts[3] = v2(1) / v2(2);
                    }
                }
            }
        }

        cv::Vec4f centerlinePts;
        cv::fitLine(centerns, centerlinePts, cv::DIST_HUBER, 0, 0.01, 0.01);
        centerlinePts[0] = centerlinePts[2] - centerlinePts[0];
        centerlinePts[1] = centerlinePts[3] - centerlinePts[1];

        cv::Point2f infinatePt;
        getIntersection(vanishingLinePts, centerlinePts, infinatePt);

        cv::Point2f finalCenter(0.f, 0.f);
        int validCount = 0;
        for (int k = 0; k < 4; ++k) {
            auto intersectPts =
                findEllipseLineIntersections(rects[d][k], centerlinePts);

            if (intersectPts.size() != 2)
                continue;

            cv::Point2f realCenter = estimateCenter(
                intersectPts[0], intersectPts[1], infinatePt, -1);

            finalCenter += realCenter;
            validCount++;
        }

        finalCenter /= validCount;

        points.push_back(finalCenter);
    }
}

void ConcentricRingCalibrator::getEllipseNormalQuation(
    const cv::RotatedRect &rotateRect, cv::Mat &quationMat) {

    float angle = rotateRect.angle * CV_PI / 180.0;
    float a = rotateRect.size.width / 2.f;
    float b = rotateRect.size.height / 2.f;
    float xc = rotateRect.center.x;
    float yc = rotateRect.center.y;

    float cosAngle = cos(angle);
    float sinAngle = sin(angle);
    float A = cosAngle * cosAngle / (a * a) + sinAngle * sinAngle / (b * b);
    float B = 2 * cosAngle * sinAngle * (1 / (a * a) - 1 / (b * b));
    float C = sinAngle * sinAngle / (a * a) + cosAngle * cosAngle / (b * b);
    float D = -2 * A * xc - B * yc;
    float E = -2 * C * yc - B * xc;
    float F = A * xc * xc + B * xc * yc + C * yc * yc - 1;

    cv::Mat normalMat(3, 3, CV_32F, cv::Scalar(0.f));
    normalMat.at<float>(0, 0) = A;
    normalMat.at<float>(0, 1) = B / 2.f;
    normalMat.at<float>(0, 2) = D / 2.f;
    normalMat.at<float>(1, 0) = normalMat.at<float>(0, 1);
    normalMat.at<float>(1, 1) = C;
    normalMat.at<float>(1, 2) = E / 2.f;
    normalMat.at<float>(2, 0) = normalMat.at<float>(0, 2);
    normalMat.at<float>(2, 1) = normalMat.at<float>(1, 2);
    normalMat.at<float>(2, 2) = F;

    normalMat.copyTo(quationMat);
}

bool ConcentricRingCalibrator::sortKeyPoints(
    const std::vector<cv::Point2f> &inputPoints, const cv::Size patternSize,
    std::vector<cv::Point2f> &outputPoints) {
    cv::RotatedRect rect = cv::minAreaRect(inputPoints);
    cv::Point2f rectTopPoints[4];
    rect.points(rectTopPoints);
    cv::Point2f leftUpper;
    cv::Point2f rightUpper;
    std::vector<cv::Point2f> comparePoint;
    std::vector<int> indexPoint;
    int index_LeftUp;
    std::vector<float> x_Rect = {rectTopPoints[0].x, rectTopPoints[1].x,
                                 rectTopPoints[2].x, rectTopPoints[3].x};
    std::sort(x_Rect.begin(), x_Rect.end());
    for (int i = 0; i < 4; i++) {
        if (rectTopPoints[i].x == x_Rect[0] ||
            rectTopPoints[i].x == x_Rect[1]) {
            comparePoint.push_back(rectTopPoints[i]);
            indexPoint.push_back(i);
        }
    }
    if (comparePoint[0].y < comparePoint[1].y) {
        leftUpper = comparePoint[0];
        index_LeftUp = indexPoint[0];
    } else {
        leftUpper = comparePoint[1];
        index_LeftUp = indexPoint[1];
    }
    if (index_LeftUp == 0) {
        float value_first = std::pow((leftUpper.x - rectTopPoints[1].x), 2) +
                            std::pow((leftUpper.y - rectTopPoints[1].y), 2);
        float value_second = std::pow((leftUpper.x - rectTopPoints[3].x), 2) +
                             std::pow((leftUpper.y - rectTopPoints[3].y), 2);
        if (value_first < value_second) {
            rightUpper = rectTopPoints[3];
        } else {
            rightUpper = rectTopPoints[1];
        }
    } else {
        float value_first =
            std::pow((leftUpper.x - rectTopPoints[(index_LeftUp + 1) % 4].x),
                     2) +
            std::pow((leftUpper.y - rectTopPoints[(index_LeftUp + 1) % 4].y),
                     2);
        float value_second =
            std::pow((leftUpper.x - rectTopPoints[index_LeftUp - 1].x), 2) +
            std::pow((leftUpper.y - rectTopPoints[3].y), 2);
        if (value_first < value_second) {
            rightUpper = rectTopPoints[index_LeftUp - 1];
        } else {
            rightUpper = rectTopPoints[(index_LeftUp + 1) % 4];
        }
    }
    std::vector<float> distance;
    std::vector<float> distance_Sort;
    std::vector<cv::Point2f> sortDistancePoints;
    for (int i = 0; i < inputPoints.size(); i++) {
        float distanceValue =
            getDist_P2L(inputPoints[i], leftUpper, rightUpper);
        distance.push_back(distanceValue);
        distance_Sort.push_back(distanceValue);
    }
    std::sort(distance_Sort.begin(), distance_Sort.end());
    for (int i = 0; i < inputPoints.size(); i++) {
        for (int j = 0; j < inputPoints.size(); j++) {
            if (distance[j] == distance_Sort[i]) {
                sortDistancePoints.push_back(inputPoints[j]);
                break;
            }
        }
    }
    std::vector<float> valueX;
    int match;
    int rows = patternSize.height;
    int cols = patternSize.width;
    for (int i = 0; i < rows; i++) {
        valueX.clear();
        for (int j = 0; j < cols; j++) {
            valueX.push_back(sortDistancePoints[i * cols + j].x);
        }
        std::sort(valueX.begin(), valueX.end());
        for (int j = 0; j < valueX.size(); j++) {
            for (int k = 0; k < cols; k++) {
                if (valueX[j] == sortDistancePoints[i * cols + k].x) {
                    match = i * cols + k;
                    break;
                }
            }
            outputPoints.push_back(sortDistancePoints[match]);
        }
    }

    return true;
}

float ConcentricRingCalibrator::getDist_P2L(cv::Point2f pointP,
                                            cv::Point2f pointA,
                                            cv::Point2f pointB) {
    float A = 0, B = 0, C = 0;
    A = pointA.y - pointB.y;
    B = pointB.x - pointA.x;
    C = pointA.x * pointB.y - pointA.y * pointB.x;

    float distance = 0;
    distance = ((float)abs(A * pointP.x + B * pointP.y + C)) /
               ((float)sqrtf(A * A + B * B));

    return distance;
}

} // namespace calibration
} // namespace slmaster