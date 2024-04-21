#include "circleGridCalibrator.h"

namespace slmaster {
namespace calibration {

CircleGridCalibrator::CircleGridCalibrator() {}

CircleGridCalibrator::~CircleGridCalibrator() {}

bool CircleGridCalibrator::findFeaturePoints(
    const cv::Mat &img, const cv::Size &featureNums,
    std::vector<cv::Point2f> &points, const ThreshodMethod threshodType) {
    CV_Assert(!img.empty());
    points.clear();

    cv::Mat operateImg = img.clone();

    if (img.type() == CV_8UC3) {
        cv::cvtColor(img, operateImg, cv::COLOR_BGR2GRAY);
    }

    cv::SimpleBlobDetector::Params params;
    params.blobColor = 0;
    params.filterByCircularity = true;
    cv::Ptr<cv::SimpleBlobDetector> detector =
        cv::SimpleBlobDetector::create(params);

    bool isFind = cv::findCirclesGrid(operateImg, featureNums, points,
                                      cv::CALIB_CB_SYMMETRIC_GRID, detector);

    if (!isFind) {
        points.clear();
        isFind = cv::findCirclesGrid(
            operateImg, featureNums, points,
            cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING, detector);
    }

    if (!isFind) {
        points.clear();

        params.blobColor = 255;
        detector = cv::SimpleBlobDetector::create(params);

        cv::Mat imgThreshod;
        cv::adaptiveThreshold(operateImg, imgThreshod, 255,
                          cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, 0);

        isFind = cv::findCirclesGrid(imgThreshod, featureNums, points,
                                     cv::CALIB_CB_SYMMETRIC_GRID, detector);
    }

    if (!isFind) {
        points.clear();

        cv::Mat imgThreshod;
        cv::adaptiveThreshold(operateImg, imgThreshod, 255,
                          cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, 0);

        isFind = cv::findCirclesGrid(
            imgThreshod, featureNums, points,
            cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING, detector);
    }

    if (!isFind) {
        return false;
    }

    return isFind;
}

double CircleGridCalibrator::calibrate(const std::vector<cv::Mat> &imgs,
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

} // namespace calibration
} // namespace slmaster