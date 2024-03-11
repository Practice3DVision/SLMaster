#include "chessBoardCalibrator.h"

ChessBoardCalibrator::ChessBoardCalibrator() {}

ChessBoardCalibrator::~ChessBoardCalibrator() {}

bool ChessBoardCalibrator::findFeaturePoints(
    const cv::Mat &img, const cv::Size &featureNums,
    std::vector<cv::Point2f> &points, const ThreshodMethod threshodType) {
    CV_Assert(!img.empty());
    points.clear();

    return cv::findChessboardCornersSB(img, featureNums, points);
}

double ChessBoardCalibrator::calibrate(const std::vector<cv::Mat> &imgs,
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
    double error =
        cv::calibrateCamera(worldPoints_, imgPoints_, imgs[0].size(),
                            intrinsic, distort, rvecs, tvecs);

    for (int i = 0; i < worldPoints_.size(); ++i) {
        std::vector<cv::Point2f> reprojectPoints;
        std::vector<cv::Point2f> curErrorsDistribute;
        cv::projectPoints(worldPoints_[i], rvecs[i], tvecs[i], intrinsic, distort, reprojectPoints);
        for (int j = 0; j < reprojectPoints.size(); ++j) {
            curErrorsDistribute.emplace_back(cv::Point2f(reprojectPoints[j].x - imgPoints_[i][j].x, reprojectPoints[j].y - imgPoints_[i][j].y));
        }
        errors_.emplace_back(curErrorsDistribute);
    }

    return error;
}
