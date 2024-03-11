#ifndef CHESSBOARDCALIBRATOR_H
#define CHESSBOARDCALIBRATOR_H

#include "calibrator.h"

class ChessBoardCalibrator : public Calibrator {
  public:
    ChessBoardCalibrator();
    ~ChessBoardCalibrator();
    bool
    findFeaturePoints(const cv::Mat &img, const cv::Size &featureNums,
                      std::vector<cv::Point2f> &points,
                      const ThreshodMethod threshodType = ADAPTED) override;
    double calibrate(const std::vector<cv::Mat> &imgs, cv::Mat &intrinsic,
                     cv::Mat &distort, const cv::Size &featureNums,
                     float &process,
                     const ThreshodMethod threshodType = ADAPTED,
                     const bool blobBlack = true) override;
    inline void setDistance(const float trueDistance) override {
        distance_ = trueDistance;
    };

  private:
    inline void setRadius(const std::vector<float> &radius) override{};

  private:
    float distance_;
};

#endif // CHESSBOARDCALIBRATOR_H
