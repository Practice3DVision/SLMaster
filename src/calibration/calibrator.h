/**
 * @file calibrator.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include "../common.h"

#include <opencv2/opencv.hpp>
#include <vector>

namespace slmaster {
namespace calibration {

class SLMASTER_API Calibrator {
  public:
    enum ThreshodMethod { OTSU = 0, ADAPTED };
    Calibrator();
    virtual ~Calibrator();
    inline void emplace(const cv::Mat &img) { imgs_.emplace_back(img); }
    inline void erase(const int index) { imgs_.erase(imgs_.begin() + index); }
    inline std::vector<std::vector<cv::Point2f>> &imgPoints() {
        return imgPoints_;
    }
    inline std::vector<std::vector<cv::Point3f>> &worldPoints() {
        return worldPoints_;
    }
    inline std::vector<cv::Mat> &imgs() { return imgs_; }
    inline cv::Size imgSize() {
        return imgs_.empty() ? cv::Size(0, 0) : imgs_[0].size();
    }
    inline std::vector<cv::Mat> &drawedFeaturesImgs() {
        return drawedFeaturesImgs_;
    }
    inline std::vector<std::vector<cv::Point2f>> &errors() {
        return errors_;
    }
    virtual void setDistance(const float trueDistance) = 0;
    virtual void setRadius(const std::vector<float> &radius) = 0;
    virtual bool
    findFeaturePoints(const cv::Mat &img, const cv::Size &featureNums,
                      std::vector<cv::Point2f> &points,
                      const ThreshodMethod threshodType = ADAPTED) = 0;
    // TODO(@Evans Liu):暂未用到blobBlack
    virtual double calibrate(const std::vector<cv::Mat> &imgs,
                             cv::Mat &intrinsic, cv::Mat &distort,
                             const cv::Size &featureNums, float &process,
                             const ThreshodMethod threshodType = ADAPTED,
                             const bool blobBlack = true) = 0;
  protected:
    std::vector<cv::Mat> imgs_;
    std::vector<cv::Mat> drawedFeaturesImgs_;
    std::vector<std::vector<cv::Point2f>> imgPoints_;
    std::vector<std::vector<cv::Point3f>> worldPoints_;
    std::vector<std::vector<cv::Point2f>> errors_;

  private:
};

} // namespace calibration
} // namespace slmaster

#endif
