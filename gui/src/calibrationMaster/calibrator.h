#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class Calibrator {
  public:
    enum ThreshodMethod { OTSU = 0, ADAPTED };
    Calibrator();
    virtual ~Calibrator();
    inline void emplace(const cv::Mat &img) { imgs_.emplace_back(img); }
    inline void erase(const int index) { imgs_.erase(imgs_.begin() + index); }
    inline const std::vector<std::vector<cv::Point2f>> &imgPoints() {
        return imgPoints_;
    }
    inline const std::vector<std::vector<cv::Point3f>> &worldPoints() {
        return worldPoints_;
    }
    inline const std::vector<cv::Mat> &imgs() { return imgs_; }
    inline cv::Size imgSize() {
        return imgs_.empty() ? cv::Size(0, 0) : imgs_[0].size();
    }
    inline const std::vector<cv::Mat>& drawedFeaturesImgs() { return drawedFeaturesImgs_; }
    inline const std::vector<std::vector<cv::Point2f>>& errors() { return errors_; }
    virtual void setDistance(const float trueDistance) = 0;
    virtual void setRadius(const std::vector<float> &radius) = 0;
    virtual bool
    findFeaturePoints(const cv::Mat &img, const cv::Size &featureNums,
                      std::vector<cv::Point2f> &points,
                      const ThreshodMethod threshodType = ADAPTED) = 0;
    // TODO(@Liu Yunhuang):暂未用到blobBlack
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

#endif
