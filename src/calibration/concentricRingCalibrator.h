/**
 * @file concentricRingCalibrator.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef CONCENTRICRINGCALIBRATOR_H
#define CONCENTRICRINGCALIBRATOR_H

#include "Eigen/Eigen"

#include "calibrator.h"
#include "edgesSubPix.h"

namespace slmaster {
namespace calibration {

class ConcentricRingCalibrator : public Calibrator {
  public:
    ConcentricRingCalibrator();
    ~ConcentricRingCalibrator();
    bool
    findFeaturePoints(const cv::Mat &img, const cv::Size &featureNums,
                      std::vector<cv::Point2f> &points,
                      const ThreshodMethod threshodType = ADAPTED) override;
    double calibrate(const std::vector<cv::Mat> &imgs, cv::Mat &intrinsic,
                     cv::Mat &distort, const cv::Size &featureNums,
                     float &process,
                     const ThreshodMethod threshodType = ADAPTED,
                     const bool blobBlack = true) override;
    /**
     * @brief 设置同心圆环半径
     * @param radius 半径，由小至大
     */
    inline void setRadius(const std::vector<float> &radius) override {
        radius_ = radius;
    };

  private:
    inline void setDistance(const float trueDistance) override {
        distance_ = trueDistance;
    };
    bool findConcentricRingGrid(const cv::Mat &inputImg,
                                const cv::Size patternSize,
                                const std::vector<float> radius,
                                std::vector<cv::Point2f> &centerPoints);
    /**
     * @brief 获取椭圆一般方程的矩阵形式
     *
     * @param rotateRect 输入，Opencv中的旋转矩阵结构体，可通过fitEllipse()获得
     * @param quationMat 输出，椭圆一般方程的矩阵形式
     */
    void getEllipseNormalQuation(const cv::RotatedRect &rotateRect,
                                 cv::Mat &quationMat);
    /**
     * @brief 排序归类拟合的椭圆
     *
     * @param rects 输入，椭圆
     * @param rectsPoints 输入，拟合椭圆时所用的点
     * @param sortedCenters 输入，排序好的中心点
     * @param sortedRects 输出，排序归类后的椭圆集合
     */
    void sortElipse(const std::vector<cv::RotatedRect> &rects,
                    const std::vector<std::vector<cv::Point2f>> &rectsPoints,
                    const std::vector<cv::Point2f> &sortedCenters,
                    std::vector<std::vector<cv::RotatedRect>> &sortedRects);
    /**
     * @brief 对特征点进行分类(暂时弃用)：
     *        *首先根据y坐标将所有点进行一次排序，则每规定数量的特征点即为同行特征点
     *        *其次再根据x坐标对每行特征点进行排序
     *        *排序方式为从小到大排序
     *        *该排序方法对大旋转角图片失效，有待改进，论文可参考
     *        *@ Ellipse Detection ER_Davies 1990
     *
     * @param inputPoints 输入，特征点
     * @param patternSize 输入，特征点行列数目
     * @param outputPoints 输出，已排序特征点
     * @return true 返回值，成功
     * @return false 返回值，失败
     */
    bool sortKeyPoints(const std::vector<cv::Point2f> &inputPoints,
                       const cv::Size patternSize,
                       std::vector<cv::Point2f> &outputPoints);
    /**
     * @brief 通过方程求解双圆环中心点
     *
     * @param normalMats 输入，分类好的椭圆方程
     * @param radius 输入，双圆环半径(从小到大)
     * @param points 输出，求解得到的双圆环中心点
     */
    void getRingCenters(const std::vector<std::vector<cv::Mat>> &normalMats,
                        const std::vector<float> &radius,
                        std::vector<cv::Point2f> &points);
    /**
     * @brief 计算点P到直线AB的距离
     *
     * @param pointP 输入，点P坐标
     * @param pointA 输入，点A坐标
     * @param pointA 输入，点B坐标
     * @param float 返回值，点P到直线AB的距离
     */
    float getDist_P2L(cv::Point2f pointP, cv::Point2f pointA,
                      cv::Point2f pointB);
    // 由小至大排序
    std::vector<float> radius_;
    float distance_;
};
} // namespace calibration
} // namespace slmaster

#endif // CONCENTRICRINGCALIBRATOR_H
