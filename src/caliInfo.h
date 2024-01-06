/**
 * @file matrixsInfo.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  标定信息工具类
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef __CALI_INFO_H_
#define __CALI_INFO_H_

#include "common.h"

#include <opencv2/opencv.hpp>

namespace slmaster {
/** @brief 信息结构体    */
struct SLMASTER_API Info {
    /** \左相机内参矩阵(CV_64FC1) **/
    cv::Mat M1_;
    /** \右相机内参矩阵(CV_64FC1) **/
    cv::Mat M2_;
    /** \彩色相机内参矩阵(CV_64FC1) **/
    cv::Mat M3_;
    /** \投影仪内参矩阵(CV_64FC1) **/
    cv::Mat M4_;
    /** \左相机相机坐标系到右相机坐标系的旋转矩阵(CV_64FC1) **/
    cv::Mat R1_;
    /** \右相机相机坐标系到左相机坐标系的旋转矩阵(CV_64FC1) **/
    cv::Mat R2_;
    /** \左相机相机坐标系到右相机的投影矩阵(CV_64FC1) **/
    cv::Mat P1_;
    /** \右相机相机坐标系到世界坐标系的投影矩阵(CV_64FC1) **/
    cv::Mat P2_;
    /** \左相机的畸变矩阵(CV_64FC1) **/
    cv::Mat D1_;
    /** \右相机的畸变矩阵(CV_64FC1) **/
    cv::Mat D2_;
    /** \彩色相机的畸变矩阵(CV_64FC1) **/
    cv::Mat D3_;
    /** \投影仪的畸变矩阵(CV_64FC1) **/
    cv::Mat D4_;
    /** \深度映射矩阵(CV_64FC1) **/
    cv::Mat Q_;
    /** \左相机八参数矩阵(CV_64FC1) **/
    cv::Mat K1_;
    /** \右相机八参数矩阵(CV_64FC1) **/
    cv::Mat K2_;
    /** \左相机至右相机旋转矩阵(CV_64FC1) **/
    cv::Mat Rlr_;
    /** \左相机至右相机平移矩阵(CV_64FC1) **/
    cv::Mat Tlr_;
    /** \左相机至投影仪旋转矩阵(CV_64FC1) **/
    cv::Mat Rlp_;
    /** \左相机至投影仪平移矩阵(CV_64FC1) **/
    cv::Mat Tlp_;
    /** \右相机至投影仪旋转矩阵(CV_64FC1) **/
    cv::Mat Rrp_;
    /** \右相机至投影仪平移矩阵(CV_64FC1) **/
    cv::Mat Trp_;
    /** \左相机至右相机本质矩阵(CV_64FC1) **/
    cv::Mat E_;
    /** \左相机至右相机本质矩阵(CV_64FC1) **/
    cv::Mat F_;
    /** \左相机至彩色相机旋转矩阵(CV_64FC1) **/
    cv::Mat Rlc_;
    /** \左相机至彩色相机平移矩阵(CV_64FC1) **/
    cv::Mat Tlc_;
    /** \相机幅面(CV_64FC1) **/
    cv::Mat S_;
};

/** @brief 相机标定信息类 */
class SLMASTER_API CaliInfo {
  public:
    CaliInfo() = default;
    ~CaliInfo() = default;
    CaliInfo(IN const std::string filePath);
    /** \读取到的校正信息 **/
    Info info_;
  private:
};
} // namespace slmaster
#endif // !__CALI_INFO_H_
