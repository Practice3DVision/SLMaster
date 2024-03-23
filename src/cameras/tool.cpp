#include "tool.h"

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

namespace slmaster {
namespace cameras {
void fromDispairtyMapToCloud(const cv::Mat &disparityMap,
                             const cv::Mat &textureMap,
                             const CaliInfo &caliInfo,
                             pcl::PointCloud<pcl::PointXYZRGB> &cloud,
                             cv::Mat &textureCamDepthMap,
                             const bool isRemapColorCam) {
    const int rows = disparityMap.rows;
    const int cols = disparityMap.cols;
    const float f = caliInfo.info_.Q_.at<double>(2, 3);
    const float tx = -1.f / caliInfo.info_.Q_.at<double>(3, 2);
    const float cxlr = caliInfo.info_.Q_.at<double>(3, 3) * tx;
    const float cx = -1.f * caliInfo.info_.Q_.at<double>(0, 3);
    const float cy = -1.f * caliInfo.info_.Q_.at<double>(1, 3);

    cv::Mat R1InvCV = caliInfo.info_.R1_.inv();
    R1InvCV.convertTo(R1InvCV, CV_32FC1);
    Eigen::Matrix3f R1InvEigen;
    cv::cv2eigen(R1InvCV, R1InvEigen);

    Eigen::Matrix3f R13Eigen, M3Eigen;
    Eigen::Vector3f T13Eigen;
    if (isRemapColorCam) {
        cv::Mat R13CV = caliInfo.info_.Rlc_;
        R13CV.convertTo(R13CV, CV_32FC1);
        cv::cv2eigen(R13CV, R13Eigen);

        cv::Mat T13CV = caliInfo.info_.Tlc_;
        T13CV.convertTo(T13CV, CV_32FC1);
        cv::cv2eigen(T13CV, T13Eigen);

        cv::Mat M3CV = caliInfo.info_.M3_;
        T13CV.convertTo(T13CV, CV_32FC1);
        cv::cv2eigen(T13CV, T13Eigen);
    }

    cloud.clear();
    textureCamDepthMap =
        cv::Mat::zeros(disparityMap.size(), disparityMap.type());

    std::mutex mutex;

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto disparityMapPtr = disparityMap.ptr<float>(i);
            auto textureCamDepthMapPtr = textureCamDepthMap.ptr<float>(i);
            auto textureMapPtr = textureMap.ptr<cv::Vec3b>(i);
            for (int j = 0; j < cols; ++j) {
                if (std::abs(disparityMapPtr[j]) < 0.001f) {
                    continue;
                }

                Eigen::Vector3f point;
                point(0, 0) =
                    -1.f * tx * (j - cx) / (disparityMapPtr[j] - cxlr);
                point(1, 0) =
                    -1.f * tx * (i - cy) / (disparityMapPtr[j] - cxlr);
                point(2, 0) = -1.f * tx * f / (disparityMapPtr[j] - cxlr);

                point = R1InvEigen * point;

                pcl::PointXYZRGB cloudPoint(point(0, 0), point(1, 0),
                                            point(2, 0), 255, 255, 255);

                if (isRemapColorCam) {
                    point = M3Eigen * (R13Eigen * point + T13Eigen);
                    const int xLoc = point(0, 0) / point(2, 0);
                    const int yLoc = point(1, 0) / point(2, 0);
                    if (xLoc >= 0 && xLoc < cols && yLoc >= 0 && yLoc < rows) {
                        auto color = textureMap.ptr<cv::Vec3b>(yLoc)[xLoc];
                        cloudPoint.r = color[2];
                        cloudPoint.g = color[1];
                        cloudPoint.b = color[0];

                        {
                            std::lock_guard<std::mutex> guard(mutex);
                            textureCamDepthMap.ptr<float>(yLoc)[xLoc] =
                                cloudPoint.z;
                        }
                    }
                } else {
                    cloudPoint.r = textureMapPtr[j][2];
                    cloudPoint.g = textureMapPtr[j][1];
                    cloudPoint.b = textureMapPtr[j][0];

                    textureCamDepthMapPtr[j] = cloudPoint.z;
                }

                {
                    std::lock_guard<std::mutex> guard(mutex);
                    cloud.emplace_back(cloudPoint);
                }
            }
        }
    });

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = false;
}

void fromDepthMapToCloud(const cv::Mat &depthMap, const cv::Mat &textureMap,
                         const CaliInfo &caliInfo,
                         pcl::PointCloud<pcl::PointXYZRGB> &cloud,
                         cv::Mat &textureCamDepthMap,
                         const bool isRemapColorCam) {
    const int rows = depthMap.rows;
    const int cols = depthMap.cols;

    cv::Mat M1InvCV = caliInfo.info_.M1_.inv();
    M1InvCV.convertTo(M1InvCV, CV_32FC1);

    Eigen::Matrix3f M1InvEigen, R13Eigen, M3Eigen;
    cv::cv2eigen(M1InvCV, M1InvEigen);

    Eigen::Vector3f T13Eigen;

    if (isRemapColorCam) {
        cv::Mat temp;
        caliInfo.info_.Rlc_.convertTo(temp, CV_32FC1);
        cv::cv2eigen(temp, R13Eigen);
        caliInfo.info_.Tlc_.convertTo(temp, CV_32FC1);
        cv::cv2eigen(temp, T13Eigen);
        caliInfo.info_.M3_.convertTo(temp, CV_32FC1);
        cv::cv2eigen(temp, M3Eigen);
    }

    cloud.clear();
    textureCamDepthMap = isRemapColorCam
                             ? cv::Mat(textureMap.size(), textureMap.type())
                             : textureMap.clone();

    std::mutex mutex;

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto depthMapPtr = depthMap.ptr<float>(i);
            auto textureMapPtr = textureMap.ptr<cv::Vec3b>(i);
            for (int j = 0; j < cols; ++j) {
                if (std::abs(depthMapPtr[j]) < 0.001f) {
                    continue;
                }

                Eigen::Vector3f point =
                    M1InvEigen * Eigen::Vector3f(j, i, 1) * depthMapPtr[j];

                pcl::PointXYZRGB cloudPoint(point(0, 0), point(1, 0),
                                            point(2, 0), 255, 255, 255);

                if (isRemapColorCam) {
                    point = M3Eigen * (R13Eigen * point + T13Eigen);
                    const int xLoc = point(0, 0) / point(2, 0);
                    const int yLoc = point(1, 0) / point(2, 0);
                    if (xLoc >= 0 && xLoc < cols - 1 && yLoc >= 0 &&
                        yLoc < rows - 1) {
                        auto color = textureMap.ptr<cv::Vec3b>(yLoc)[xLoc];
                        cloudPoint.r = color[2];
                        cloudPoint.g = color[1];
                        cloudPoint.b = color[0];

                        {
                            std::lock_guard<std::mutex> guard(mutex);
                            textureCamDepthMap.ptr<float>(yLoc)[xLoc] =
                                cloudPoint.z;
                        }
                    }
                } else {
                    cloudPoint.r = textureMapPtr[j][2];
                    cloudPoint.g = textureMapPtr[j][1];
                    cloudPoint.b = textureMapPtr[j][0];
                }

                {
                    std::lock_guard<std::mutex> guard(mutex);
                    cloud.emplace_back(cloudPoint);
                }
            }
        }
    });

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = false;
}
} // namespace cameras
} // namespace slmaster
