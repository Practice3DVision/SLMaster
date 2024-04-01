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

    Eigen::Matrix3f R13Eigen, M3Eigen, M1Eigen;
    Eigen::Vector3f T13Eigen;
    if (isRemapColorCam) {
        cv::Mat R13CV = caliInfo.info_.Rlc_;
        R13CV.convertTo(R13CV, CV_32FC1);
        cv::cv2eigen(R13CV, R13Eigen);

        cv::Mat T13CV = caliInfo.info_.Tlc_;
        T13CV.convertTo(T13CV, CV_32FC1);
        cv::cv2eigen(T13CV, T13Eigen);

        cv::Mat M3CV = caliInfo.info_.M3_;
        M3CV.convertTo(M3CV, CV_32FC1);
        cv::cv2eigen(M3CV, M3Eigen);
    }

    cv::Mat M1CV = caliInfo.info_.M1_;
    M1CV.convertTo(M1CV, CV_32FC1);
    cv::cv2eigen(M1CV, M1Eigen);

    cloud.clear();
    textureCamDepthMap = cv::Mat::zeros(textureMap.size(), disparityMap.type());

    std::mutex mutex;

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto disparityMapPtr = disparityMap.ptr<float>(i);
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
                Eigen::Vector3f imgPoint;

                if (isRemapColorCam) {
                    point = R13Eigen * point + T13Eigen;
                    imgPoint = M3Eigen * point;
                } else {
                    imgPoint = M1Eigen * point;
                }

                const int xLoc = imgPoint(0, 0) / imgPoint(2, 0);
                const int yLoc = imgPoint(1, 0) / imgPoint(2, 0);

                if (xLoc >= 0 && xLoc < cols && yLoc >= 0 && yLoc < rows) {
                    auto color = textureMap.ptr<cv::Vec3b>(yLoc)[xLoc];
                    pcl::PointXYZRGB cloudPoint(point(0, 0), point(1, 0),
                                                point(2, 0), color[2], color[1],
                                                color[0]);

                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        textureCamDepthMap.ptr<float>(yLoc)[xLoc] =
                            cloudPoint.z;
                        cloud.emplace_back(cloudPoint);
                    }
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

    Eigen::Matrix3f M1InvEigen, R13Eigen, M3Eigen, M1Eigen;
    cv::cv2eigen(M1InvCV, M1InvEigen);

    cv::Mat M1CV = caliInfo.info_.M1_;
    M1CV.convertTo(M1CV, CV_32FC1);
    cv::cv2eigen(M1CV, M1Eigen);

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
                             ? cv::Mat(textureMap.size(), depthMap.type())
                             : depthMap.clone();

    std::mutex mutex;

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            auto depthMapPtr = depthMap.ptr<float>(i);
            for (int j = 0; j < cols; ++j) {
                if (std::abs(depthMapPtr[j]) < 0.001f) {
                    continue;
                }

                Eigen::Vector3f point =
                    M1InvEigen * Eigen::Vector3f(j, i, 1) * depthMapPtr[j];
                Eigen::Vector3f imgPoint;

                if (isRemapColorCam) {
                    point = R13Eigen * point + T13Eigen;
                    imgPoint = M3Eigen * point;
                } else {
                    imgPoint = M1Eigen * point;
                }

                const int xLoc = imgPoint(0, 0) / imgPoint(2, 0);
                const int yLoc = imgPoint(1, 0) / imgPoint(2, 0);

                if (xLoc >= 0 && xLoc < cols - 1 && yLoc >= 0 &&
                    yLoc < rows - 1) {
                    auto color = textureMap.ptr<cv::Vec3b>(yLoc)[xLoc];
                    pcl::PointXYZRGB cloudPoint(point(0, 0), point(1, 0),
                                                point(2, 0), color[2], color[1],
                                                color[0]);

                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        cloud.emplace_back(cloudPoint);
                        if (isRemapColorCam) {
                            textureCamDepthMap.ptr<float>(yLoc)[xLoc] =
                                cloudPoint.z;
                        }
                    }
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
