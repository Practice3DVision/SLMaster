#include "tool.h"

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

namespace slmaster {
void fromDispairtyMapToCloud(const cv::Mat& disparityMap, const cv::Mat& textureMap, const CaliInfo& caliInfo, pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
    const int rows = disparityMap.rows;
    const int cols = disparityMap.cols;
    const float f = caliInfo.info_.Q_.at<double>(2, 3);
    const float tx = -1.f / caliInfo.info_.Q_.at<double>(3, 2);
    const float cxlr = caliInfo.info_.Q_.at<double>(3, 3) * tx;
    const float cx = -1.f / caliInfo.info_.Q_.at<double>(0, 3);
    const float cy = -1.f / caliInfo.info_.Q_.at<double>(1, 3);

    cv::Mat R1InvCV = caliInfo.info_.R1_.inv();
    R1InvCV.convertTo(R1InvCV, CV_32FC1);
    Eigen::Matrix3f R1InvEigen;
    cv::cv2eigen(R1InvCV, R1InvEigen);

    cloud.clear();

    std::mutex mutex;

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for(int i = range.start; i< range.end; ++i) {
            auto disparityMapPtr = disparityMap.ptr<float>(i);
            auto textureMapPtr = textureMap.ptr<cv::Vec3b>(i);
            for (int j = 0; j < cols; ++j) {
                if(std::abs(disparityMapPtr[j]) < 0.001f) {
                    continue;
                }

                Eigen::Vector3f point;
                point(0, 0) = -1.f * tx * (j - cx) / (disparityMapPtr[j] - cxlr);
                point(1, 0) = -1.f * tx * (i - cy) / (disparityMapPtr[j] - cxlr);
                point(2, 0) = -1.f * tx * f / (disparityMapPtr[j] - cxlr);

                point = R1InvEigen * point;

                pcl::PointXYZRGB cloudPoint(point(0, 0), point(1, 0), point(2, 0), textureMapPtr[j][2], textureMapPtr[j][1], textureMapPtr[j][0]);

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

void fromDepthMapToCloud(const cv::Mat& depthMap, const cv::Mat& textureMap, const CaliInfo& caliInfo, pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
    const int rows = depthMap.rows;
    const int cols = depthMap.cols;

    cv::Mat M1InvCV = caliInfo.info_.M1_.inv();
    M1InvCV.convertTo(M1InvCV, CV_32FC1);
    Eigen::Matrix3f M1InvEigen;
    cv::cv2eigen(M1InvCV, M1InvEigen);

    cloud.clear();

    std::mutex mutex;

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for(int i = range.start; i< range.end; ++i) {
            auto depthMapPtr = depthMap.ptr<float>(i);
            auto textureMapPtr = textureMap.ptr<cv::Vec3b>(i);
            for (int j = 0; j < cols; ++j) {
                if(std::abs(depthMapPtr[j]) < 0.001f) {
                    continue;
                }

                Eigen::Vector3f point = M1InvEigen * Eigen::Vector3f(j ,i, 1) * depthMapPtr[j];

                pcl::PointXYZRGB cloudPoint(point(0, 0), point(1, 0), point(2, 0), textureMapPtr[j][2], textureMapPtr[j][1], textureMapPtr[j][0]);

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

}
