#ifndef __TOOL_H_
#define __TOOL_H_

#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common.h"
#include "caliInfo.h"

namespace slmaster {
void SLMASTER_API fromDispairtyMapToCloud(const cv::Mat& disparityMap, const cv::Mat& textureMap, const CaliInfo& caliInfo, pcl::PointCloud<pcl::PointXYZRGB>& cloud, const bool isRemapColorCam);

void SLMASTER_API fromDepthMapToCloud(const cv::Mat& depthMap, const cv::Mat& textureMap, const CaliInfo& caliInfo, pcl::PointCloud<pcl::PointXYZRGB>& cloud, const bool isRemapColorCam);
}

#endif // __TOOL_H_
