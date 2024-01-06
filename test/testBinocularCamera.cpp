#include <binoocularCamera.h>
#include <gtest/gtest.h>

#include <pcl/io/ply_io.h>

const std::string configPath = "../../gui/qml/res/config/binoocularCameraConfig.json";

TEST(BinocularCamera, contrust) {
    slmaster::BinocularCamera* pCamera = new slmaster::BinocularCamera(configPath);
    ASSERT_NE(pCamera, nullptr);
}

TEST(BinocularCamera, isFind) {
    slmaster::BinocularCamera* pCamera = new slmaster::BinocularCamera(configPath);
    auto info = pCamera->getCameraInfo();
    ASSERT_EQ(info.isFind_, true);
}

TEST(BinocularCamera, connect) {
    slmaster::BinocularCamera* pCamera = new slmaster::BinocularCamera(configPath);
    auto info = pCamera->getCameraInfo();
    ASSERT_EQ(pCamera->connect(), true);
    pCamera->disConnect();
}
/*
TEST(BinocularCamera, disConnect) {
    sl::slCamera::BinocularCamera* pCamera = new sl::slCamera::BinocularCamera(configPath);
    auto info = pCamera->getCameraInfo();
    pCamera->connect();
    ASSERT_EQ(pCamera->disConnect(), true);
}
TEST(BinocularCamera, disableDepth) {
    sl::slCamera::BinocularCamera* pCamera = new sl::slCamera::BinocularCamera(configPath);
    auto info = pCamera->getCameraInfo();
    bool isSucess = pCamera->connect();
    pCamera->setDepthCameraEnabled(false);
    sl::slCamera::FrameData frameData;
    pCamera->capture(frameData);
}

TEST(BinocularCamera, capture) {
    sl::slCamera::BinocularCamera* pCamera = new sl::slCamera::BinocularCamera(configPath);
    auto info = pCamera->getCameraInfo();
    bool isSucess = pCamera->setDepthCameraEnabled(true);
    bool isConnect = pCamera->connect();
    isSucess = pCamera->isConnect();
    sl::slCamera::FrameData frameData;
    isSucess = pCamera->capture(frameData);
    pCamera->disConnect();
}
*/
/*
TEST(BinocularCamera, offlineCapture) {
    //请将内参路径设置为测试数据中cali文件夹中的标定文件
    sl::slCamera::BinocularCamera* pCamera = new sl::slCamera::BinocularCamera(configPath);

    std::vector<cv::Mat> imgsL, imgsR;
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat imgL = cv::imread("../../test/data/phaseGray/L/" + std::to_string(i) + ".bmp", 0);
        cv::Mat imgR = cv::imread("../../test/data/phaseGray/R/" + std::to_string(i) + ".bmp", 0);
        imgsL.emplace_back(imgL);
        imgsR.emplace_back(imgR);
    }   

    sl::slCamera::FrameData data;
    pCamera->offLineCapture(imgsL, imgsR, data);
    
    pcl::io::savePLYFile("../../test/data/out/cloud.ply", data.__pointCloud);
}
*/
