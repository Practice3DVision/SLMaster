#include "cameraFactory.h"

#include <gtest/gtest.h>
/*
TEST(HuarayCameraLib, getCamera) {
    auto pCameraFactory = new device::camera::CameraFactory();
    device::camera::Camera* pCamera = pCameraFactory->getCamera("Left", device::camera::CameraFactory::CameraManufactor::Huaray);
    ASSERT_NE(pCamera, nullptr);
}

TEST(HuarayCameraLib, getCameraInfo) {
    auto pCameraFactory = new device::camera::CameraFactory();
    device::camera::Camera* pCamera = pCameraFactory->getCamera("Left", device::camera::CameraFactory::CameraManufactor::Huaray);
    auto info = pCamera->getCameraInfo();
    ASSERT_EQ(info.isFind_, true);
}

TEST(HuarayCameraLib, connect) {
    auto pCameraFactory = new device::camera::CameraFactory();
    device::camera::Camera* pCamera = pCameraFactory->getCamera("Left", device::camera::CameraFactory::CameraManufactor::Huaray);
    bool isSucess = pCamera->connect();
    ASSERT_EQ(isSucess, true);
    pCamera->disConnect();
}

TEST(HuarayCameraLib, disConnect) {
    auto pCameraFactory = new device::camera::CameraFactory();
    device::camera::Camera* pCamera = pCameraFactory->getCamera("Left", device::camera::CameraFactory::CameraManufactor::Huaray);
    bool isSucess = pCamera->connect();
    isSucess = pCamera->start();
    isSucess = pCamera->disConnect();
    ASSERT_EQ(pCamera->isConnect(), false);
}

TEST(HuarayCameraLib, start) {
    auto pCameraFactory = new device::camera::CameraFactory();
    device::camera::Camera* pCamera = pCameraFactory->getCamera("Left", device::camera::CameraFactory::CameraManufactor::Huaray);
    bool isSucess = pCamera->connect();
    isSucess = pCamera->start();
    ASSERT_EQ(isSucess, true);
    pCamera->disConnect();
}

TEST(HuarayCameraLib, capture) {
    auto pCameraFactory = new device::camera::CameraFactory();
    device::camera::Camera* pCamera = pCameraFactory->getCamera("Left", device::camera::CameraFactory::CameraManufactor::Huaray);
    bool isSucess = pCamera->connect();
    isSucess = pCamera->start();
    cv::Mat img = pCamera->capture();
    ASSERT_NE(img.rows, 0);
    pCamera->disConnect();
}

TEST(HuarayCameraLib, popImg) {
    auto pCameraFactory = new device::camera::CameraFactory();
    device::camera::Camera* pCamera = pCameraFactory->getCamera("Left", device::camera::CameraFactory::CameraManufactor::Huaray);
    bool isSucess = pCamera->connect();
    isSucess = pCamera->start();
    cv::Mat img = pCamera->capture();
    cv::Mat popImg = pCamera->popImg();
    ASSERT_EQ(pCamera->getImgs().size(), 0);
    pCamera->disConnect();
}
*/
TEST(HuarayCameraLib, setNumberAttribute) {
    auto pCameraFactory = new device::camera::CameraFactory();
    device::camera::Camera* pCamera = pCameraFactory->getCamera("Left", device::camera::CameraFactory::CameraManufactor::Huaray);
    bool isSucess = pCamera->connect();
    pCamera->setTrigMode(device::camera::trigLine);
    pCamera->start();
    isSucess = pCamera->setNumberAttribute("ExposureTime", 4000);
    ASSERT_EQ(isSucess, true);
    pCamera->disConnect();
}
