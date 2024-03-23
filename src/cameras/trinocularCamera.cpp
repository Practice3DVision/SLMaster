#include "trinocularCamera.h"

#include "binosSinusCompleGrayCodePattern.h"
#include "tool.h"

namespace slmaster {
namespace cameras {
TrinocularCamera::TrinocularCamera(IN const std::string jsonPath)
    : SLCamera(jsonPath), jsonPath_(jsonPath), isInitial_(false),
      isCaptureStop_(true) {
    if (loadParams(jsonPath, jsonVal_)) {
        if (booleanProperties_["Gpu Accelerate"]) {
#ifndef WITH_CUDASTRUCTUREDLIGHT_MODULE
            std::clog << "lib isn't build with cuda, we will disable it!"
                      << std::endl;
            booleanProperties_["Gpu Accelerate"] = false;
#endif //! WITH_CUDASTRUCTUREDLIGHT_MODULE
        }

        isInitial_ = true;
    }
}

TrinocularCamera::~TrinocularCamera() { updateCamera(); }

bool TrinocularCamera::loadParams(const std::string jsonPath,
                                  Json::Value &jsonVal) {
    if (!readJsonFile(jsonPath, jsonVal)) {
        std::cerr << "trinocular camera parse json file error!" << std::endl;
        return false;
    }

    parseArray(jsonVal["camera"]["device"], false);
    parseArray(jsonVal["camera"]["algorithm"], false);

    if (caliInfo_) {
        delete caliInfo_;
        caliInfo_ = nullptr;
    }

    caliInfo_ = new CaliInfo(stringProperties_["Calibration File Path"]);

    return true;
}

bool TrinocularCamera::saveParams(const std::string jsonPath,
                                  Json::Value &jsonVal) {
    if (jsonVal.empty()) {
        std::cerr << "trinocular camera write json file error, json value is "
                     "empty! \n"
                  << std::endl;
        return false;
    }

    parseArray(jsonVal["camera"]["device"], true);
    parseArray(jsonVal["camera"]["algorithm"], true);

    std::ofstream writeStream(jsonPath);
    Json::StyledWriter writer;
    writeStream << writer.write(jsonVal);

    return true;
}

void TrinocularCamera::parseArray(Json::Value &jsonVal, const bool isWrite) {
    const int numOfCameraInfo = jsonVal.size();

    for (int i = 0; i < numOfCameraInfo; ++i) {
        const std::string titleString = jsonVal[i]["type"].asString();
        if (titleString == "string") {
            if (!isWrite) {
                setStringAttribute(jsonVal[i]["title"].asString(),
                                   jsonVal[i]["data"].asString());
            } else {
                jsonVal[i]["data"] =
                    stringProperties_[jsonVal[i]["title"].asString()];
            }
        } else if (titleString == "number" || titleString == "enum") {
            if (!isWrite) {
                setNumberAttribute(jsonVal[i]["title"].asString(),
                                   jsonVal[i]["data"].asDouble());
            } else {
                jsonVal[i]["data"] =
                    numbericalProperties_[jsonVal[i]["title"].asString()];
            }
        } else if (titleString == "bool") {
            if (!isWrite) {
                setBooleanAttribute(jsonVal[i]["title"].asString(),
                                    jsonVal[i]["data"].asBool());
            } else {
                jsonVal[i]["data"] =
                    (int)booleanProperties_[jsonVal[i]["title"].asString()];
            }
        }
    }
}

SLCameraInfo TrinocularCamera::getCameraInfo() {
    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    auto pLeftCamera = cameraFactory_.getCamera(
        stringProperties_["Left Camera Name"], manufator);
    auto pRightCamera = cameraFactory_.getCamera(
        stringProperties_["Right Camera Name"], manufator);
    auto pColorCamera = cameraFactory_.getCamera(
        stringProperties_["Color Camera Name"], manufator);
    auto leftCameraInfo = pLeftCamera->getCameraInfo();
    auto rightCameraInfo = pRightCamera->getCameraInfo();
    auto colorCameraInfo = pColorCamera->getCameraInfo();

    auto pProjector =
        projectorFactory_.getProjector(stringProperties_["DLP Evm"]);

    device::ProjectorInfo projectorInfo =
        pProjector ? pProjector->getInfo() : device::ProjectorInfo();
    if (!pProjector) {
        projectorInfo.isFind_ = false;
    }

    SLCameraInfo slCameraInfo;
    slCameraInfo.isFind_ = leftCameraInfo.isFind_ && rightCameraInfo.isFind_ &&
                           colorCameraInfo.isFind_ && projectorInfo.isFind_;
    slCameraInfo.cameraName_ =
        slCameraInfo.isFind_ ? stringProperties_["Camera Name"] : "NOT_FOUND";
    slCameraInfo.intrinsic_ = slCameraInfo.isFind_
                                  ? caliInfo_->info_.M1_
                                  : cv::Mat::zeros(3, 3, CV_32FC1);

    return slCameraInfo;
}

bool TrinocularCamera::connect() {
    bool connectState = false;

    try {
        const device::CameraFactory::CameraManufactor manufator =
            stringProperties_["2D Camera Manufactor"] == "Huaray"
                ? device::CameraFactory::Huaray
                : device::CameraFactory::Halcon;
        const bool connectLeftCamera =
            cameraFactory_
                .getCamera(stringProperties_["Left Camera Name"], manufator)
                ->connect();
        const bool connectRightCamera =
            cameraFactory_
                .getCamera(stringProperties_["Right Camera Name"], manufator)
                ->connect();
        const bool connectColorCamera =
            cameraFactory_
                .getCamera(stringProperties_["Color Camera Name"], manufator)
                ->connect();
        const bool connectProjector =
            projectorFactory_.getProjector(stringProperties_["DLP Evm"])
                ->connect();
        connectState = connectLeftCamera && connectRightCamera &&
                       connectColorCamera && connectProjector;

        if (connectState) {
            cameraFactory_
                .getCamera(stringProperties_["Left Camera Name"], manufator)
                ->start();
            cameraFactory_
                .getCamera(stringProperties_["Right Camera Name"], manufator)
                ->start();
            cameraFactory_
                .getCamera(stringProperties_["Color Camera Name"], manufator)
                ->start();
            cameraFactory_
                .getCamera(stringProperties_["Color Camera Name"], manufator)
                ->setEnumAttribute("BalanceRatioSelector", "Red");
            cameraFactory_
                .getCamera(stringProperties_["Color Camera Name"], manufator)
                ->setNumberAttribute("Balance", 1.72477);
            cameraFactory_
                .getCamera(stringProperties_["Color Camera Name"], manufator)
                ->setEnumAttribute("BalanceRatioSelector", "Green");
            cameraFactory_
                .getCamera(stringProperties_["Color Camera Name"], manufator)
                ->setNumberAttribute("Balance", 1.0);
            cameraFactory_
                .getCamera(stringProperties_["Color Camera Name"], manufator)
                ->setEnumAttribute("BalanceRatioSelector", "Blue");
            cameraFactory_
                .getCamera(stringProperties_["Color Camera Name"], manufator)
                ->setNumberAttribute("Balance", 1.66687);

            updateEnableDepthCamera();
            updateExposureTime();
            updateLightStrength();
        } else {
            if (cameraFactory_
                    .getCamera(stringProperties_["Left Camera Name"], manufator)
                    ->isConnect()) {
                cameraFactory_
                    .getCamera(stringProperties_["Left Camera Name"], manufator)
                    ->disConnect();
            }

            if (cameraFactory_
                    .getCamera(stringProperties_["Right Camera Name"],
                               manufator)
                    ->isConnect()) {
                cameraFactory_
                    .getCamera(stringProperties_["Right Camera Name"],
                               manufator)
                    ->disConnect();
            }

            if (cameraFactory_
                    .getCamera(stringProperties_["Color Camera Name"],
                               manufator)
                    ->isConnect()) {
                cameraFactory_
                    .getCamera(stringProperties_["Color Camera Name"],
                               manufator)
                    ->disConnect();
            }

            if (projectorFactory_.getProjector(stringProperties_["DLP Evm"])
                    ->isConnect()) {
                projectorFactory_.getProjector(stringProperties_["DLP Evm"])
                    ->disConnect();
            }
        }
    } catch (...) {
        std::cerr << "Connect trinocular camera error! \n" << std::endl;
        return false;
    }

    return connectState;
}

bool TrinocularCamera::disConnect() {
    // TODO@Evans
    // Liu:中途篡改不正确的2D相机制造商将导致漏洞，系统会重新查找该制造商同ID相机
    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    const bool disConnectLeftCamera =
        cameraFactory_
            .getCamera(stringProperties_["Left Camera Name"], manufator)
            ->disConnect();
    const bool disConnectRightCamera =
        cameraFactory_
            .getCamera(stringProperties_["Right Camera Name"], manufator)
            ->disConnect();
    const bool disConnectColorCamera =
        cameraFactory_
            .getCamera(stringProperties_["Color Camera Name"], manufator)
            ->disConnect();
    const bool disConnectProjector =
        projectorFactory_.getProjector(stringProperties_["DLP Evm"])
            ->disConnect();
    return disConnectLeftCamera && disConnectRightCamera &&
           disConnectColorCamera && disConnectProjector;
}

bool TrinocularCamera::isConnect() {
    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    const bool isConnectLeftCamera =
        cameraFactory_
            .getCamera(stringProperties_["Left Camera Name"], manufator)
            ->isConnect();
    const bool isConnectRightCamera =
        cameraFactory_
            .getCamera(stringProperties_["Right Camera Name"], manufator)
            ->isConnect();
    const bool isConnectColorCamera =
        cameraFactory_
            .getCamera(stringProperties_["Color Camera Name"], manufator)
            ->isConnect();
    const bool isConnectProjector =
        projectorFactory_.getProjector(stringProperties_["DLP Evm"])
            ->isConnect();

    return isConnectLeftCamera && isConnectRightCamera &&
           isConnectColorCamera && isConnectProjector;
}

void TrinocularCamera::decode(const std::vector<std::vector<cv::Mat>> &imgs,
                              FrameData &frameData) {
    // 默认深度相机为左相机，纹理相机为彩色相机
    // 顺序依次为左相机、右相机、彩色相机(Camera 1、Camera 2、Camera 3)
    frameData.textureMap_ =
        cv::Mat::zeros(imgs[2][0].size(), imgs[2][0].type());
    for (int i = 0; i < pattern_->params_->shiftTime_; ++i) {
        frameData.textureMap_ += (imgs[2][i] / pattern_->params_->shiftTime_);
    }

    std::vector<std::vector<cv::Mat>> imgsOperat(imgs.begin(), imgs.cend());

    if (frameData.textureMap_.type() == CV_8UC1) {
        cv::cvtColor(frameData.textureMap_, frameData.textureMap_,
                     cv::COLOR_GRAY2BGR);
    } else {
        for (auto img : imgsOperat[2]) {
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }
    }

    cv::Mat depthMap;

    pattern_->decode(imgsOperat, depthMap,
                     booleanProperties_["Gpu Accelerate"]);

    if (booleanProperties_["Noise Filter"]) {
        cv::Mat operateMap = depthMap.clone();
        cv::bilateralFilter(operateMap, depthMap, 15, 20, 50);
    }

    fromDepthMapToCloud(depthMap, frameData.textureMap_, *caliInfo_,
                        *frameData.pointCloud_, frameData.depthMap_, true);
}

bool TrinocularCamera::continuesCapture(SafeQueue<FrameData> &frameDataQueue) {
    if (!isCaptureStop_.load(std::memory_order_acquire)) {
        return true;
    }

    if (imgCreateThread_.joinable()) {
        imgCreateThread_.join();
    }

    if (frameDataCreateThread_.joinable()) {
        frameDataCreateThread_.join();
    }

    isCaptureStop_.store(false, std::memory_order_release);

    imgCreateThread_ = std::thread([&] {
        const device::CameraFactory::CameraManufactor manufator =
            stringProperties_["2D Camera Manufactor"] == "Huaray"
                ? device::CameraFactory::Huaray
                : device::CameraFactory::Halcon;
        auto pLeftCamera = cameraFactory_.getCamera(
            stringProperties_["Left Camera Name"], manufator);
        auto pRightCamera = cameraFactory_.getCamera(
            stringProperties_["Right Camera Name"], manufator);
        auto pColorCamera = cameraFactory_.getCamera(
            stringProperties_["Color Camera Name"], manufator);
        const int imgSizeWaitFor = numbericalProperties_["Total Fringes"];

        while (!isCaptureStop_.load(std::memory_order_acquire)) {
            auto leftImgs = pLeftCamera->getImgs();
            auto rightImgs = pRightCamera->getImgs();
            auto colorImgs = pColorCamera->getImgs();
            if (leftImgs.size() >= imgSizeWaitFor &&
                rightImgs.size() >= imgSizeWaitFor &&
                colorImgs.size() >= imgSizeWaitFor) {
                std::vector<std::vector<cv::Mat>> imgs(3);
                int index = 0;
                while (index != imgSizeWaitFor) {
                    imgs[0].emplace_back(pLeftCamera->popImg());
                    imgs[1].emplace_back(pRightCamera->popImg());
                    imgs[2].emplace_back(pColorCamera->popImg());
                    ++index;
                }

                if (imgsCreated_.size() < 2) {
                    imgsCreated_.push(std::move(imgs));
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    });

    frameDataCreateThread_ = std::thread([&] {
        while (!isCaptureStop_.load(std::memory_order_acquire)) {
            std::vector<std::vector<cv::Mat>> imgs;
            if (imgsCreated_.try_pop(imgs)) {
                for (int i = 0; i < imgs[2].size(); ++i) {
                    cv::cvtColor(imgs[2][i], imgs[2][i], cv::COLOR_BayerBG2BGR);
                }

                FrameData curFrameData;
                decode(imgs, curFrameData);
                frameDataQueue.push(curFrameData);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    });

    projectorFactory_.getProjector(stringProperties_["DLP Evm"])->project(true);

    return true;
}

bool TrinocularCamera::stopContinuesCapture() {
    auto pProjector =
        projectorFactory_.getProjector(stringProperties_["DLP Evm"]);
    pProjector->stop();
    isCaptureStop_.store(true, std::memory_order_release);

    if (imgCreateThread_.joinable()) {
        imgCreateThread_.join();
    }

    if (frameDataCreateThread_.joinable()) {
        frameDataCreateThread_.join();
    }

    // 等待100ms后，将相机所有图片清空
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    SafeQueue<std::vector<std::vector<cv::Mat>>> emptyQueue;
    imgsCreated_.swap(emptyQueue);

    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    cameraFactory_.getCamera(stringProperties_["Left Camera Name"], manufator)
        ->clearImgs();
    cameraFactory_.getCamera(stringProperties_["Right Camera Name"], manufator)
        ->clearImgs();
    cameraFactory_.getCamera(stringProperties_["Color Camera Name"], manufator)
        ->clearImgs();

    return true;
}

bool TrinocularCamera::offlineCapture(
    const std::vector<std::vector<cv::Mat>> &imgs, FrameData &frameData) {
    decode(imgs, frameData);
    return true;
}

bool TrinocularCamera::capture(FrameData &frameData) {
    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    auto pLeftCamera = cameraFactory_.getCamera(
        stringProperties_["Left Camera Name"], manufator);
    auto pRightCamera = cameraFactory_.getCamera(
        stringProperties_["Right Camera Name"], manufator);
    auto pColorCamera = cameraFactory_.getCamera(
        stringProperties_["Color Camera Name"], manufator);
    auto pProjector =
        projectorFactory_.getProjector(stringProperties_["DLP Evm"]);
    const int imgSizeWaitFor = numbericalProperties_["Total Fringes"];
    const int totalExposureTime = (numbericalProperties_["Pre Exposure Time"] +
                                   numbericalProperties_["Exposure Time"] +
                                   numbericalProperties_["Aft Exposure Time"]) *
                                  imgSizeWaitFor;

    pProjector->project(false);

    auto endTime = std::chrono::steady_clock::now() +
                   std::chrono::duration<int, std::ratio<1, 1000000>>(
                       totalExposureTime + 1000000);
    // 注意相机回调函数是主函数运行，因此尽量将该线程设置为多线程，从而不影响相机取图
    while (pLeftCamera->getImgs().size() < imgSizeWaitFor ||
           pRightCamera->getImgs().size() < imgSizeWaitFor ||
           pColorCamera->getImgs().size() < imgSizeWaitFor) {
        if (std::chrono::steady_clock::now() > endTime) {
            pLeftCamera->clearImgs();
            pRightCamera->clearImgs();
            pColorCamera->clearImgs();
            return false;
        }
    }

    std::vector<std::vector<cv::Mat>> imgs(3);
    while (pLeftCamera->getImgs().size()) {
        imgs[0].emplace_back(pLeftCamera->popImg());
        imgs[1].emplace_back(pRightCamera->popImg());
        cv::Mat img = pColorCamera->popImg();
        cv::cvtColor(img, img, cv::COLOR_BayerBG2BGR);
        imgs[2].emplace_back(img);
    }

    decode(imgs, frameData);

    return true;
}

bool TrinocularCamera::setDepthCameraEnabled(const bool isEnable) {
    return setBooleanAttribute("Enable Depth Camera", isEnable);
}

bool TrinocularCamera::getStringAttribute(const std::string attributeName,
                                          std::string &val) {

    if (!stringProperties_.count(attributeName)) {
        std::cerr << "property " << attributeName.data()
                  << " is not be supported !" << std::endl;
        return false;
    }

    val = stringProperties_[attributeName];

    return true;
}

bool TrinocularCamera::getNumbericalAttribute(const std::string attributeName,
                                              double &val) {
    if (!numbericalProperties_.count(attributeName)) {
        std::cerr << "property " << attributeName.data()
                  << " is not be supported !\n"
                  << std::endl;
        return false;
    }
    val = numbericalProperties_[attributeName];

    return true;
}

bool TrinocularCamera::getBooleanAttribute(const std::string attributeName,
                                           bool &val) {
    if (!booleanProperties_.count(attributeName)) {
        std::cerr << "property " << attributeName.data()
                  << " is not be supported !\n"
                  << std::endl;
        return false;
    }

    val = booleanProperties_[attributeName];

    return true;
}

bool TrinocularCamera::setStringAttribute(const std::string attributeName,
                                          const std::string val) {
    stringProperties_[attributeName] = val;

    if (isInitial_) {
        propertiesChangedSignals_[attributeName] = true;
        parseSignals();
    }

    return true;
}

bool TrinocularCamera::setNumberAttribute(const std::string attributeName,
                                          const double val) {

    numbericalProperties_[attributeName] = val;

    if (isInitial_) {
        propertiesChangedSignals_[attributeName] = true;
        parseSignals();
    }

    return true;
}

bool TrinocularCamera::setBooleanAttribute(const std::string attributeName,
                                           const bool val) {
    booleanProperties_[attributeName] = val;

    if (isInitial_) {
        propertiesChangedSignals_[attributeName] = true;
        parseSignals();
    }

    return true;
}

bool TrinocularCamera::resetCameraConfig() {
    stringProperties_["Camera Name"] = "Trinocular Camera";
    stringProperties_["Manufactor"] = "@Evans Liu";
    stringProperties_["Email"] = "@1369215984@qq.com";
    stringProperties_["Accuracy"] = "0.2mm @1m";
    stringProperties_["DLP Height"] = "1080";
    stringProperties_["DLP Width"] = "1920";
    stringProperties_["True Width"] = "600mm";
    stringProperties_["True Height"] = "500mm";
    stringProperties_["DLP Evm"] = "DLP4710";
    stringProperties_["Color Camera Name"] = "Color";
    stringProperties_["Left Camera Name"] = "Left";
    stringProperties_["Right Camera Name"] = "Right";
    stringProperties_["2D Camera Manufactor"] = "Huaray";
    stringProperties_["Calibration File Path"] = "../../data/caliInfo.yml";
    stringProperties_["Intrinsic Path"] = "../../data/caliInfo.yml";

    numbericalProperties_["Contrast Threshold"] = 5;
    numbericalProperties_["Cost Min Diff"] = 0.001;
    numbericalProperties_["Cost Max Diff"] = 1.0;
    numbericalProperties_["Max Cost"] = 0.1;
    numbericalProperties_["Light Strength"] = 0.9;
    numbericalProperties_["Exposure Time"] = 20000;
    numbericalProperties_["Pre Exposure Time"] = 5000;
    numbericalProperties_["Aft Exposure Time"] = 5000;
    numbericalProperties_["Phase Shift Times"] = 3;
    numbericalProperties_["Cycles"] = 32;
    numbericalProperties_["Total Fringes"] = 3;
    numbericalProperties_["Pattern"] = 2.0;
    numbericalProperties_["Minimum Depth"] = 0.0;
    numbericalProperties_["Maximum Depth"] = 1500;

    booleanProperties_["Is One Bit"] = true;
    booleanProperties_["Enable Depth Camera"] = true;
    booleanProperties_["Noise Filter"] = true;
    booleanProperties_["Gpu Accelerate"] = true;
    booleanProperties_["Is Vertical"] = true;

    saveParams(jsonPath_, jsonVal_);
    loadParams(jsonPath_, jsonVal_);

    return updateCamera();
}

bool TrinocularCamera::burnPatterns(const std::vector<cv::Mat> &imgs) {
    // TODO@Evans Liu:
    // 目前只支持烧录单方向条纹，双方向需要使用更底层的投影仪SDK，示例CameraEngine.cpp中的burnStripe函数
    auto pProjector =
        projectorFactory_.getProjector(stringProperties_["DLP Evm"]);

    auto projectorInfo = pProjector->getInfo();
    cv::Size imgSize = cv::Size(projectorInfo.width_, projectorInfo.height_);

    const int numOfPatternOrderSets = std::ceil(imgs.size() / 6.f);
    std::vector<device::PatternOrderSet> patternSets(
        numOfPatternOrderSets);
    for (size_t i = 0; i < patternSets.size(); ++i) {
        patternSets[i].exposureTime_ = numbericalProperties_["Exposure Time"];
        patternSets[i].preExposureTime_ =
            numbericalProperties_["Pre Exposure Time"];
        patternSets[i].postExposureTime_ =
            numbericalProperties_["Aft Exposure Time"];
        patternSets[i].illumination_ = device::Blue;
        patternSets[i].isOneBit_ = booleanProperties_["Is One Bit"];
        patternSets[i].isVertical_ = booleanProperties_["Is Vertical"];
        patternSets[i].patternArrayCounts_ =
            booleanProperties_["Is Vertical"] ? imgSize.width : imgSize.height;
        patternSets[i].invertPatterns_ = false;
        patternSets[i].imgs_ =
            6 * i + 6 < imgs.size()
                ? std::vector<cv::Mat>(imgs.begin() + 6 * i,
                                       imgs.begin() + 6 * (i + 1))
                : std::vector<cv::Mat>(imgs.begin() + 6 * i, imgs.end());
    }

    bool isSuccess = pProjector->populatePatternTableData(patternSets);

    if (isSuccess) {
        updateExposureTime();
    }

    return isSuccess;
}

void TrinocularCamera::updateExposureTime() {
    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    cameraFactory_.getCamera(stringProperties_["Color Camera Name"], manufator)
        ->setNumberAttribute("ExposureTime",
                             numbericalProperties_["Exposure Time"]);

    cameraFactory_.getCamera(stringProperties_["Left Camera Name"], manufator)
        ->setNumberAttribute("ExposureTime",
                             numbericalProperties_["Exposure Time"]);
    cameraFactory_.getCamera(stringProperties_["Right Camera Name"], manufator)
        ->setNumberAttribute("ExposureTime",
                             numbericalProperties_["Exposure Time"]);
}

void TrinocularCamera::updateEnableDepthCamera() {
    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    cameraFactory_.getCamera(stringProperties_["Color Camera Name"], manufator)
        ->setTrigMode(device::trigLine);
    cameraFactory_.getCamera(stringProperties_["Left Camera Name"], manufator)
        ->setTrigMode(device::trigLine);
    cameraFactory_.getCamera(stringProperties_["Right Camera Name"], manufator)
        ->setTrigMode(device::trigLine);
}

void TrinocularCamera::updateLightStrength() {
    projectorFactory_.getProjector(stringProperties_["DLP Evm"])
        ->setLEDCurrent(numbericalProperties_["Light Strength"],
                        numbericalProperties_["Light Strength"],
                        numbericalProperties_["Light Strength"]);
}

void TrinocularCamera::parseSignals() {
    if (propertiesChangedSignals_["Enable Depth Camera"]) {
        if (booleanProperties_["Enable Depth Camera"]) {
            updateEnableDepthCamera();
            propertiesChangedSignals_["Enable Depth Camera"] = false;
        }
    }

    if (propertiesChangedSignals_["Light Strength"]) {
        updateLightStrength();
        propertiesChangedSignals_["Light Strength"] = false;
    }
}

bool TrinocularCamera::updateCamera() {
    saveParams(jsonPath_, jsonVal_);

    return true;
}
} // namespace cameras

} // namespace slmaster
