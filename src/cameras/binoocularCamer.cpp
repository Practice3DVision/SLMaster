#include "binoocularCamera.h"

#include "binosSinusCompleGrayCodePattern.h"
#include "tool.h"

namespace slmaster {
namespace cameras {
BinocularCamera::BinocularCamera(const std::string jsonPath)
    : SLCamera(jsonPath), jsonPath_(jsonPath), isInitial_(false),
      isCaptureStop_(true) {

    if (loadParams(jsonPath, jsonVal_)) {
        if (booleanProperties_["Gpu Accelerate"]) {
#ifndef OPENCV_WITH_CUDA_MODULE
            std::clog << "lib isn't build with cuda, we will disable it!"
                      << std::endl;
            booleanProperties_["Gpu Accelerate"] = false;
#endif //! OPENCV_WITH_CUDA_MODULE
        }

        isInitial_ = true;
    }
}

BinocularCamera::~BinocularCamera() { updateCamera(); }

bool BinocularCamera::loadParams(const std::string jsonPath,
                                 Json::Value &jsonVal) {
    if (!readJsonFile(jsonPath, jsonVal)) {
        std::cerr << "binocular camera parse json file error!" << std::endl;
        return false;
    }

    parseArray(jsonVal["camera"]["device"], false);
    parseArray(jsonVal["camera"]["algorithm"], false);

    if (caliInfo_) {
        delete caliInfo_;
        caliInfo_ = nullptr;
    }

    caliInfo_ = new CaliInfo(stringProperties_["Calibration File Path"]);

    cv::Size imgSize(caliInfo_->info_.S_.at<double>(0, 0),
                     caliInfo_->info_.S_.at<double>(1, 0));
    cv::initUndistortRectifyMap(caliInfo_->info_.M1_, caliInfo_->info_.D1_,
                                caliInfo_->info_.R1_, caliInfo_->info_.P1_,
                                imgSize, CV_32FC1, mapLX_, mapLY_);
    cv::initUndistortRectifyMap(caliInfo_->info_.M2_, caliInfo_->info_.D2_,
                                caliInfo_->info_.R2_, caliInfo_->info_.P2_,
                                imgSize, CV_32FC1, mapRX_, mapRY_);

    return true;
}

bool BinocularCamera::saveParams(const std::string jsonPath,
                                 Json::Value &jsonVal) {
    if (jsonVal.empty()) {
        std::cerr
            << "binocular camera write json file error, json value is empty! \n"
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

void BinocularCamera::parseArray(Json::Value &jsonVal, const bool isWrite) {
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

SLCameraInfo BinocularCamera::getCameraInfo() {
    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    auto pLeftCamera = cameraFactory_.getCamera(
        stringProperties_["Left Camera Name"], manufator);
    auto pRightCamera = cameraFactory_.getCamera(
        stringProperties_["Right Camera Name"], manufator);
    auto leftCameraInfo = pLeftCamera->getCameraInfo();
    auto rightCameraInfo = pRightCamera->getCameraInfo();

    auto pProjector =
        projectorFactory_.getProjector(stringProperties_["DLP Evm"]);

    device::ProjectorInfo projectorInfo =
        pProjector ? pProjector->getInfo() : device::ProjectorInfo();
    if (!pProjector) {
        projectorInfo.isFind_ = false;
    }

    SLCameraInfo slCameraInfo;
    slCameraInfo.isFind_ = leftCameraInfo.isFind_ && rightCameraInfo.isFind_ &&
                           projectorInfo.isFind_;
    slCameraInfo.cameraName_ =
        slCameraInfo.isFind_ ? stringProperties_["Camera Name"] : "NOT_FOUND";
    slCameraInfo.intrinsic_ = slCameraInfo.isFind_
                                  ? caliInfo_->info_.M1_
                                  : cv::Mat::zeros(3, 3, CV_32FC1);

    return slCameraInfo;
}

bool BinocularCamera::connect() {
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
        const bool connectProjector =
            projectorFactory_.getProjector(stringProperties_["DLP Evm"])
                ->connect();

        bool hasColorCamera = (stringProperties_["Color Camera Name"] != "");
        bool connectColorCamera = true;
        if (hasColorCamera) {
            connectColorCamera =
                cameraFactory_
                    .getCamera(stringProperties_["Color Camera Name"],
                               manufator)
                    ->connect();
        }

        connectState = connectLeftCamera && connectRightCamera &&
                       connectProjector && connectColorCamera;

        if (connectState) {
            updateEnableDepthCamera();
            updateExposureTime();
            updateLightStrength();

            cameraFactory_
                .getCamera(stringProperties_["Left Camera Name"], manufator)
                ->start();
            cameraFactory_
                .getCamera(stringProperties_["Right Camera Name"], manufator)
                ->start();

            if (hasColorCamera) {
                cameraFactory_
                    .getCamera(stringProperties_["Color Camera Name"],
                               manufator)
                    ->start();
            }
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

            if (hasColorCamera &&
                cameraFactory_
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
        std::cerr << "Connect binocular camera error! \n" << std::endl;
        return false;
    }

    return connectState;
}

bool BinocularCamera::disConnect() {
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
    const bool disConnectProjector =
        projectorFactory_.getProjector(stringProperties_["DLP Evm"])
            ->disConnect();
    bool disConnectcolorCamera = true;
    if (stringProperties_["Color Camera Name"] == "") {
        disConnectcolorCamera =
            cameraFactory_
                .getCamera(stringProperties_["Color Camera Name"], manufator)
                ->disConnect();
    }

    return disConnectLeftCamera && disConnectRightCamera &&
           disConnectProjector && disConnectcolorCamera;
}

bool BinocularCamera::isConnect() {
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
    const bool isConnectProjector =
        projectorFactory_.getProjector(stringProperties_["DLP Evm"])
            ->isConnect();
    bool isConnectColorCamera = true;
    if (stringProperties_["Color Camera Name"] == "") {
        isConnectColorCamera =
            cameraFactory_
                .getCamera(stringProperties_["Color Camera Name"], manufator)
                ->isConnect();
    }

    return isConnectLeftCamera && isConnectRightCamera && isConnectProjector &&
           isConnectColorCamera;
}

void BinocularCamera::decode(const std::vector<std::vector<cv::Mat>> &imgs,
                             FrameData &frameData) {
    // 默认深度相机为左相机，纹理相机为左相机
    const int index = imgs.size() == 3 ? 2 : 0;
    frameData.textureMap_ =
        cv::Mat::zeros(imgs[index][0].size(), imgs[index][0].type());
    for (int i = 0; i < pattern_->params_->shiftTime_; ++i) {
        frameData.textureMap_ +=
            (imgs[index][i] / pattern_->params_->shiftTime_);
    }
    if (frameData.textureMap_.type() == CV_8UC1) {
        cv::cvtColor(frameData.textureMap_, frameData.textureMap_,
                     cv::COLOR_GRAY2BGR);
    }

    cv::Mat disparityMap, textureMapped;

    std::vector<std::vector<cv::Mat>> remapedImgs(
        2, std::vector<cv::Mat>(imgs[0].size()));
    cv::parallel_for_(cv::Range(0, imgs[0].size()),
                      [&](const cv::Range &range) {
                          for (int i = range.start; i < range.end; ++i) {
                              cv::remap(imgs[0][i], remapedImgs[0][i], mapLX_,
                                        mapLY_, cv::INTER_LINEAR);
                              cv::remap(imgs[1][i], remapedImgs[1][i], mapRX_,
                                        mapRY_, cv::INTER_LINEAR);
                          }
                          if (range.start == 0 && index == 0) {
                              cv::remap(frameData.textureMap_, textureMapped,
                                        mapLX_, mapLY_, cv::INTER_LINEAR);
                          }
                      });

    pattern_->decode(remapedImgs, disparityMap,
                     booleanProperties_["Gpu Accelerate"]);

    if (booleanProperties_["Noise Filter"]) {
        cv::Mat operateMap = disparityMap.clone();
        cv::bilateralFilter(operateMap, disparityMap, 15, 20, 50);
    }

    fromDispairtyMapToCloud(disparityMap, textureMapped, *caliInfo_,
                            *frameData.pointCloud_, frameData.depthMap_,
                            index == 2);
}

bool BinocularCamera::offlineCapture(
    const std::vector<std::vector<cv::Mat>> &imgs, FrameData &frameData) {

    decode(imgs, frameData);
    return true;
}

bool BinocularCamera::continuesCapture(SafeQueue<FrameData> &frameDataQueue) {
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

        device::Camera *pColorCamera = nullptr;
        if (stringProperties_["Color Camera Name"] != "") {
            pColorCamera = cameraFactory_.getCamera(
                stringProperties_["Color Camera Name"], manufator);
        }

        const int imgSizeWaitFor = numbericalProperties_["Total Fringes"];

        while (!isCaptureStop_.load(std::memory_order_acquire)) {
            if (pLeftCamera->getImgs().size() >= imgSizeWaitFor &&
                pRightCamera->getImgs().size() >= imgSizeWaitFor &&
                (pColorCamera ? pColorCamera->getImgs().size() >= imgSizeWaitFor
                              : true)) {
                std::vector<std::vector<cv::Mat>> imgs(pColorCamera ? 3 : 2);
                int index = 0;
                while (index != imgSizeWaitFor) {
                    imgs[0].emplace_back(pLeftCamera->popImg());
                    imgs[1].emplace_back(pRightCamera->popImg());
                    if (pColorCamera) {
                        imgs[2].emplace_back(pColorCamera->popImg());
                    }
                    ++index;
                }

                if (imgsCreated_.size() > 2) {
                    continue;
                }

                imgsCreated_.push(imgs);
            }
        }
    });

    frameDataCreateThread_ = std::thread([&] {
        while (!isCaptureStop_.load(std::memory_order_acquire)) {
            if (imgsCreated_.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            std::vector<std::vector<cv::Mat>> imgs;
            imgsCreated_.move_pop(imgs);

            if (stringProperties_["Color Camera Name"] != "") {
                for (int i = 0; i < imgs.size(); ++i) {
                    cv::cvtColor(imgs[i], imgs[i], cv::COLOR_BayerBG2BGR);
                }
            }

            FrameData curFrameData;
            decode(imgs, curFrameData);
            frameDataQueue.push(curFrameData);
        }
    });

    projectorFactory_.getProjector(stringProperties_["DLP Evm"])->project(true);

    return true;
}

bool BinocularCamera::stopContinuesCapture() {
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
    if (stringProperties_["Color Camera Name"] != "") {
        cameraFactory_
            .getCamera(stringProperties_["Color Camera Name"], manufator)
            ->clearImgs();
    }

    return true;
}

bool BinocularCamera::capture(FrameData &frameData) {
    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    auto pLeftCamera = cameraFactory_.getCamera(
        stringProperties_["Left Camera Name"], manufator);
    auto pRightCamera = cameraFactory_.getCamera(
        stringProperties_["Right Camera Name"], manufator);

    device::Camera *pColorCamera = nullptr;
    if (stringProperties_["Color Camera Name"] != "") {
        pColorCamera = cameraFactory_.getCamera(
            stringProperties_["Color Camera Name"], manufator);
    }

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
           (pColorCamera ? (pColorCamera->getImgs().size() < imgSizeWaitFor)
                         : false)) {
        if (std::chrono::steady_clock::now() > endTime) {
            pLeftCamera->clearImgs();
            pRightCamera->clearImgs();

            if (pColorCamera) {
                pColorCamera->clearImgs();
            }

            return false;
        }
    }

    std::vector<std::vector<cv::Mat>> imgs(pColorCamera ? 3 : 2);
    while (pLeftCamera->getImgs().size()) {
        imgs[0].emplace_back(pLeftCamera->popImg());
        imgs[1].emplace_back(pRightCamera->popImg());
        if (pColorCamera) {
            imgs[2].emplace_back(pColorCamera->popImg());
        }
    }

    decode(imgs, frameData);

    return true;
}

bool BinocularCamera::setDepthCameraEnabled(const bool isEnable) {
    return setBooleanAttribute("Enable Depth Camera", isEnable);
}

bool BinocularCamera::getStringAttribute(const std::string attributeName,
                                         std::string &val) {

    if (!stringProperties_.count(attributeName)) {
        std::cerr << "property " << attributeName.data()
                  << " is not be supported !" << std::endl;
        return false;
    }

    val = stringProperties_[attributeName];

    return true;
}

bool BinocularCamera::getNumbericalAttribute(const std::string attributeName,
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

bool BinocularCamera::getBooleanAttribute(const std::string attributeName,
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

bool BinocularCamera::setStringAttribute(const std::string attributeName,
                                         const std::string val) {
    stringProperties_[attributeName] = val;

    if (isInitial_) {
        propertiesChangedSignals_[attributeName] = true;
        parseSignals();
    }

    return true;
}

bool BinocularCamera::setNumberAttribute(const std::string attributeName,
                                         const double val) {

    numbericalProperties_[attributeName] = val;

    if (isInitial_) {
        propertiesChangedSignals_[attributeName] = true;
        parseSignals();
    }

    return true;
}

bool BinocularCamera::setBooleanAttribute(const std::string attributeName,
                                          const bool val) {
    booleanProperties_[attributeName] = val;

    if (isInitial_) {
        propertiesChangedSignals_[attributeName] = true;
        parseSignals();
    }

    return true;
}

bool BinocularCamera::resetCameraConfig() {
    stringProperties_["Camera Name"] = "Binocular Camera";
    stringProperties_["Manufactor"] = "@Evans Liu";
    stringProperties_["Email"] = "@1369215984@qq.com";
    stringProperties_["Accuracy"] = "0.2mm @1m";
    stringProperties_["DLP Height"] = "1080";
    stringProperties_["DLP Width"] = "1920";
    stringProperties_["True Width"] = "600mm";
    stringProperties_["True Height"] = "500mm";
    stringProperties_["DLP Evm"] = "DLP4710";
    stringProperties_["Left Camera Name"] = "Left";
    stringProperties_["Right Camera Name"] = "Right";
    stringProperties_["2D Camera Manufactor"] = "Huaray";
    stringProperties_["Intrinsic Path"] = "../../data/caliInfo.yml";

    numbericalProperties_["Contrast Threshold"] = 5;
    numbericalProperties_["Cost Min Diff"] = 0.001;
    numbericalProperties_["Max Cost"] = 0.1;
    numbericalProperties_["Min Disparity"] = -300;
    numbericalProperties_["Max Disparity"] = 300;
    numbericalProperties_["Light Strength"] = 0.9;
    numbericalProperties_["Exposure Time"] = 20000;
    numbericalProperties_["Pre Exposure Time"] = 5000;
    numbericalProperties_["Aft Exposure Time"] = 5000;
    numbericalProperties_["Phase Shift Times"] = 12;
    numbericalProperties_["Cycles"] = 64;
    numbericalProperties_["Total Fringes"] = 19;
    numbericalProperties_["Pattern"] = 0.0;
    numbericalProperties_["Minimum Depth"] = 0.0;
    numbericalProperties_["Maximum Depth"] = 1500;

    booleanProperties_["Is One Bit"] = false;
    booleanProperties_["Enable Depth Camera"] = true;
    booleanProperties_["Noise Filter"] = true;
    booleanProperties_["Gpu Accelerate"] = false;
    booleanProperties_["Is Vertical"] = true;

    saveParams(jsonPath_, jsonVal_);
    loadParams(jsonPath_, jsonVal_);

    return updateCamera();
}

bool BinocularCamera::burnPatterns(const std::vector<cv::Mat> &imgs) {
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

void BinocularCamera::updateExposureTime() {
    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    bool isSucess =
        cameraFactory_
            .getCamera(stringProperties_["Left Camera Name"], manufator)
            ->setNumberAttribute("ExposureTime",
                                 numbericalProperties_["Exposure Time"]);
    isSucess = cameraFactory_
                   .getCamera(stringProperties_["Right Camera Name"], manufator)
                   ->setNumberAttribute("ExposureTime",
                                        numbericalProperties_["Exposure Time"]);
    if (stringProperties_["Color Camera Name"] != "") {
        cameraFactory_
            .getCamera(stringProperties_["Color Camera Name"], manufator)
            ->setNumberAttribute("ExposureTime",
                                 numbericalProperties_["Exposure Time"]);
    }
}

void BinocularCamera::updateEnableDepthCamera() {
    const device::CameraFactory::CameraManufactor manufator =
        stringProperties_["2D Camera Manufactor"] == "Huaray"
            ? device::CameraFactory::Huaray
            : device::CameraFactory::Halcon;
    cameraFactory_.getCamera(stringProperties_["Left Camera Name"], manufator)
        ->setTrigMode(device::trigLine);
    cameraFactory_.getCamera(stringProperties_["Right Camera Name"], manufator)
        ->setTrigMode(device::trigLine);
    if (stringProperties_["Color Camera Name"] != "") {
        cameraFactory_
            .getCamera(stringProperties_["Color Camera Name"], manufator)
            ->setTrigMode(device::trigLine);
    }
}

void BinocularCamera::updateLightStrength() {
    projectorFactory_.getProjector(stringProperties_["DLP Evm"])
        ->setLEDCurrent(numbericalProperties_["Light Strength"],
                        numbericalProperties_["Light Strength"],
                        numbericalProperties_["Light Strength"]);
}

void BinocularCamera::parseSignals() {
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

bool BinocularCamera::updateCamera() {
    saveParams(jsonPath_, jsonVal_);

    return true;
}
} // namespace cameras
} // namespace slmaster
