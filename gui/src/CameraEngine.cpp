#include "CameraEngine.h"
#include "VtkProcessEngine.h"

#include <QDebug>
#include <QString>

#include <binosSinusCompleGrayCodePattern.h>
#include <defocusMethod.hpp>
#ifdef WITH_CUDASTRUCTUREDLIGHT_MODULE
#include <opencv2/cudastructuredlight.hpp>
#endif

#include <pcl/io/pcd_io.h>

using namespace slmaster;
using namespace std;

CameraEngine* CameraEngine::engine_ = new CameraEngine();

CameraEngine* CameraEngine::instance() {
    return engine_;
}

static auto lastTime = std::chrono::steady_clock::now();

CameraEngine::CameraEngine() : stripePaintItem_(nullptr), isOnLine_(false), isConnected_(false), appExit_(false), cameraType_(AppType::BinocularSLCamera), isProject_(false), isContinusStop_(true) { }

void CameraEngine::startDetectCameraState() {
    onlineDetectThread_ = std::thread([&] {
        while(!appExit_.load(std::memory_order_acquire)) {
            //TODO@LiuYunhuang: 在相机捕图时，不能去检测相机在线状态等相机操作，否则，将导致丢帧现象（已解决）
            if(isContinusStop_.load(std::memory_order_acquire) && !isProject_.load(std::memory_order_acquire)) {
                auto camera = slCameraFactory_.getCamera(slmaster::CameraType(cameraType_));
                if (camera) {
                    if(isOnLine_ ^ camera->getCameraInfo().isFind_) {
                        isOnLine_ = camera->getCameraInfo().isFind_;
                        isOnLineChanged();
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    });
}

CameraEngine::~CameraEngine() {
    appExit_.store(true, std::memory_order_release);

    if(onlineDetectThread_.joinable()) {
        onlineDetectThread_.join();
    }

    if(workThread_.joinable()) {
        workThread_.join();
    }
}

int CameraEngine::createStripe(const int pixelDepth, const int direction, const int stripeType, const int defocusMethod, const int imgWidth, const int imgHeight, const int cycles, const int shiftTime, const bool isKeepAdd) {
    qInfo() << "start create stripe...";

    unique_ptr<Pattern> pattern;
    if(stripeType == AppType::StripeType::SineComplementaryGrayCode) {
        pattern = make_unique<BinoSinusCompleGrayCodePattern>();
    }

    pattern->params_->height_ = imgHeight;
    pattern->params_->width_ = imgWidth;
    pattern->params_->cycles_ = cycles;
    pattern->params_->horizontal_ = direction == AppType::Direction::Horizion;
    pattern->params_->shiftTime_ = shiftTime;

    std::vector<cv::Mat> imgs;
    pattern->generate(imgs);

    if(defocusMethod != AppType::DefocusEncoding::Disable) {
        defocusStripeCreate(imgs, direction, cycles, shiftTime, AppType::DefocusEncoding(defocusMethod));
    }

    auto formatType = AppType::PixelDepth(pixelDepth) == AppType::PixelDepth::OneBit ? QImage::Format_Mono : QImage::Format_Grayscale8;

    std::vector<QImage> tempStripes(imgs.size(), QImage(imgWidth, imgHeight, formatType));
    cv::parallel_for_(cv::Range(0, imgs.size()), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            for (int j = 0; j < imgHeight; ++j) {
                auto imgPtr = imgs[i].ptr(j);
                for (int k = 0; k < imgWidth; ++k) {
                    formatType == QImage::Format_Mono ? tempStripes[i].setPixel(k, j, imgPtr[k]) : tempStripes[i].setPixel(k, j, qRgb(imgPtr[k], imgPtr[k], imgPtr[k]));
                }
            }
        }
    });

    if(!isKeepAdd) {
        stripeImgs_.clear();
        orderTableRecord_.clear();
    }

    stripeImgs_.insert(stripeImgs_.end(), tempStripes.begin(), tempStripes.end());
    orderTableRecord_.emplace_back(OrderTableRecord(tempStripes.size(), shiftTime, direction == AppType::Direction::Vertical));

    qInfo() << "create stripe sucess!";

    emit stripeImgsChanged(stripeImgs_.size());
    return stripeImgs_.size();
}

void CameraEngine::defocusStripeCreate(std::vector<cv::Mat>& imgs, const int direction, const int cycles, const int shiftTime, AppType::DefocusEncoding method) {
    Q_ASSERT(!imgs.empty());

    //TODO@LiuYunhuang: 使用浮点数相移图案会更精确点
    for (int i = 0; i < imgs.size(); ++i) {
        if(i < shiftTime) {
            if(method == AppType::DefocusEncoding::ErrorDiffusionMethod) {
                twoDimensionErrorExpand(imgs[i]);
            }
            else if(method == AppType::DefocusEncoding::Binary) {
                binary(imgs[i]);
            }
            else if(method == AppType::DefocusEncoding::OptimalPlusWithModulation) {
                const float shiftVal = static_cast<float>(i) / shiftTime * CV_2PI;
                opwm(imgs[i], cycles, shiftVal, direction == AppType::Direction::Horizion);
            }
        }
        else {
            binary(imgs[i]);
        }
    }
}

void CameraEngine::createTenLine() {
    auto pProjector = getSLCamera()->getProjectorFactory()->getProjector(getStringAttribute("DLP Evm").toStdString());
    const int width = getStringAttribute("DLP Width").toInt();
    const int height = getStringAttribute("DLP Height").toInt();

    std::vector<cv::Mat> verticalLine(1, cv::Mat::zeros(height, width, CV_8UC1));
    std::vector<cv::Mat> honrizonLine(1, cv::Mat::zeros(height, width, CV_8UC1));

    verticalLine[0](cv::Rect(width / 2 - 1, 0, 2, height)) = 255 - cv::Mat::zeros(height, 2, CV_8UC1);
    honrizonLine[0](cv::Rect(0, height / 2 - 1, width, 2)) = 255 - cv::Mat::zeros(2, width, CV_8UC1);

    std::vector<device::projector::PatternOrderSet> patternSets(2);
    for (size_t i = 0; i < patternSets.size(); ++i) {
        patternSets[i].exposureTime_ = getNumberAttribute("Exposure Time");
        patternSets[i].preExposureTime_ = getNumberAttribute("Pre Exposure Time");
        patternSets[i].postExposureTime_ = getNumberAttribute("Aft Exposure Time");
        patternSets[i].illumination_ = device::projector::Blue;
        patternSets[i].isOneBit_ = getBooleanAttribute("Is One Bit");
        patternSets[i].isVertical_ = i == 0 ? true : false;
        patternSets[i].patternArrayCounts_ = i == 0 ? width : height;
        patternSets[i].invertPatterns_ = false;
        patternSets[i].imgs_ = i == 0 ? verticalLine : honrizonLine;
    }

    switchTrigMode(false, getNumberAttribute("Exposure Time"));

    bool isSuccess = pProjector->populatePatternTableData(patternSets);

    switchTrigMode(true, (getNumberAttribute("Exposure Time") + getNumberAttribute("Pre Exposure Time") + getNumberAttribute("Aft Exposure Time")) * 2 - 1000);

    isBurnWorkFinish_ = true;
    emit isBurnWorkFinishChanged();
}

void CameraEngine::switchTrigMode(const bool isTrigLine, const int exposureTime) {
    const device::camera::CameraFactory::CameraManufactor manufator =
        getStringAttribute("2D Camera Manufactor") == "Huaray"
            ? device::camera::CameraFactory::Huaray
            : device::camera::CameraFactory::Halcon;
    auto leftCamera = getSLCamera()->getCameraFactory()->getCamera(getStringAttribute("Left Camera Name").toStdString(), manufator);
    std::string rightCameraName, colorCameraName;
    auto rightCamera = getSLCamera()->getStringAttribute("Right Camera Name", rightCameraName) ? getSLCamera()->getCameraFactory()->getCamera(rightCameraName, manufator) : nullptr;
    auto colorCamera = getSLCamera()->getStringAttribute("Color Camera Name", colorCameraName) ? getSLCamera()->getCameraFactory()->getCamera(colorCameraName, manufator) : nullptr;

    auto trigMode = isTrigLine ? device::camera::trigLine : device::camera::trigSoftware;

    leftCamera->setTrigMode(trigMode);
    leftCamera->setNumberAttribute("ExposureTime", exposureTime);
    leftCamera->clearImgs();
    if(rightCamera) {
        rightCamera->setTrigMode(trigMode);
        rightCamera->setNumberAttribute("ExposureTime", exposureTime);
        rightCamera->clearImgs();
    }
    if(colorCamera) {
        colorCamera->setTrigMode(trigMode);
        colorCamera->setNumberAttribute("ExposureTime", exposureTime);
        colorCamera->clearImgs();
    }
}

void CameraEngine::displayStripe(const int stripeIndex) {
    qInfo() << QString("start display stripe: %1").arg(stripeIndex);
    QImage img = stripeImgs_[stripeIndex - 1];
    stripePaintItem_->updateImage(img);
}

void CameraEngine::saveStripe(const QString& path) {
    qInfo() << QString("the path to save stripe is : %1").arg(path.mid(8));

    if(stripeImgs_.empty()) {
        return;
    }

    for (int i = 0; i < stripeImgs_.size(); ++i) {
        stripeImgs_[i].save(path.mid(8) + "/IMG_" + QString::number(i) + ".bmp", "bmp", 100);
    }

    qInfo() << QString("save stripe sucess!");
}

void CameraEngine::selectCamera(const int cameraType) {
    qInfo() << QString("select camera: %1.").arg(cameraType);

    cameraType_ = AppType::CameraType(cameraType);
}

void CameraEngine::setCameraJsonPath(const std::string jsonPath) {
    qInfo() << QString("set camera config json path...");
    slCameraFactory_.setCameraJsonPath(jsonPath);
    getSLCamera();
}

bool CameraEngine::connectCamera() {
    qInfo() << QString("connect camera...");

    auto camera = slCameraFactory_.getCamera(slmaster::CameraType(cameraType_));
    bool isSuccess = false;
    if (camera) {
        if(camera->getCameraInfo().isFind_) {
            isSuccess = camera->connect();
        }
    }
    if (isSuccess) {
        qDebug() << QString("connect camera sucess!");
        isConnected_ = true;
        isConnectedChanged();
    }
    else {
        qDebug() << QString("connect camera failed!");
    }

    //setPatternType(0);
    //continuesScan();

    return isSuccess;
}

bool CameraEngine::disConnectCamera() {
    qInfo() << QString("disconnect camera...");

    auto camera = slCameraFactory_.getCamera(slmaster::CameraType(cameraType_));
    bool isSuccess = false;
    if (camera) {
        isSuccess = camera->disConnect();
    }
    if(isSuccess){
        qDebug() << QString("disconnect camera sucessed!");
        isConnected_ = false;
        isConnectedChanged();
    }
    else {
        qDebug() << QString("disconnect camera failed!");
    }

    return isSuccess;
}

void CameraEngine::burnStripe() {
    qInfo() << QString("burn stripe...");
    if(stripeImgs_.empty()) {
        qDebug() << QString("stripe is empty! cancel operation!");
        isBurnWorkFinish_ = true;
        isBurnWorkFinishChanged();
        return;
    }

    qDebug() << QString("stripe img widht: %1").arg(stripeImgs_[0].width());
    qDebug() << QString("stripe img height: %1").arg(stripeImgs_[0].height());
    qDebug() << QString("stripe img depth: %1").arg(stripeImgs_[0].depth());

    if(workThread_.joinable()) {
        workThread_.join();
    }

    workThread_ = std::thread([&]{
        auto pProjector = getSLCamera()->getProjectorFactory()->getProjector(getStringAttribute("DLP Evm").toStdString());
        const int width = getStringAttribute("DLP Width").toInt();
        const int height = getStringAttribute("DLP Height").toInt();

        std::vector<cv::Mat> imgs;
        for (auto img : stripeImgs_) {
            cv::Mat convertImg(img.height(), img.width(), CV_8UC1);
            for (int i = 0; i < img.height(); ++i) {
                auto ptrConvertImg = convertImg.ptr(i);
                for (int j = 0; j < img.width(); ++j) {
                    ptrConvertImg[j] = qGray(img.pixel(j, i));
                }
            }
            imgs.emplace_back(convertImg);
        }

        std::vector<device::projector::PatternOrderSet> sumPatternSets;
        int indexImg = 0;
        for (int i = 0; i < orderTableRecord_.size(); ++i) {
            const int numOfPatternOrderSets = std::ceil(orderTableRecord_[i].patternsNum_ / 6.f);
            std::vector<device::projector::PatternOrderSet> patternSets(numOfPatternOrderSets);
            for (int j = 0; j < patternSets.size(); ++j) {
                patternSets[j].exposureTime_ = getNumberAttribute("Exposure Time");
                patternSets[j].preExposureTime_ = getNumberAttribute("Pre Exposure Time");
                patternSets[j].postExposureTime_ = getNumberAttribute("Aft Exposure Time");
                patternSets[j].illumination_ = device::projector::Blue;
                patternSets[j].isOneBit_ = getBooleanAttribute("Is One Bit");
                patternSets[j].isVertical_ = orderTableRecord_[i].isVertical_;
                patternSets[j].patternArrayCounts_ = patternSets[j].isVertical_ ? imgs[0].cols : imgs[0].rows;
                patternSets[j].invertPatterns_ = false;
                patternSets[j].imgs_ = 6 * j + 6 < orderTableRecord_[i].patternsNum_ ? std::vector<cv::Mat>(imgs.begin() + indexImg + 6 * j, imgs.begin() + indexImg + 6 * (j + 1)) : std::vector<cv::Mat>(imgs.begin() + indexImg + 6 * j, imgs.begin() + indexImg + orderTableRecord_[i].patternsNum_);
            }

            indexImg += orderTableRecord_[i].patternsNum_;
            sumPatternSets.insert(sumPatternSets.end(), patternSets.begin(), patternSets.end());
        }

        switchTrigMode(false, getNumberAttribute("Exposure Time"));

        bool isSuccess = pProjector->populatePatternTableData(sumPatternSets);

        if (isSuccess) {
            qDebug() << QString("burn stripes sucessed!");
            setNumberAttribute("Total Fringes", imgs.size());
        }
        else {
            qDebug() << QString("burn stripes failed!");
        }

        switchTrigMode(true, getNumberAttribute("Exposure Time"));

        isBurnWorkFinish_ = true;
        isBurnWorkFinishChanged();
    });
}

void CameraEngine::updateDisplayImg(const QString& imgPath) {
    if(offlineCamPaintItem_) {
        QImage img(imgPath);

        if(!img.isNull())
            offlineCamPaintItem_->updateImage(img);
    }
}

void CameraEngine::startScan() {
    isProject_.store(true, std::memory_order_release);

    if (workThread_.joinable()) {
        workThread_.join();
    }

    if(scanMode_ == AppType::ScanModeType::Offline) {
        std::vector<std::vector<cv::Mat>> imgs;

        auto leftImgPaths = leftCamModel_->imgPaths();
        if(!leftImgPaths.empty()) {
            std::vector<cv::Mat> leftImgs;
            for(auto path : leftImgPaths) {
                cv::Mat img = cv::imread((leftCamModel_->curFolderPath() + "/" + path).toStdString(), 0);
                leftImgs.emplace_back(img);
            }
            imgs.emplace_back(leftImgs);
        }

        auto rightImgPaths = rightCamModel_->imgPaths();
        if(!rightImgPaths.empty()) {
            std::vector<cv::Mat> rightImgs;
            for(auto path : rightImgPaths) {
                cv::Mat img = cv::imread((rightCamModel_->curFolderPath() + "/" + path).toStdString(), 0);
                rightImgs.emplace_back(img);
            }
            imgs.emplace_back(rightImgs);
        }

        auto colorImgPaths = colorCamModel_->imgPaths();
        if(!colorImgPaths.empty()) {
            std::vector<cv::Mat> colorImgs;
            for(auto path : colorImgPaths) {
                cv::Mat img = cv::imread((colorCamModel_->curFolderPath() + "/" + path).toStdString(), 0);
                colorImgs.emplace_back(img);
            }
            imgs.emplace_back(colorImgs);
        }

        auto slCamera = getSLCamera();
        slCamera->offlineCapture(imgs, frame_);
        scanTexturePaintItem_->updateImage(QImage(frame_.textureMap_.data, frame_.textureMap_.cols, frame_.textureMap_.rows, frame_.textureMap_.step, QImage::Format_BGR888));
        VTKProcessEngine::instance()->emplaceRenderCloud(frame_.pointCloud_);
    }
    else if(scanMode_ == AppType::ScanModeType::Static) {
        workThread_ = std::thread([&]{
            auto slCamera = getSLCamera();
            if(slCamera->capture(frame_)) {
                scanTexturePaintItem_->updateImage(QImage(frame_.textureMap_.data, frame_.textureMap_.cols, frame_.textureMap_.rows, frame_.textureMap_.step, QImage::Format_BGR888));
                VTKProcessEngine::instance()->emplaceRenderCloud(frame_.pointCloud_);
            }
        });
    }

    isProject_.store(false, std::memory_order_release);
}

void CameraEngine::continuesScan() {
    isContinusStop_.store(false, std::memory_order_release);

    SafeQueue<FrameData> emptyData;
    frameDatasQueue_.swap(emptyData);

    if(workThread_.joinable()) {
        workThread_.join();
    }

    workThread_ = std::thread([&]{
        auto renderLastTime = std::chrono::steady_clock::now();
        while(!isContinusStop_.load(std::memory_order_acquire)) {
            if(frameDatasQueue_.empty()) {
                continue;
            }

            FrameData frame_;
            frameDatasQueue_.move_pop(frame_);
            
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - renderLastTime).count() > 0.035) {
                scanTexturePaintItem_->updateImage(QImage(frame_.textureMap_.data, frame_.textureMap_.cols, frame_.textureMap_.rows, frame_.textureMap_.step, QImage::Format_BGR888).copy());
                VTKProcessEngine::instance()->emplaceRenderCloud(std::move(frame_.pointCloud_));
                renderLastTime = std::chrono::steady_clock::now();
            }
        }
    });

    auto slCamera = getSLCamera();
    slCamera->continuesCapture(frameDatasQueue_);
}

void CameraEngine::pauseScan() {
    auto slCamera = getSLCamera();
    slCamera->stopContinuesCapture();

    isContinusStop_.store(true, std::memory_order_release);

    if(workThread_.joinable()) {
        workThread_.join();
    }
}

void CameraEngine::setPatternType(const int patternType) {
    if(pattern_ != nullptr) {
        delete pattern_;
        pattern_ = nullptr;
    }

    auto slCamera = getSLCamera();
    double shiftTime, cycles, confidenceThreshold, costMinDiff, costMaxDiff, maxCost, minDisp, maxDisp, minDepth, maxDepth;
    double lightStrength, exposureTime;
    double curPattern;
    std::string dlpWidth, dlpHeight;
    bool isVertical, isGpuAccelerate, isEnableDetpthCamera, isNoiseFilter;
    slCamera->getNumbericalAttribute("Phase Shift Times", shiftTime);
    slCamera->getNumbericalAttribute("Cycles", cycles);
    slCamera->getBooleanAttribute("Is Vertical", isVertical);
    slCamera->getBooleanAttribute("Gpu Accelerate", isGpuAccelerate);
    slCamera->getBooleanAttribute("Enable Depth Camera", isEnableDetpthCamera);
    slCamera->getBooleanAttribute("Noise Filter", isNoiseFilter);
    slCamera->getStringAttribute("DLP Width", dlpWidth);
    slCamera->getStringAttribute("DLP Height", dlpHeight);
    slCamera->getNumbericalAttribute("Contrast Threshold", confidenceThreshold);
    slCamera->getNumbericalAttribute("Cost Min Diff", costMinDiff);
    slCamera->getNumbericalAttribute("Cost Max Diff", costMaxDiff);
    slCamera->getNumbericalAttribute("Max Cost", maxCost);
    slCamera->getNumbericalAttribute("Min Disparity", minDisp);
    slCamera->getNumbericalAttribute("Max Disparity", maxDisp);
    slCamera->getNumbericalAttribute("Minimum Depth", minDepth);
    slCamera->getNumbericalAttribute("Maximum Depth", maxDepth);
    slCamera->getNumbericalAttribute("Light Strength", lightStrength);
    slCamera->getNumbericalAttribute("Exposure Time", exposureTime);
    slCamera->getNumbericalAttribute("Pattern", curPattern);

    if(patternType == AppType::PatternMethod::SinusCompleGrayCode) {
        pattern_ = new BinoSinusCompleGrayCodePattern();
        auto castPatternParams = static_cast<BinoPatternParams*>(pattern_->params_);
        castPatternParams->shiftTime_ = std::round(shiftTime);
        castPatternParams->cycles_ = std::round(cycles);
        castPatternParams->horizontal_ = !isVertical;
        castPatternParams->width_ = std::stoi(dlpWidth);
        castPatternParams->height_ = std::stoi(dlpHeight);
        castPatternParams->confidenceThreshold_ = confidenceThreshold;
        castPatternParams->costMinDiff_ = costMinDiff;
        castPatternParams->costMaxDiff_ = costMaxDiff;
        castPatternParams->maxCost_ = maxCost;
        castPatternParams->minDisparity_ = minDisp;
        castPatternParams->maxDisparity_ = maxDisp;
    }
    else if(patternType == AppType::PatternMethod::MutiplyFrequency) {

    }
    else if(patternType == AppType::PatternMethod::MultiViewStereoGeometry) {

    }

    slCamera->setPattern(pattern_);

    patternType_ = AppType::PatternMethod(patternType);
}

bool CameraEngine::setNumberAttribute(const QString& attributeName,
                        const double val) {
    bool isSucess = getSLCamera()->setNumberAttribute(attributeName.toStdString(), val);
    setPatternType(patternType_);
    return isSucess;
}

bool CameraEngine::setBooleanAttribute(const QString& attributeName, const bool val) {
    bool isSucess = getSLCamera()->setBooleanAttribute(attributeName.toStdString(), val);
    setPatternType(patternType_);
    return isSucess;
}

double CameraEngine::getNumberAttribute(const QString& attributeName) {
    double val;
    getSLCamera()->getNumbericalAttribute(attributeName.toStdString(), val);
    return val;
}

bool CameraEngine::getBooleanAttribute(const QString& attributeName) {
    bool val;
    getSLCamera()->getBooleanAttribute(attributeName.toStdString(), val);
    return val;
}

QString CameraEngine::getStringAttribute(const QString& attributeName) {
    std::string val;
    getSLCamera()->getStringAttribute(attributeName.toStdString(), val);
    QString qVal = QString::fromStdString(val);

    return qVal;
}

void CameraEngine::projectOnce() {
    isProject_.store(false, std::memory_order_release);

    if (workThread_.joinable()) {
        workThread_.join();
    }

    workThread_ = std::thread([&]{
        const device::camera::CameraFactory::CameraManufactor manufator =
            getStringAttribute("2D Camera Manufactor") == "Huaray"
                ? device::camera::CameraFactory::Huaray
                : device::camera::CameraFactory::Halcon;
        auto leftCamera = getSLCamera()->getCameraFactory()->getCamera(getStringAttribute("Left Camera Name").toStdString(), manufator);
        auto projector = getSLCamera()->getProjectorFactory()->getProjector(getStringAttribute("DLP Evm").toStdString());

        stripeImgs_.clear();
        emit stripeImgsChanged(stripeImgs_.size());

        isProject_.store(true, std::memory_order_release);

        projector->project(false);

        const int imgSizeWaitFor = getNumberAttribute("Total Fringes");
        const int totalExposureTime = (getNumberAttribute("Pre Exposure Time") + getNumberAttribute("Exposure Time") + getNumberAttribute("Aft Exposure Time")) * imgSizeWaitFor;
        auto endTime = std::chrono::steady_clock::now() + std::chrono::duration<int, std::ratio<1, 1000000>>(totalExposureTime + 1000000);
        while (std::chrono::steady_clock::now() < endTime || !leftCamera->getImgs().empty()) {
            if(!leftCamera->getImgs().empty()) {
                cv::Mat img = leftCamera->popImg();
                QImage qImage = QImage(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8).copy();
                stripeImgs_.emplace_back(qImage);
                realTimeRenderImg(qImage);
            }
        }

        emit stripeImgsChanged(stripeImgs_.size());

        leftCamera->clearImgs();
        std::string rightCameraName, colorCameraName;
        if(getSLCamera()->getStringAttribute("Right Camera Name", rightCameraName)) {
            getSLCamera()->getCameraFactory()->getCamera(rightCameraName, manufator)->clearImgs();
        }

        if(getSLCamera()->getStringAttribute("Color Camera Name", colorCameraName)) {
            getSLCamera()->getCameraFactory()->getCamera(colorCameraName, manufator)->clearImgs();
        }

        isProject_.store(false, std::memory_order_release);
    });
}

void CameraEngine::projectContinues() {
    isProject_.store(false, std::memory_order_release);

    if (workThread_.joinable()) {
        workThread_.join();
    }

    workThread_ = std::thread([&]{
        const device::camera::CameraFactory::CameraManufactor manufator =
            getStringAttribute("2D Camera Manufactor") == "Huaray"
                ? device::camera::CameraFactory::Huaray
                : device::camera::CameraFactory::Halcon;
        auto leftCamera = getSLCamera()->getCameraFactory()->getCamera(getStringAttribute("Left Camera Name").toStdString(), manufator);
        auto projector = getSLCamera()->getProjectorFactory()->getProjector(getStringAttribute("DLP Evm").toStdString());

        std::string rightCameraName, colorCameraName;
        bool hasRightCamera = getSLCamera()->getStringAttribute("Right Camera Name", rightCameraName);
        bool hasColorCamera = getSLCamera()->getStringAttribute("Color Camera Name", rightCameraName);
        device::camera::Camera* rightCamera = hasRightCamera ? getSLCamera()->getCameraFactory()->getCamera(getStringAttribute("Right Camera Name").toStdString(), manufator) : nullptr;
        device::camera::Camera* colorCamera = hasRightCamera ? getSLCamera()->getCameraFactory()->getCamera(getStringAttribute("Color Camera Name").toStdString(), manufator) : nullptr;

        stripeImgs_.clear();
        emit stripeImgsChanged(stripeImgs_.size());

        isProject_.store(true, std::memory_order_release);

        projector->project(true);

        while(isProject_.load(std::memory_order_acquire)) {
            cv::Mat img;
            if(!leftCamera->getImgs().empty()) {
                img = leftCamera->popImg();
            }

            if(rightCamera) {
                if(!rightCamera->getImgs().empty()) {
                    rightCamera->popImg();
                }
            }

            if(colorCamera) {
                if(!colorCamera->getImgs().empty()) {
                    colorCamera->popImg();
                }
            }

            QImage qImg = QImage(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8).copy();

            realTimeRenderImg(qImg);
        }

        //等待100ms后，将相机所有图片清空
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if(!leftCamera->getImgs().empty()) {
            leftCamera->clearImgs();
        }

        if(rightCamera) {
            if(!rightCamera->getImgs().empty()) {
                rightCamera->clearImgs();
            }
        }

        if(colorCamera) {
            if(!colorCamera->getImgs().empty()) {
                colorCamera->clearImgs();
            }
        }
    });
}

void CameraEngine::realTimeRenderImg(const QImage& img) {
    if(std::chrono::duration<double>(std::chrono::steady_clock::now() - lastTime).count() < 0.035) {
        return;
    }
    else {
        if(!img.isNull()) {
            stripePaintItem_->updateImage(img);
            lastTime = std::chrono::steady_clock::now();
        }
    }
}

void CameraEngine::pauseProject(const bool isResume) {
    auto projector = getSLCamera()->getProjectorFactory()->getProjector(getStringAttribute("DLP Evm").toStdString());
    isResume ? projector->resume() : projector->pause();
}

void CameraEngine::stepProject() {
    auto projector = getSLCamera()->getProjectorFactory()->getProjector(getStringAttribute("DLP Evm").toStdString());
    projector->step();
}

void CameraEngine::stopProject() {
    auto projector = getSLCamera()->getProjectorFactory()->getProjector(getStringAttribute("DLP Evm").toStdString());
    projector->stop();

    isProject_.store(false, std::memory_order_release);

    if (workThread_.joinable()) {
        workThread_.join();
    }
}

void CameraEngine::tenLine() {
    isProject_.store(false, std::memory_order_release);

    if (workThread_.joinable()) {
        workThread_.join();
    }

    isBurnWorkFinish_ = false;

    workThread_ = std::thread(&CameraEngine::createTenLine, this);
}
