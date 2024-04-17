#include "CalibrateEngine.h"

#include "CameraEngine.h"

#include <chrono>

using namespace slmaster;
using namespace slmaster::algorithm;
using namespace slmaster::device;
using namespace slmaster::calibration;
using namespace slmaster::cameras;

CalibrateEngine *CalibrateEngine::calibrateEngine_ = new CalibrateEngine();

CalibrateEngine *CalibrateEngine::instance() { return calibrateEngine_; }

CalibrateEngine::CalibrateEngine()
    : progress_(0), imgProcessIndex_(0), isFinish_(false),
      leftCalibrator_(nullptr), rightCalibrator_(nullptr),
      leftCameraModel_(nullptr), rightCameraModel_(nullptr) {}

Calibrator *
CalibrateEngine::getCalibrator(const AppType::TargetType targetType) {
    Calibrator *calibrator = nullptr;

    if (targetType == AppType::TargetType::ChessBoard) {
        calibrator = new ChessBoardCalibrator();
    } else if (targetType == AppType::TargetType::Blob) {
        calibrator = new CircleGridCalibrator();
    } else if (targetType == AppType::TargetType::ConcentricCircle) {
        calibrator = new ConcentricRingCalibrator();
        calibrator->setRadius(
            std::vector<float>{concentricRingParams_.innerCircleInnerRadius_,
                               concentricRingParams_.innerCircleExterRadius_,
                               concentricRingParams_.exterCircleInnerRadius_,
                               concentricRingParams_.exterCircleExterRadius_});
    }

    return calibrator;
}

void CalibrateEngine::singleCalibrate(const int targetType, const int rowNum,
                                      const int colNum,
                                      const float trueDistance,
                                      const bool useCurrentFeaturePoints) {
    qInfo() << "start calibrate single camera...";

    curCaliType_ = AppType::CaliType::Single;

    if (calibrationThread_.joinable()) {
        calibrationThread_.join();
    }

    if (useCurrentFeaturePoints) {
        qDebug() << QString("use current feature points to calibration.");

        std::vector<cv::Mat> r, t;
        auto error = cv::calibrateCamera(
            leftCalibrator_->worldPoints(), leftCalibrator_->imgPoints(),
            cv::Size(rowNum, colNum), caliInfo_.info_.M1_, caliInfo_.info_.D1_,
            r, t);

        emit errorReturn(error);
        qDebug() << QString("calibrate sucess, error is %1").arg(error);
    } else {
        leftCalibrator_.reset(getCalibrator(AppType::TargetType(targetType)));
        leftCalibrator_->setDistance(trueDistance);

        auto imgPaths = leftCameraModel_->imgPaths();
        if (imgPaths.empty()) {
            return;
        }

        for (size_t i = 0; i < imgPaths.size(); ++i) {
            auto imgPath =
                (leftCameraModel_->curFolderPath() + "/" + imgPaths[i])
                    .toLocal8Bit()
                    .toStdString();
            cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
            leftCalibrator_->emplace(img);
        }

        caliInfo_.info_.S_ =
            (cv::Mat_<double>(2, 1) << leftCalibrator_->imgs()[0].cols,
             leftCalibrator_->imgs()[0].rows);

        isFinish_.store(false, std::memory_order_release);
        updateThread_ = std::thread(&CalibrateEngine::timer_timeout_slot, this);

        calibrationThread_ = std::thread([&, colNum, rowNum] {
            double error = leftCalibrator_->calibrate(
                leftCalibrator_->imgs(), caliInfo_.info_.M1_,
                caliInfo_.info_.D1_, cv::Size(rowNum, colNum), progress_);

            if (error > 0.99 || error < 0.001) {
                emit findFeaturePointsError(
                    leftCameraModel_->curFolderPath() + "/" +
                    leftCameraModel_->imgPaths()[(int)error]);
                qDebug()
                    << QString("Img cann't find feature points, img path is %1")
                           .arg(leftCameraModel_->curFolderPath() + "/" +
                                leftCameraModel_->imgPaths()[(int)error]);
            } else {
                emit errorReturn(error);
                qDebug() << QString("calibrate sucess, error is %1").arg(error);
            }

            isFinish_.store(true, std::memory_order_release);
            if (updateThread_.joinable()) {
                updateThread_.join();
            }
        });
    }

    qInfo() << "wating calibrate result...";

    cameraTragetRowNum_ = rowNum;
    cameraTragetColNum_ = colNum;

    return;
}

void CalibrateEngine::stereoCalibrate(const int targetType, const int rowNum,
                                      const int colNum,
                                      const float trueDistance,
                                      const bool exportEpilorLine,
                                      const bool useCurrentFeaturePoints) {
    qInfo() << "start calibrate stereo camera...";

    curCaliType_ = AppType::CaliType::Stereo;

    if (calibrationThread_.joinable()) {
        calibrationThread_.join();
    }

    if (updateThread_.joinable()) {
        updateThread_.join();
    }

    if (useCurrentFeaturePoints) {
        qDebug() << QString("use current feature points to calibration.");

        auto error = cv::stereoCalibrate(
            leftCalibrator_->worldPoints(), leftCalibrator_->imgPoints(),
            rightCalibrator_->imgPoints(), caliInfo_.info_.M1_,
            caliInfo_.info_.D1_, caliInfo_.info_.M2_, caliInfo_.info_.D2_,
            cv::Size(rowNum, colNum), caliInfo_.info_.Rlr_,
            caliInfo_.info_.Tlr_, caliInfo_.info_.E_, caliInfo_.info_.F_);
        qDebug() << QString("stereo camera calibrate sucess, error is %1")
                        .arg(error);
        // TODO@Evans Liu: 增加零视差标志位以应对传统立体匹配算法
        cv::stereoRectify(
            caliInfo_.info_.M1_, caliInfo_.info_.D1_, caliInfo_.info_.M2_,
            caliInfo_.info_.D2_, cv::Size(rowNum, colNum), caliInfo_.info_.Rlr_,
            caliInfo_.info_.Tlr_, caliInfo_.info_.R1_, caliInfo_.info_.R2_,
            caliInfo_.info_.P1_, caliInfo_.info_.P2_, caliInfo_.info_.Q_, 0);
        rectify(leftCalibrator_->imgs()[0], rightCalibrator_->imgs()[0],
                caliInfo_, rectifiedImg_);

        if (exportEpilorLine) {
            findEpilines(leftCalibrator_->imgs()[0].rows,
                         leftCalibrator_->imgs()[0].cols, caliInfo_.info_.F_,
                         caliInfo_.info_.epilines12_);
        }

        emit errorReturn(error);
        qDebug() << QString("calibrate sucess, error is %1").arg(error);
    } else {
        leftCalibrator_.reset(getCalibrator(AppType::TargetType(targetType)));
        leftCalibrator_->setDistance(trueDistance);
        rightCalibrator_.reset(getCalibrator(AppType::TargetType(targetType)));
        rightCalibrator_->setDistance(trueDistance);

        auto leftImgPaths = leftCameraModel_->imgPaths();
        auto rightImgPaths = rightCameraModel_->imgPaths();
        if (leftImgPaths.empty() || rightImgPaths.empty() ||
            leftImgPaths.size() != rightImgPaths.size()) {
            return;
        }

        for (size_t i = 0; i < leftImgPaths.size(); ++i) {
            cv::Mat img = cv::imread(
                (leftCameraModel_->curFolderPath() + "/" + leftImgPaths[i])
                    .toLocal8Bit()
                    .toStdString(),
                cv::IMREAD_UNCHANGED);
            leftCalibrator_->emplace(img);
            img = cv::imread(
                (rightCameraModel_->curFolderPath() + "/" + rightImgPaths[i])
                    .toLocal8Bit()
                    .toStdString(),
                cv::IMREAD_UNCHANGED);
            rightCalibrator_->emplace(img);
        }

        caliInfo_.info_.S_ =
            (cv::Mat_<double>(2, 1) << leftCalibrator_->imgs()[0].cols,
             leftCalibrator_->imgs()[0].rows);

        isFinish_.store(false, std::memory_order_release);
        updateThread_ = std::thread(&CalibrateEngine::timer_timeout_slot, this);

        calibrationThread_ = std::thread([&, colNum, rowNum, exportEpilorLine] {
            double leftError = leftCalibrator_->calibrate(
                leftCalibrator_->imgs(), caliInfo_.info_.M1_,
                caliInfo_.info_.D1_, cv::Size(rowNum, colNum), progress_);

            if (leftError > 0.99 || leftError < 0.001) {
                emit findFeaturePointsError(
                    leftCameraModel_->curFolderPath() + "/" +
                    leftCameraModel_->imgPaths()[(int)leftError]);
                qDebug()
                    << QString("left imgs cann't find feature points, img path "
                               "is %1")
                           .arg(leftCameraModel_->curFolderPath() + "/" +
                                leftCameraModel_->imgPaths()[(int)leftError]);

                isFinish_.store(true, std::memory_order_release);
                if (updateThread_.joinable()) {
                    updateThread_.join();
                }

                return;
            } else {
                qDebug() << QString("left camera calibrate sucess, error is %1")
                                .arg(leftError);
            }

            double rightError = rightCalibrator_->calibrate(
                rightCalibrator_->imgs(), caliInfo_.info_.M2_,
                caliInfo_.info_.D2_, cv::Size(rowNum, colNum), progress_);

            if (rightError > 0.99 || rightError < 0.001) {
                emit findFeaturePointsError(
                    rightCameraModel_->curFolderPath() + "/" +
                    rightCameraModel_->imgPaths()[(int)rightError]);
                qDebug()
                    << QString("right imgs cann't find feature points, img "
                               "path is %1")
                           .arg(rightCameraModel_->curFolderPath() + "/" +
                                rightCameraModel_->imgPaths()[(int)rightError]);

                isFinish_.store(true, std::memory_order_release);
                if (updateThread_.joinable()) {
                    updateThread_.join();
                }

                return;
            } else {
                qDebug() << QString(
                                "right camera calibrate sucess, error is %1")
                                .arg(rightError);
            }

            // TODO@Evans Liu: 增加标定板无法检测到特征点情况（@Finish）
            auto error = cv::stereoCalibrate(
                leftCalibrator_->worldPoints(), leftCalibrator_->imgPoints(),
                rightCalibrator_->imgPoints(), caliInfo_.info_.M1_,
                caliInfo_.info_.D1_, caliInfo_.info_.M2_, caliInfo_.info_.D2_,
                cv::Size(rowNum, colNum), caliInfo_.info_.Rlr_,
                caliInfo_.info_.Tlr_, caliInfo_.info_.E_, caliInfo_.info_.F_);
            qDebug() << QString("stereo camera calibrate sucess, error is %1")
                            .arg(error);
            // TODO@Evans Liu: 增加零视差标志位以应对传统立体匹配算法
            cv::stereoRectify(caliInfo_.info_.M1_, caliInfo_.info_.D1_,
                              caliInfo_.info_.M2_, caliInfo_.info_.D2_,
                              cv::Size(rowNum, colNum), caliInfo_.info_.Rlr_,
                              caliInfo_.info_.Tlr_, caliInfo_.info_.R1_,
                              caliInfo_.info_.R2_, caliInfo_.info_.P1_,
                              caliInfo_.info_.P2_, caliInfo_.info_.Q_, 0);
            rectify(leftCalibrator_->imgs()[0], rightCalibrator_->imgs()[0],
                    caliInfo_, rectifiedImg_);

            if (exportEpilorLine) {
                findEpilines(leftCalibrator_->imgs()[0].rows,
                             leftCalibrator_->imgs()[0].cols,
                             caliInfo_.info_.F_, caliInfo_.info_.epilines12_);
            }

            emit errorReturn(error);

            isFinish_.store(true, std::memory_order_release);
            if (updateThread_.joinable()) {
                updateThread_.join();
            }
        });
    }

    qInfo() << "wating calibrate result...";

    cameraTragetRowNum_ = rowNum;
    cameraTragetColNum_ = colNum;

    return;
}

void CalibrateEngine::findEpilines(const int rows, const int cols,
                                   const cv::Mat &fundermental,
                                   cv::Mat &epiline) {
    CV_Assert(!fundermental.empty());

    epiline = cv::Mat(rows, cols, CV_32FC3, cv::Scalar(0.f, 0.f, 0.f));

    std::vector<cv::Point2f> points;
    std::vector<cv::Vec3f> epilinesVec;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv::Point2f imgPoint(j, i);
            points.emplace_back(imgPoint);
        }
    }

    computeCorrespondEpilines(points, 1, fundermental, epilinesVec);
    for (int i = 0; i < rows; ++i) {
        auto ptrEpilines = epiline.ptr<cv::Vec3f>(i);
        for (int j = 0; j < cols; ++j) {
            ptrEpilines[j] = cv::Vec3f(epilinesVec[cols * i + j][0],
                                       epilinesVec[cols * i + j][1],
                                       epilinesVec[cols * i + j][2]);
        }
    }
}

void CalibrateEngine::rectify(const cv::Mat &leftImg, const cv::Mat &rightImg,
                              const CaliInfo &info, cv::Mat &rectifyImg) {
    rectifyImg = cv::Mat(leftImg.rows, leftImg.cols * 2, CV_8UC1);
    cv::Mat img_rect_Left =
        rectifyImg(cv::Rect(0, 0, rectifyImg.cols / 2, rectifyImg.rows));
    cv::Mat img_rect_Right = rectifyImg(
        cv::Rect(rectifyImg.cols / 2, 0, rectifyImg.cols / 2, rectifyImg.rows));

    cv::Mat copyImgLeft, copyImgRight;

    if (leftImg.type() == CV_8UC1) {
        leftImg.copyTo(copyImgLeft);
    } else {
        cv::cvtColor(leftImg, copyImgLeft, cv::COLOR_RGBA2GRAY);
    }

    if (rightImg.type() == CV_8UC1) {
        rightImg.copyTo(copyImgRight);
    } else {
        cv::cvtColor(rightImg, copyImgRight, cv::COLOR_RGBA2GRAY);
    }

    cv::Mat map_x_left, map_x_right, map_y_left, map_y_right;
    initUndistortRectifyMap(info.info_.M1_, info.info_.D1_, info.info_.R1_,
                            info.info_.P1_, leftImg.size(), CV_16SC2,
                            map_x_left, map_y_left);
    initUndistortRectifyMap(info.info_.M2_, info.info_.D2_, info.info_.R2_,
                            info.info_.P2_, leftImg.size(), CV_16SC2,
                            map_x_right, map_y_right);
    remap(copyImgLeft, img_rect_Left, map_x_left, map_y_left, cv::INTER_LINEAR);
    remap(copyImgRight, img_rect_Right, map_x_right, map_y_right,
          cv::INTER_LINEAR);

    cvtColor(rectifyImg, rectifyImg, cv::COLOR_GRAY2BGR);

    cv::RNG rng(255);
    for (int i = 0; i < 30; i++) {
        line(rectifyImg, cv::Point(0, rectifyImg.rows / 30 * i),
             cv::Point(rectifyImg.cols, rectifyImg.rows / 30 * i),
             cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                        rng.uniform(0, 255)),
             1);
    }
}

void CalibrateEngine::timer_timeout_slot() {
    float progressStereo = 0.f;
    float progressTriple = 0.f;
    float progressPre = progress_;
    while (!isFinish_.load(std::memory_order_acquire)) {
        if (progressPre != progress_) {
            progressPre = progress_;

            if (AppType::CaliType::Single == curCaliType_) {
                emit progressChanged(progress_ * 100);
            } else if (AppType::CaliType::Stereo == curCaliType_) {
                progressStereo +=
                    progressStereo < 0.5
                        ? progress_ / 2.f - progressStereo
                        : progress_ / 2.f - (progressStereo - 0.5);
                emit progressChanged(progressStereo * 100);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}

void CalibrateEngine::removeProjectImg(const QString &path) {
    for (int i = 0; i < projectorModel_->imgPaths().size(); ++i) {
        if (projectorModel_->imgPaths()[i] == path) {
            projectorCaliImgs_.erase(projectorCaliImgs_.begin() + i);
            projCameraPoints_.erase(projCameraPoints_.begin() + i);
            projectorPoints_.erase(projectorPoints_.begin() + i);
            if (!projectorErrorDistributes_.empty()) {
                projectorErrorDistributes_.erase(
                    projectorErrorDistributes_.begin() + i);
            }

            projectorModel_->erase(i);

            emit projectorModelChanged();
        }
    }
}

void CalibrateEngine::removeCameraImg(const QString &path, const bool isLeft) {
    auto model = isLeft ? leftCameraModel_ : rightCameraModel_;

    int index = 0;
    for (int i = 0; i < model->imgPaths().size(); ++i) {
        if (model->imgPaths()[i] == path) {
            index = i;
        }
    }

    if (isLeft && leftCalibrator_) {
        if (leftCalibrator_->drawedFeaturesImgs().size() > index) {
            leftCalibrator_->drawedFeaturesImgs().erase(
                leftCalibrator_->drawedFeaturesImgs().begin() + index);
        }
    } else if (rightCalibrator_) {
        if (rightCalibrator_->drawedFeaturesImgs().size() > index) {
            rightCalibrator_->drawedFeaturesImgs().erase(
                rightCalibrator_->drawedFeaturesImgs().begin() + index);
        }
    }
}

void CalibrateEngine::exit() {
    if (calibrationThread_.joinable()) {
        calibrationThread_.join();
    }

    if (updateThread_.joinable()) {
        updateThread_.join();
    }
}

void CalibrateEngine::updateDisplayImg(const QString &imgPath) {
    if (curCaliType_ != AppType::CaliType::Projector) {
        if (leftCalibrator_) {
            if (!leftCalibrator_->drawedFeaturesImgs().empty() &&
                !leftCameraModel_->imgPaths().empty()) {
                for (int i = 0; i < leftCameraModel_->imgPaths().size(); ++i) {
                    if (leftCameraModel_->curFolderPath() + "/" +
                            leftCameraModel_->imgPaths()[i] ==
                        imgPath) {
                        cv::Mat img =
                            leftCalibrator_->drawedFeaturesImgs().size() > i
                                ? leftCalibrator_->drawedFeaturesImgs()[i]
                                : cv::imread(
                                      imgPath.toLocal8Bit().toStdString(), 0)
                                      .clone();
                        if (img.type() == CV_8UC1) {
                            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
                        }
                        offlineCamPaintItem_->updateImage(
                            QImage(img.data, img.cols, img.rows, img.step,
                                   QImage::Format_BGR888)
                                .copy());
                        return;
                    }
                }
            }
        }

        if (rightCalibrator_) {
            if (!rightCalibrator_->drawedFeaturesImgs().empty() &&
                !rightCameraModel_->imgPaths().empty()) {
                for (int i = 0; i < rightCameraModel_->imgPaths().size(); ++i) {
                    if (rightCameraModel_->curFolderPath() + "/" +
                            rightCameraModel_->imgPaths()[i] ==
                        imgPath) {
                        cv::Mat img =
                            rightCalibrator_->drawedFeaturesImgs().size() > i
                                ? rightCalibrator_->drawedFeaturesImgs()[i]
                                : cv::imread(
                                      imgPath.toLocal8Bit().toStdString(), 0)
                                      .clone();
                        if (img.type() == CV_8UC1) {
                            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
                        }
                        offlineCamPaintItem_->updateImage(
                            QImage(img.data, img.cols, img.rows, img.step,
                                   QImage::Format_BGR888)
                                .copy());
                        return;
                    }
                }
            }
        }

        QImage img(imgPath);

        if (!img.isNull())
            offlineCamPaintItem_->updateImage(img);
    } else {
        for (int i = 0; i < projectorModel_->imgPaths().size(); ++i) {
            if (projectorModel_->imgPaths()[i] == imgPath) {
                onlineProjHonriPaintItem_->updateImage(QImage(
                    projectorCaliImgs_[i][0].data,
                    projectorCaliImgs_[i][0].cols,
                    projectorCaliImgs_[i][0].rows,
                    projectorCaliImgs_[i][0].step, QImage::Format_Grayscale8));
                onlineProjVertiPaintItem_->updateImage(QImage(
                    projectorCaliImgs_[i][1].data,
                    projectorCaliImgs_[i][1].cols,
                    projectorCaliImgs_[i][1].rows,
                    projectorCaliImgs_[i][1].step, QImage::Format_Grayscale8));
                onlineProjColorPaintItem_->updateImage(QImage(
                    projectorCaliImgs_[i][2].data,
                    projectorCaliImgs_[i][2].cols,
                    projectorCaliImgs_[i][2].rows,
                    projectorCaliImgs_[i][2].step, QImage::Format_BGR888));
                onlineProjPaintItem_->updateImage(QImage(
                    projectorCaliImgs_[i][3].data,
                    projectorCaliImgs_[i][3].cols,
                    projectorCaliImgs_[i][3].rows,
                    projectorCaliImgs_[i][3].step, QImage::Format_BGR888));
            }
        }
    }
}

const QVariantList
CalibrateEngine::updateErrorDistribute(const QString &imgPath,
                                       const bool isLeft) {
    QVariantList errorPoints;

    if (curCaliType_ != AppType::Projector) {
        if (leftCalibrator_ && isLeft) {
            if (!leftCalibrator_->errors().empty() &&
                !leftCameraModel_->imgPaths().empty()) {
                for (int i = 0; i < leftCameraModel_->imgPaths().size(); ++i) {
                    if (leftCameraModel_->imgPaths()[i] == imgPath) {
                        auto curErrorsDistribute = leftCalibrator_->errors()[i];
                        for (auto point : curErrorsDistribute) {
                            errorPoints.append(
                                QVariant::fromValue(QPointF(point.x, point.y)));
                        }
                        return errorPoints;
                    }
                }
            }
        }

        if (rightCalibrator_ && !isLeft) {
            if (!rightCalibrator_->errors().empty() &&
                !rightCameraModel_->imgPaths().empty()) {
                for (int i = 0; i < rightCameraModel_->imgPaths().size(); ++i) {
                    if (rightCameraModel_->imgPaths()[i] == imgPath) {
                        auto curErrorsDistribute =
                            rightCalibrator_->errors()[i];
                        for (auto point : curErrorsDistribute) {
                            errorPoints.append(
                                QVariant::fromValue(QPointF(point.x, point.y)));
                        }
                        return errorPoints;
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < projectorModel_->imgPaths().size(); ++i) {
            if (projectorModel_->imgPaths()[i] == imgPath) {
                if (projectorErrorDistributes_.empty()) {
                    return errorPoints;
                }

                auto curErrorsDistribute = projectorErrorDistributes_[i];
                for (auto point : curErrorsDistribute) {
                    errorPoints.append(
                        QVariant::fromValue(QPointF(point.x, point.y)));
                }
                return errorPoints;
            }
        }
    }

    return errorPoints;
}

void CalibrateEngine::saveCaliInfo(const QString &path) {
    auto filePath = path.mid(8) + "/caliInfo.yml";
    CaliPacker paker(&caliInfo_);
    paker.writeCaliInfo(curCaliType_, filePath.toLocal8Bit().toStdString());
}

void CalibrateEngine::displayStereoRectifyMap() {
    qInfo() << "Display rectify map...";

    if (rectifiedImg_.empty()) {
        qDebug() << "Rectify map is empty...";
        return;
    }

    offlineCamPaintItem_->updateImage(
        QImage(rectifiedImg_.data, rectifiedImg_.cols, rectifiedImg_.rows,
               rectifiedImg_.step, QImage::Format_BGR888));

    qInfo() << "Display rectify map sucess!";
}

bool CalibrateEngine::captureOnce() {
    if (!projectorErrorDistributes_.empty()) {
        projectorErrorDistributes_.clear();
    }

    auto slCamera = CameraEngine::instance()->getSLCamera();

    std::string camManufactor, leftCameraName, rightCameraName, colorCameraName,
        dlpEvmName;
    slCamera->getStringAttribute("2D Camera Manufactor", camManufactor);
    const CameraFactory::CameraManufactor manufator =
        camManufactor == "Huaray" ? CameraFactory::Huaray
                                  : CameraFactory::Halcon;
    slCamera->getStringAttribute("Left Camera Name", leftCameraName);
    slCamera->getStringAttribute("DLP Evm", dlpEvmName);

    Projector *projector =
        slCamera->getProjectorFactory()->getProjector(dlpEvmName);

    Camera *leftCamera = nullptr, *rightCamera = nullptr,
           *colorCamera = nullptr;
    leftCamera =
        slCamera->getCameraFactory()->getCamera(leftCameraName, manufator);
    if (slCamera->getStringAttribute("Right Camera Name", rightCameraName)) {
        rightCamera =
            slCamera->getCameraFactory()->getCamera(rightCameraName, manufator);
    }
    if (slCamera->getStringAttribute("Color Camera Name", colorCameraName)) {
        colorCamera =
            slCamera->getCameraFactory()->getCamera(colorCameraName, manufator);
    }

    projector->project(false);

    const int imgSizeWaitFor =
        CameraEngine::instance()->getNumberAttribute("Total Fringes");
    const int totalExposureTime =
        (CameraEngine::instance()->getNumberAttribute("Pre Exposure Time") +
         CameraEngine::instance()->getNumberAttribute("Exposure Time") +
         CameraEngine::instance()->getNumberAttribute("Aft Exposure Time")) *
        imgSizeWaitFor;
    auto endTime = std::chrono::steady_clock::now() +
                   std::chrono::duration<int, std::ratio<1, 1000000>>(
                       totalExposureTime + 1000000);
    while (leftCamera->getImgs().size() != imgSizeWaitFor) {
        if (std::chrono::steady_clock::now() > endTime) {
            leftCamera->clearImgs();
            if (rightCamera) {
                rightCamera->clearImgs();
            }
            if (colorCamera) {
                colorCamera->clearImgs();
            }
            return false;
        }
    }

    leftCamera->setTrigMode(trigSoftware);
    leftCamera->setNumberAttribute("ExposureTime", 100000);
    cv::Mat texture = leftCamera->capture();
    if (texture.type() == CV_8UC3) {
        cv::cvtColor(texture, texture, cv::COLOR_BGR2GRAY);
    }
    leftCamera->setNumberAttribute(
        "ExposureTime",
        CameraEngine::instance()->getNumberAttribute("Exposure Time"));

    auto orderTablesRecord = CameraEngine::instance()->getOrderTableRecord();
    static cv::Mat honrizonUnwrapMap, verticalUnwrapMap, normHUnwrapMap,
        normVUnwrapMap, textureMap;
    const int dlpHeight =
        CameraEngine::instance()->getStringAttribute("DLP Height").toInt();
    const int dlpWidth =
        CameraEngine::instance()->getStringAttribute("DLP Width").toInt();

    for (int i = 0; i < orderTablesRecord.size(); ++i) {
        std::vector<cv::Mat> imgs;
        int index = 0;
        while (index++ < orderTablesRecord[i].patternsNum_) {
            cv::Mat img = leftCamera->popImg();

            if (img.type() == CV_8UC3) {
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            }

            imgs.push_back(img);
        }

        SinusCompleGrayCodePattern::Params params;
        params.confidenceThreshold =
            CameraEngine::instance()->getNumberAttribute("Contrast Threshold");
        params.height = dlpHeight;
        params.width = dlpWidth;
        params.horizontal = !orderTablesRecord[i].isVertical_;
        params.shiftTime = orderTablesRecord[i].shiftTime_;
        params.nbrOfPeriods =
            std::pow(2, orderTablesRecord[i].patternsNum_ -
                            orderTablesRecord[i].shiftTime_ - 1);
        auto pattern = SinusCompleGrayCodePattern::create(params);

        cv::Mat confidenceMap, wrappedPhaseMap, floorMap, unwrapMap,
            normalizeUnwrapMap;

        pattern->computeConfidenceMap(imgs, confidenceMap);
        pattern->computePhaseMap(imgs, wrappedPhaseMap);
        pattern->computeFloorMap(imgs, confidenceMap, wrappedPhaseMap,
                                 floorMap);
        pattern->unwrapPhaseMap(wrappedPhaseMap, floorMap, confidenceMap, unwrapMap);

        cv::normalize(unwrapMap, normalizeUnwrapMap, 0, 255, cv::NORM_MINMAX);

        if (orderTablesRecord[i].isVertical_) {
            verticalUnwrapMap = unwrapMap;
            textureMap = texture;
            normVUnwrapMap = normalizeUnwrapMap;
            normVUnwrapMap.convertTo(normVUnwrapMap, CV_8UC1);
            onlineProjVertiPaintItem_->updateImage(QImage(
                normVUnwrapMap.data, normVUnwrapMap.cols, normVUnwrapMap.rows,
                normVUnwrapMap.step, QImage::Format_Grayscale8));
        } else {
            honrizonUnwrapMap = unwrapMap;
            normHUnwrapMap = normalizeUnwrapMap;
            normHUnwrapMap.convertTo(normHUnwrapMap, CV_8UC1);
            onlineProjHonriPaintItem_->updateImage(QImage(
                normHUnwrapMap.data, normHUnwrapMap.cols, normHUnwrapMap.rows,
                normHUnwrapMap.step, QImage::Format_Grayscale8));
        }
    }

    leftCamera->clearImgs();
    if (rightCamera) {
        rightCamera->clearImgs();
    }
    if (colorCamera) {
        colorCamera->clearImgs();
    }

    cv::Mat projectorImg = cv::Mat::zeros(dlpHeight, dlpWidth, CV_8UC3);

    auto calibrator = getCalibrator(projectorCaliParams_.targetType_);
    std::vector<cv::Point2f> featurePoints;
    // textureMap.convertTo(textureMap, CV_8UC1);

    if (!calibrator->findFeaturePoints(textureMap,
                                       cv::Size(projectorCaliParams_.rowNum_,
                                                projectorCaliParams_.colNum_),
                                       featurePoints)) {
        return false;
    } else {
        cv::cvtColor(textureMap, textureMap, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(textureMap,
                                  cv::Size(projectorCaliParams_.rowNum_,
                                           projectorCaliParams_.colNum_),
                                  featurePoints, true);
        onlineProjColorPaintItem_->updateImage(
            QImage(textureMap.data, textureMap.cols, textureMap.rows,
                   textureMap.step, QImage::Format_BGR888));

        std::vector<cv::Point2f> remapPoints;
        for (auto camPoint : featurePoints) {
            auto projPoint = remapProjectorPoint(honrizonUnwrapMap,
                                                 verticalUnwrapMap, camPoint);
            remapPoints.emplace_back(projPoint);
        }

        projCameraPoints_.emplace_back(featurePoints);
        projectorPoints_.emplace_back(remapPoints);

        cv::drawChessboardCorners(projectorImg,
                                  cv::Size(projectorCaliParams_.rowNum_,
                                           projectorCaliParams_.colNum_),
                                  remapPoints, true);
        onlineProjPaintItem_->updateImage(
            QImage(projectorImg.data, projectorImg.cols, projectorImg.rows,
                   projectorImg.step, QImage::Format_BGR888));
    }

    std::vector<cv::Mat> groupImg;
    groupImg.emplace_back(normHUnwrapMap);
    groupImg.emplace_back(normVUnwrapMap);
    groupImg.emplace_back(textureMap);
    groupImg.emplace_back(projectorImg);
    projectorCaliImgs_.emplace_back(groupImg);

    if (rightCamera) {
        rightCamera->clearImgs();
    }

    if (colorCamera) {
        colorCamera->clearImgs();
    }

    const QString index =
        projectorModel_->imgPaths().empty()
            ? "0"
            : QString::number(
                  projectorModel_
                      ->imgPaths()[projectorModel_->imgPaths().size() - 1]
                      .toInt() +
                  1);
    projectorModel_->emplace_back(index);
    emit projectorModelChanged();

    return true;
}

const cv::Point2f
CalibrateEngine::remapProjectorPoint(const cv::Mat &honrizonPhaseMap,
                                     const cv::Mat &verticalPhaseMap,
                                     const cv::Point2f &camPoint) {
#ifdef LinearInter
    int index_Y_Upper = std::ceil(camPoint.y);
    int index_Y_Lower = std::floor(camPoint.y);
    int index_X_Upper = std::ceil(camPoint.x);
    int index_X_Lower = std::floor(camPoint.x);
    const float vPLU =
        verticalPhaseMap.ptr<float>(index_Y_Upper)[index_X_Lower];
    const float vPRU =
        verticalPhaseMap.ptr<float>(index_Y_Upper)[index_X_Upper];
    const float vPLD =
        verticalPhaseMap.ptr<float>(index_Y_Lower)[index_X_Lower];
    const float vPRD =
        verticalPhaseMap.ptr<float>(index_Y_Lower)[index_X_Upper];
    const float tPLU =
        honrizonPhaseMap.ptr<float>(index_Y_Upper)[index_X_Lower];
    const float tPRU =
        honrizonPhaseMap.ptr<float>(index_Y_Upper)[index_X_Upper];
    const float tPLD =
        honrizonPhaseMap.ptr<float>(index_Y_Lower)[index_X_Lower];
    const float tPRD =
        honrizonPhaseMap.ptr<float>(index_Y_Lower)[index_X_Upper];
    const float vfR1 =
        (index_X_Upper - camPoint.x) / (index_X_Upper - index_X_Lower) * vPLD +
        (camPoint.x - index_X_Lower) / (index_X_Upper - index_X_Lower) * vPRD;
    const float vfR2 =
        (index_X_Upper - camPoint.x) / (index_X_Upper - index_X_Lower) * vPLU +
        (camPoint.x - index_X_Lower) / (index_X_Upper - index_X_Lower) * vPRU;
    const float tfR1 =
        (index_X_Upper - camPoint.x) / (index_X_Upper - index_X_Lower) * tPLD +
        (camPoint.x - index_X_Lower) / (index_X_Upper - index_X_Lower) * tPRD;
    const float tfR2 =
        (index_X_Upper - camPoint.x) / (index_X_Upper - index_X_Lower) * tPLU +
        (camPoint.x - index_X_Lower) / (index_X_Upper - index_X_Lower) * tPRU;
    const float verticalPhaseValue =
        (index_Y_Upper - camPoint.y) / (index_Y_Upper - index_Y_Lower) * vfR1 +
        (camPoint.y - index_Y_Lower) / (index_Y_Upper - index_Y_Lower) * vfR2;
    const float transversePhaseValue =
        (index_Y_Upper - camPoint.y) / (index_Y_Upper - index_Y_Lower) * tfR1 +
        (camPoint.y - index_Y_Lower) / (index_Y_Upper - index_Y_Lower) * tfR2;
#else
    int index_Y = std::round(camPoint.y);
    int index_X = std::round(camPoint.x);
    const float verticalPhaseValue =
        verticalPhaseMap.ptr<float>(index_Y)[index_X];
    const float transversePhaseValue =
        honrizonPhaseMap.ptr<float>(index_Y)[index_X];
#endif
    const float xLocation =
        (verticalPhaseValue) / CV_2PI * projectorCaliParams_.verticalPitch_;
    const float yLocation =
        (transversePhaseValue) / CV_2PI * projectorCaliParams_.honrizonPitch_;
    return cv::Point2f(xLocation, yLocation);
}

double CalibrateEngine::calibrateProjector() {
    curCaliType_ = AppType::CaliType::Projector;

    const int dlpHeight =
        CameraEngine::instance()->getStringAttribute("DLP Height").toInt();
    const int dlpWidth =
        CameraEngine::instance()->getStringAttribute("DLP Width").toInt();

    std::vector<cv::Point3f> worldPoints;
    for (int j = projectorCaliParams_.colNum_ - 1; j >= 0; --j) {
        for (int k = projectorCaliParams_.rowNum_ - 1; k >= 0; --k) {
            cv::Point3f worldPoint(k * projectorCaliParams_.trueDistance_,
                                   j * projectorCaliParams_.trueDistance_, 0);
            worldPoints.emplace_back(worldPoint);
        }
    }

    projectorErrorDistributes_.clear();

    if (projectorCaliParams_.projCaliType_ ==
        AppType::ProjectorCaliType::Intrinsic) {
        std::vector<std::vector<cv::Point3f>> totoalWorldPoints(
            projectorPoints_.size(), worldPoints);
        std::vector<cv::Mat> r, t;
        const double calibrationErrors = cv::calibrateCamera(
            totoalWorldPoints, projectorPoints_, cv::Size(dlpWidth, dlpHeight),
            caliInfo_.info_.M4_, caliInfo_.info_.D4_, r, t);

        for (int i = 0; i < totoalWorldPoints.size(); ++i) {
            std::vector<cv::Point2f> reprojectPoints;
            std::vector<cv::Point2f> curErrorsDistribute;
            cv::projectPoints(totoalWorldPoints[i], r[i], t[i],
                              caliInfo_.info_.M4_, caliInfo_.info_.D4_,
                              reprojectPoints);
            for (int j = 0; j < reprojectPoints.size(); ++j) {
                curErrorsDistribute.emplace_back(cv::Point2f(
                    reprojectPoints[j].x - projectorPoints_[i][j].x,
                    reprojectPoints[j].y - projectorPoints_[i][j].y));
            }
            projectorErrorDistributes_.emplace_back(curErrorsDistribute);
        }

        emit projectErrorReturn(calibrationErrors);

        return calibrationErrors;
    } else { // 外参标定
        std::vector<std::vector<cv::Point3f>> totoalWorldPoints(
            projectorPoints_.size(), worldPoints);
        std::vector<cv::Mat> tempR, tempT;
        cv::Mat M1, D1;
        const double calibrationErrors =
            cv::calibrateCamera(totoalWorldPoints, projCameraPoints_,
                                cv::Size(1280, 1024), M1, D1, tempR, tempT);

        std::vector<cv::Point3f> camPoints;
        for (int i = 0; i < projCameraPoints_.size(); ++i) {
            cv::Mat r, t;
            cv::solvePnP(worldPoints, projCameraPoints_[i], caliInfo_.info_.M1_,
                         caliInfo_.info_.D1_, r, t);
            cv::Rodrigues(r, r);
            for (auto point : worldPoints) {
                cv::Mat worldLoc =
                    (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
                cv::Mat camLoc = r * worldLoc + t;
                camPoints.emplace_back(cv::Point3f(camLoc.ptr<double>(0)[0],
                                                   camLoc.ptr<double>(1)[0],
                                                   camLoc.ptr<double>(2)[0]));
            }
        }

        std::vector<cv::Point2f> projectorPixelPoints;
        cv::Mat solveL;
        for (auto points : projectorPoints_) {
            projectorPixelPoints.insert(projectorPixelPoints.end(),
                                        points.begin(), points.end());

            for (auto point : points) {
                const float phaseVal =
                    (point.x / projectorCaliParams_.verticalPitch_) * CV_2PI;
                cv::Mat rowData(1, 8, CV_32FC1);
                const int curIndex = solveL.rows;
                rowData.ptr<float>(0)[0] = camPoints[curIndex].x;
                rowData.ptr<float>(0)[1] = camPoints[curIndex].y;
                rowData.ptr<float>(0)[2] = camPoints[curIndex].z;
                rowData.ptr<float>(0)[3] = 1.f;
                rowData.ptr<float>(0)[4] = -phaseVal * camPoints[curIndex].x;
                rowData.ptr<float>(0)[5] = -phaseVal * camPoints[curIndex].y;
                rowData.ptr<float>(0)[6] = -phaseVal * camPoints[curIndex].z;
                rowData.ptr<float>(0)[7] = -phaseVal;
                solveL.push_back(rowData);
            }
        }

        cv::solvePnPRansac(camPoints, projectorPixelPoints, caliInfo_.info_.M4_,
                           caliInfo_.info_.D4_, caliInfo_.info_.Rlp_,
                           caliInfo_.info_.Tlp_, false, 1000000);
        cv::Rodrigues(caliInfo_.info_.Rlp_, caliInfo_.info_.Rlp_);
        // 八参数标定
        caliInfo_.info_.K1_ = cv::Mat(8, 1, CV_64FC1);
        cv::SVD::solveZ(solveL, caliInfo_.info_.K1_);
    }

    return 0.f;
}

void CalibrateEngine::readLocalCaliFile(const QString &path) {
    cv::FileStorage fileRead(path.mid(8).toLocal8Bit().toStdString(),
                             cv::FileStorage::READ);

    cv::Mat M1, D1, M4, D4;
    fileRead["M1"] >> M1;
    fileRead["D1"] >> D1;
    fileRead["M4"] >> M4;
    fileRead["D4"] >> D4;

    if (!M1.empty()) {
        caliInfo_.info_.M1_ = M1;
    }

    if (!D1.empty()) {
        caliInfo_.info_.D1_ = D1;
    }

    if (!M4.empty()) {
        caliInfo_.info_.M4_ = M4;
    }

    if (!D4.empty()) {
        caliInfo_.info_.D4_ = D4;
    }
}

void CalibrateEngine::invFeatureSequence(const QString &path,
                                         const bool isLeft) {
    auto model = isLeft ? leftCameraModel_ : rightCameraModel_;

    int index = 0;
    for (int i = 0; i < model->imgPaths().size(); ++i) {
        if (model->imgPaths()[i] == path) {
            index = i;
        }
    }

    auto &points = isLeft ? leftCalibrator_->imgPoints()[index]
                          : rightCalibrator_->imgPoints()[index];
    std::reverse(points.begin(), points.end());

    cv::Mat newDrawedImg = isLeft ? leftCalibrator_->imgs()[index].clone()
                                  : rightCalibrator_->imgs()[index].clone();
    if (newDrawedImg.type() == CV_8UC1) {
        cv::cvtColor(newDrawedImg, newDrawedImg, cv::COLOR_GRAY2BGR);
    }
    cv::drawChessboardCorners(
        newDrawedImg, cv::Size(cameraTragetRowNum_, cameraTragetColNum_),
        points, true);

    if (isLeft) {
        newDrawedImg.copyTo(leftCalibrator_->drawedFeaturesImgs()[index]);
    } else {
        newDrawedImg.copyTo(rightCalibrator_->drawedFeaturesImgs()[index]);
    }

    updateDisplayImg(path);
}

void CalibrateEngine::invFeatureHVSequence(const QString &path,
                                           const bool isLeft) {
    auto model = isLeft ? leftCameraModel_ : rightCameraModel_;

    int index = 0;
    for (int i = 0; i < model->imgPaths().size(); ++i) {
        if (model->imgPaths()[i] == path) {
            index = i;
        }
    }

    auto &points = isLeft ? leftCalibrator_->imgPoints()[index]
                          : rightCalibrator_->imgPoints()[index];

    std::vector<cv::Point2f> newPoints;
    for (int i = 0; i < cameraTragetColNum_; ++i) {
        for (int j = 0; j < cameraTragetRowNum_; ++j) {
            auto tempPoint =
                points[cameraTragetColNum_ * j + (cameraTragetColNum_ - 1 - i)];
            newPoints.emplace_back(tempPoint);
        }
    }

    points = newPoints;

    cv::Mat newDrawedImg = isLeft ? leftCalibrator_->imgs()[index].clone()
                                  : rightCalibrator_->imgs()[index].clone();
    if (newDrawedImg.type() == CV_8UC1) {
        cv::cvtColor(newDrawedImg, newDrawedImg, cv::COLOR_GRAY2BGR);
    }
    cv::drawChessboardCorners(
        newDrawedImg, cv::Size(cameraTragetRowNum_, cameraTragetColNum_),
        points, true);

    if (isLeft) {
        newDrawedImg.copyTo(leftCalibrator_->drawedFeaturesImgs()[index]);
    } else {
        newDrawedImg.copyTo(rightCalibrator_->drawedFeaturesImgs()[index]);
    }

    updateDisplayImg(path);
}