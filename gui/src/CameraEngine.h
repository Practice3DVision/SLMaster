#ifndef __CAMERA_ENGINE_H_
#define __CAMERA_ENGINE_H_

#include <vector>
#include <thread>

#include <QObject>
#include <QThread>

#include <opencv2/opencv.hpp>

#include "AppType.h"
#include "ImagePaintItem.h"
#include "typeDef.h"
#include "CameraModel.h"

#include "slCameraFactory.h"
#include "safeQueue.hpp"

class CameraEngine : public QObject {
    Q_OBJECT
    Q_PROPERTY_AUTO(bool, isOnLine)
    Q_PROPERTY_AUTO(bool, isConnected)
    Q_PROPERTY_AUTO(bool, isBurnWorkFinish)
  public:
    static CameraEngine* instance();
    struct OrderTableRecord {
        OrderTableRecord() {}
        OrderTableRecord(const int patternsNum, const int shiftTime, const bool isVertical) : patternsNum_(patternsNum), shiftTime_(shiftTime), isVertical_(isVertical) {}
        int patternsNum_;
        int shiftTime_;
        bool isVertical_;
    };
    //Device page
    Q_INVOKABLE void startDetectCameraState();
    Q_INVOKABLE int createStripe(const int pixelDepth, const int direction, const int stripeType, const int defocusMethod, const int imgWidth, const int imgHeight, const int cycles, const int shiftTime, const bool isKeepAdd);
    Q_INVOKABLE void displayStripe(const int stripeIndex);
    Q_INVOKABLE void selectCamera(const int cameraType);
    Q_INVOKABLE void setCameraJsonPath(const std::string jsonPath);
    Q_INVOKABLE bool connectCamera();
    Q_INVOKABLE bool disConnectCamera();
    Q_INVOKABLE void burnStripe();
    Q_INVOKABLE void bindStripePaintItem(ImagePaintItem* stripePaintItem) { stripePaintItem_ = stripePaintItem; }
    //offlineScan page
    Q_INVOKABLE void bindOfflineCamPaintItem(ImagePaintItem* camPaintItem) { offlineCamPaintItem_ = camPaintItem; }
    //scan page
    Q_INVOKABLE void setScanMode(const int scanMode) { scanMode_ = AppType::ScanModeType(scanMode); }
    Q_INVOKABLE void projectOnce();
    Q_INVOKABLE void projectContinues();
    Q_INVOKABLE void pauseProject(const bool isResume);
    Q_INVOKABLE void stepProject();
    Q_INVOKABLE void stopProject();
    Q_INVOKABLE void tenLine();
    Q_INVOKABLE void startScan();
    Q_INVOKABLE void continuesScan();
    Q_INVOKABLE void pauseScan();
    Q_INVOKABLE void bindOfflineLeftCamModel(CameraModel* model) { leftCamModel_ = model; }
    Q_INVOKABLE void bindOfflineRightCamModel(CameraModel* model) { rightCamModel_ = model; }
    Q_INVOKABLE void bindOfflineColorCamModel(CameraModel* model) { colorCamModel_ = model; }
    Q_INVOKABLE void bindScanTexturePaintItem(ImagePaintItem* paintItem) { scanTexturePaintItem_ = paintItem; }
    Q_INVOKABLE void updateDisplayImg(const QString& imgPath);
    Q_INVOKABLE void saveStripe(const QString& path);
    Q_INVOKABLE void setPatternType(const int patternType);
    Q_INVOKABLE bool setNumberAttribute(const QString& attributeName,
                                    const double val);
    Q_INVOKABLE bool setBooleanAttribute(const QString& attributeName, const bool val);
    Q_INVOKABLE double getNumberAttribute(const QString& attributeName);
    Q_INVOKABLE bool getBooleanAttribute(const QString& attributeName);
    Q_INVOKABLE QString getStringAttribute(const QString& attributeName);
    Q_INVOKABLE const slmaster::FrameData& getCurFrame() { return frame_; }
    std::shared_ptr<slmaster::SLCamera> getSLCamera() { return slCameraFactory_.getCamera(slmaster::CameraType(cameraType_)); }
    std::vector<OrderTableRecord> getOrderTableRecord() { return orderTableRecord_; }
  signals:
    void stripeImgsChanged(const int num);
    void frameCaptured();
  private:
    CameraEngine();
    ~CameraEngine();
    CameraEngine(const CameraEngine&) = delete;
    const CameraEngine& operator=(const CameraEngine&) = delete;
    void defocusStripeCreate(std::vector<cv::Mat>& imgs, const int direction, const int cycles, const int shiftTime, AppType::DefocusEncoding method);
    void realTimeRenderImg(const QImage& img);
    void createTenLine();
    void switchTrigMode(const bool isTrigLine, const int exposureTime);
    std::vector<OrderTableRecord> orderTableRecord_;
    std::vector<QImage> stripeImgs_;
    AppType::ScanModeType scanMode_;
    AppType::CameraType cameraType_;
    AppType::PatternMethod patternType_;
    static CameraEngine* engine_;
    ImagePaintItem* stripePaintItem_;
    ImagePaintItem* offlineCamPaintItem_ = nullptr;
    ImagePaintItem* scanTexturePaintItem_ = nullptr;
    std::thread onlineDetectThread_;
    std::thread workThread_;
    slmaster::SLCameraFactory slCameraFactory_;
    slmaster::Pattern* pattern_ = nullptr;
    CameraModel* leftCamModel_ = nullptr;
    CameraModel* rightCamModel_ = nullptr;
    CameraModel* colorCamModel_ = nullptr;
    slmaster::FrameData frame_;
    std::thread test_thread_;
    std::atomic_bool appExit_;
    std::atomic_bool isProject_;
    SafeQueue<slmaster::FrameData> frameDatasQueue_;
    std::atomic_bool isContinusStop_;
};

#endif// !__CAMERA_ENGINE_H_
