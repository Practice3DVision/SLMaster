#ifndef CALIBRATEENGINE_H
#define CALIBRATEENGINE_H

#define LinearInter

#include <filesystem>
#include <thread>

#include <QDir>
#include <QUrl>
#include <QObject>
#include <QString>
#include <QFileInfo>
#include <QVector4D>

#include "caliPacker.h"
#include "chessBoardCalibrator.h"
#include "circleGridCalibrator.h"
#include "concentricRingCalibrator.h"
#include "ImagePaintItem.h"
#include "CameraModel.h"

class CalibrateEngine : public QObject {
    Q_OBJECT
    Q_PROPERTY(float progress READ progress NOTIFY progressChanged)
  public:
    enum BoardType { chess = 0, circleGrid, concentricCircle };
    static CalibrateEngine* instance();
    Q_INVOKABLE void setCurCaliType(const int caliType) { curCaliType_ = AppType::CaliType(caliType); }
    Q_INVOKABLE void singleCalibrate(const int targetType, const int rowNum, const int colNum,
                                 const float trueDistance);
    Q_INVOKABLE void stereoCalibrate(const int targetType, const int rowNum, const int colNum,
                                       const float trueDistance, const bool exportEpilorLine);
    Q_INVOKABLE void setConcentricCirclesRadius(const float innerCircleInnerRadius_, const float innerCircleExterRadius_, const float exterCircleInnerRadius_, const float exterCircleExterRadius_) {
        concentricRingParams_.innerCircleInnerRadius_ = innerCircleInnerRadius_;
        concentricRingParams_.innerCircleExterRadius_ = innerCircleExterRadius_;
        concentricRingParams_.exterCircleInnerRadius_ = exterCircleInnerRadius_;
        concentricRingParams_.exterCircleExterRadius_ = exterCircleExterRadius_;
    }
    Q_INVOKABLE void updateDisplayImg(const QString& imgPath);
    Q_INVOKABLE const QVariantList updateErrorDistribute(const QString& imgPath, const bool isLeft);
    Q_INVOKABLE void displayStereoRectifyMap();
    Q_INVOKABLE void exit();
    Q_INVOKABLE void bindOfflineCamPaintItem(ImagePaintItem* item) { offlineCamPaintItem_ = item; };
    Q_INVOKABLE void bindOnlineProjectorPaintItem(ImagePaintItem* colorItem, ImagePaintItem* honrizonItem, ImagePaintItem* verticalItem, ImagePaintItem* projectorItem) {
        onlineProjColorPaintItem_ = colorItem;
        onlineProjHonriPaintItem_ = honrizonItem;
        onlineProjVertiPaintItem_ = verticalItem;
        onlineProjPaintItem_ = projectorItem;
    };
    Q_INVOKABLE void bindLeftCameraModel(CameraModel* cameraModel) { leftCameraModel_ = cameraModel; connect(leftCameraModel_, &CameraModel::updateImgs, [&]{leftCalibrator_.reset();});};
    Q_INVOKABLE void bindRightCameraModel(CameraModel* cameraModel) { rightCameraModel_ = cameraModel; connect(rightCameraModel_, &CameraModel::updateImgs, [&]{rightCalibrator_.reset();});};
    Q_INVOKABLE void bindProjectorModel(CameraModel* projectorModel) { projectorModel_ = projectorModel; };
    Q_INVOKABLE void saveCaliInfo(const QString& path);
    Q_INVOKABLE void setProjectorCaliParams(const int targetType, const int rowNum, const int colNum,
                                            const float trueDistance, const float honrizonPitch, const float verticalPitch, const int projCaliType) {
        projectorCaliParams_.targetType_ = AppType::TargetType(targetType);
        projectorCaliParams_.projCaliType_ = AppType::ProjectorCaliType(projCaliType);
        projectorCaliParams_.rowNum_ = rowNum;
        projectorCaliParams_.colNum_ = colNum;
        projectorCaliParams_.trueDistance_ = trueDistance;
        projectorCaliParams_.honrizonPitch_ = honrizonPitch;
        projectorCaliParams_.verticalPitch_ = verticalPitch;
    }
    Q_INVOKABLE void removeProjectImg(const QString& path);
    Q_INVOKABLE bool captureOnce();
    Q_INVOKABLE double calibrateProjector();
    Q_INVOKABLE void readLocalCaliFile(const QString& path);
    float progress() { return progress_; }
  signals:
    void progressChanged(float progress);
    void drawKeyPointsChanged(const QString path);
    void errorReturn(const double error);
    void projectErrorReturn(const double error);
    void findFeaturePointsError(const QString path);
    void projectorModelChanged();
  private:
    struct ConcentricRingParams {
        float innerCircleInnerRadius_;
        float innerCircleExterRadius_;
        float exterCircleInnerRadius_;
        float exterCircleExterRadius_;
    } concentricRingParams_;
    struct ProjectorCaliParams {
        AppType::TargetType targetType_;
        AppType::ProjectorCaliType projCaliType_;
        int rowNum_;
        int colNum_;
        float honrizonPitch_;
        float verticalPitch_;
        float trueDistance_;
    } projectorCaliParams_;
    CalibrateEngine();
    ~CalibrateEngine() = default;
    CalibrateEngine(const CalibrateEngine&) = default;
    CalibrateEngine& operator=(const CalibrateEngine&) = default;
    Calibrator* getCalibrator(const AppType::TargetType targetType);
    void findEpilines(const int rows, const int cols,
                      const cv::Mat &fundermental, cv::Mat &epilines);
    void rectify(const cv::Mat &leftImg, const cv::Mat &rightImg,
                 const slmaster::CaliInfo &info, cv::Mat &rectifyImg);
    const cv::Point2f remapProjectorPoint(const cv::Mat& honrizonPhaseMap, const cv::Mat& verticalPhaseMap, const cv::Point2f& camPoint);
    static CalibrateEngine* calibrateEngine_;
    ImagePaintItem* offlineCamPaintItem_;
    ImagePaintItem* onlineProjColorPaintItem_;
    ImagePaintItem* onlineProjHonriPaintItem_;
    ImagePaintItem* onlineProjVertiPaintItem_;
    ImagePaintItem* onlineProjPaintItem_;
    float progress_;
    int imgProcessIndex_;
    std::unique_ptr<Calibrator> leftCalibrator_;
    std::unique_ptr<Calibrator> rightCalibrator_;
    std::atomic_bool isFinish_;
    std::thread updateThread_;
    std::thread calibrationThread_;
    AppType::CaliType curCaliType_;
    CameraModel* leftCameraModel_;
    CameraModel* rightCameraModel_;
    CameraModel* projectorModel_;
    slmaster::CaliInfo caliInfo_;
    cv::Mat rectifiedImg_;
    std::vector<std::vector<cv::Mat>> projectorCaliImgs_;
    std::vector<std::vector<cv::Point2f>> projCameraPoints_;
    std::vector<std::vector<cv::Point2f>> projectorPoints_;
    std::vector<std::vector<cv::Point2f>> projectorErrorDistributes_;
  public slots:
    void timer_timeout_slot();
};

#endif // CALIBRATEENGINE_H
