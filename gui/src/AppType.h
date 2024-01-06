#ifndef __APP_TYPE_H_
#define __APP_TYPE_H_

#include <QObject>

class AppType : public QObject
{
    Q_OBJECT
  public:
    enum PageType {
        Device,
        Calibration,
        ScanMode,
        Scan,
        PostProcess,
        Measurement
    };

    enum CameraType {
        MonocularSLCamera = 0,
        BinocularSLCamera,
        TripleSLCamera
    };

    enum PixelDepth {
        OneBit = 0,
        EightBit
    };

    enum Direction {
        Horizion = 0,
        Vertical
    };

    enum StripeType {
        SineComplementaryGrayCode = 0,
    };

    enum DefocusEncoding {
        Disable = 0,
        Binary,
        ErrorDiffusionMethod,
        OptimalPlusWithModulation,
    };

    enum ConnectState {
        Disconnect = 0,
        Connect,
    };

    enum CameraFolderType {
        Left = 0,
        Right,
        Color
    };

    enum CaliType {
        Single = 0,
        Stereo,
        Triple,
        Projector,
    };

    enum TargetType {
        ChessBoard = 0,
        Blob,
        ConcentricCircle,
    };

    enum ProjectorCaliType {
        Intrinsic = 0,
        Extrinsic,
    };

    enum ScanModeType {
        Offline = 0,
        Static,
        Dynamic
    };

    enum PatternMethod {
        SinusCompleGrayCode = 0,
        MutiplyFrequency,
        MultiViewStereoGeometry,
    };

    Q_ENUM(PageType)
    Q_ENUM(CameraType)
    Q_ENUM(PixelDepth)
    Q_ENUM(Direction)
    Q_ENUM(StripeType)
    Q_ENUM(DefocusEncoding)
    Q_ENUM(ConnectState)
    Q_ENUM(CameraFolderType)
    Q_ENUM(CaliType)
    Q_ENUM(TargetType)
    Q_ENUM(ProjectorCaliType)
    Q_ENUM(ScanModeType)
    Q_ENUM(PatternMethod)

    explicit AppType(QObject *parent = nullptr) : QObject(parent) {};
};

#endif// !__APP_TYPE_H_
