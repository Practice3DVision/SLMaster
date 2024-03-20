#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickWindow>

#include "AppType.h"
#include "ImagePaintItem.h"
#include "CircularReveal.h"
#include "CameraEngine.h"
#include "CameraModel.h"
#include "CalibrateEngine.h"
#include "VtkRenderItem.h"
#include "VtkProcessEngine.h"
#include "logger.h"
#include "settingsHelper.h"

#include "QuickQanava"
#include "flowNode.h"
#include "flowGraph.h"
#include "cloudInputNode.h"
#include "CloudOutputNode.h"

#include <vtkOutputWindow.h>

#ifdef FLUENTUI_BUILD_STATIC_LIB
#if (QT_VERSION > QT_VERSION_CHECK(6, 2, 0))
Q_IMPORT_QML_PLUGIN(FluentUIPlugin)
#endif
#include <FluentUI.h>
#endif

int main(int argc, char *argv[])
{
    logger::setUp("SLMasterGui");
#if (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
    QGuiApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QGuiApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
    QGuiApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);
#endif
#endif
    qputenv("QT_QUICK_CONTROLS_STYLE","Basic");
    QGuiApplication::setWindowIcon(QIcon("qrc:/res/image/icons8-maple-leaf-48.ico"));
    QGuiApplication::setOrganizationName("YunhuangLiu");
    QGuiApplication::setOrganizationDomain("https://github.com/Yunhuang-Liu");
    QGuiApplication::setApplicationName("SLMasterGui");
    SettingsHelper::instance()->init("SLMasterGui");
    if(SettingsHelper::instance()->getRender() == "SoftWare") {
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
        QQuickWindow::setGraphicsApi(QSGRendererInterface::OpenGL);
#elif (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
        QQuickWindow::setSceneGraphBackend(QSGRendererInterface::OpenGL);
#endif
    }

#ifdef WITH_CUDASTRUCTUREDLIGHT_MODULE
    auto deviceNum = cv::cuda::getCudaEnabledDeviceCount();
    qDebug() << "CUDA Device nums: " << deviceNum;
    if(deviceNum > 0) {
        cv::cuda::setDevice(0);
    }
    else {
#undef WITH_CUDASTRUCTUREDLIGHT_MODULE
        qDebug() << "CUDA diasble." << deviceNum;
    }
#endif    

    QQuickVTKRenderWindow::setupGraphicsBackend();
    vtkOutputWindow::SetGlobalWarningDisplay(0);

    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    QuickQanava::initialize(&engine);
    //TODO@Evans Liu: 转换为绝对路径或者拷贝临时文件
    //slmaster::BinocularCamera camera("../../gui/qml/res/config/binoocularCameraConfig.json");
    //camera.resetCameraConfig();
    //camera.updateCamera();
    CameraEngine::instance()->setCameraJsonPath("../../gui/qml/res/config/binoocularCameraConfig.json");
    CameraEngine::instance()->startDetectCameraState();

    engine.rootContext()->setContextProperty("SettingsHelper", SettingsHelper::instance());
    qmlRegisterType<CircularReveal>("SLMasterGui", 1, 0, "CircularReveal");
    qmlRegisterType<AppType>("SLMasterGui", 1, 0, "AppType");
    qmlRegisterType<ImagePaintItem>("SLMasterGui", 1, 0, "ImagePaintItem");
    qmlRegisterType<CameraModel>("SLMasterGui", 1, 0, "CameraModel");
    qmlRegisterType<QQuickVTKRenderWindow>("SLMasterGui", 1, 0, "VTKRenderWindow");
    qmlRegisterType<VTKRenderItem>("SLMasterGui", 1, 0, "VTKRenderItem");
    qmlRegisterSingletonInstance("SLMasterGui", 1, 0, "VTKProcessEngine", VTKProcessEngine::instance());
    qmlRegisterSingletonInstance("SLMasterGui", 1, 0, "CameraEngine", CameraEngine::instance());
    qmlRegisterSingletonInstance("SLMasterGui", 1, 0, "CalibrateEngine", CalibrateEngine::instance());
    qmlRegisterType<FlowNode>("SLMasterGui", 1, 0, "FlowNode");
    qmlRegisterType<FlowGraph>("SLMasterGui", 1, 0, "FlowGraph");
    qmlRegisterType<CloudInputNode>("SLMasterGui", 1, 0, "CloudInputNode");
    qmlRegisterType<CloudOutputNode>("SLMasterGui", 1, 0, "CloudOutputNode");

#ifdef FLUENTUI_BUILD_STATIC_LIB
    FluentUI::getInstance()->registerTypes(&engine);
#endif
    qDebug()<< engine.importPathList();

    const QUrl url(QStringLiteral("qrc:/ui/App.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
        &app, [url](QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        }, Qt::QueuedConnection);
    engine.load(url);

    VTKProcessEngine::instance()->bindEngine(&engine);

    int exitCode = QGuiApplication::exec();

    CameraEngine::instance()->getSLCamera()->updateCamera();

    return exitCode;
}
