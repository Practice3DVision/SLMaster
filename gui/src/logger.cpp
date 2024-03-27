#include "logger.h"

#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QString>

#include <iostream>

using namespace std;
namespace logger {

unique_ptr<QFile> fileHandle = nullptr;
unique_ptr<QTextStream> textStream = nullptr;

static inline void myMessageHandler(const QtMsgType type,
                                    const QMessageLogContext &context,
                                    const QString &message) {
    if (message.isEmpty() || type == QtWarningMsg)
        return;

    const QString formatMessage =
        qFormatLogMessage(type, context, message).trimmed();

    if (type == QtInfoMsg || type == QtDebugMsg) {
        std::cout << qUtf8Printable(message) << std::endl;
    } 
    /*
    else {
        std::cerr << qUtf8Printable(message) << std::endl;
    }
    */
    if (textStream && fileHandle) {
        *textStream << formatMessage << Qt::endl;
    }
}

void setUp(const QString &appName) {
    Q_ASSERT(!appName.isEmpty());

    if (!fileHandle || !textStream) {
        fileHandle = make_unique<QFile>(QString("debug-%1.log").arg(appName));
        if (!fileHandle->open(QIODevice::ReadWrite | QIODevice::Text |
                              QIODevice::Append | QIODevice::Truncate)) {
            fileHandle->reset();
        } else {
            textStream = make_unique<QTextStream>();
            textStream->setDevice(fileHandle.get());
        }
    }

    qSetMessagePattern(QString(
        "[%{time yyyy/MM/dd hh:mm:ss.zzz}] "
        "<%{if-info}INFO%{endif}%{if-debug}DEBUG"
        "%{endif}%{if-warning}WARNING%{endif}%{if-critical}CRITICAL%{endif}%{"
        "if-fatal}"
        "FATAL%{endif}> %{if-category}%{category}: %{endif}%{message}"));
    qInstallMessageHandler(myMessageHandler);
}
}// namespace logger
