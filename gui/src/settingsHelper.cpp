#include "settingsHelper.h"

#include <QDebug>
#include <QStandardPaths>

using namespace std;

SettingsHelper* SettingsHelper::helper_ = new SettingsHelper();

SettingsHelper *SettingsHelper::instance() { 

    return helper_;
}

void SettingsHelper::init(const QString& iniFileName) {
    auto dataPath =
        QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) +
        "/" + iniFileName + ".ini";

    qInfo() << "Ini settings file path is: " + dataPath;

    settingsHandle_ = make_unique<QSettings>(dataPath, QSettings::Format::IniFormat, this);

    qInfo() << "load settings file sucess.";
}

void SettingsHelper::setValue(const QString &key, const QVariant &value) {
    Q_ASSERT(settingsHandle_);

    if (key.isEmpty() || !value.isValid()) {
        qWarning() << "setting's key is empty or value is invalid.";
    }

    settingsHandle_->setValue(key, value);
    settingsHandle_->sync();
}

QVariant SettingsHelper::getValue(const QString &key,
                                  const QVariant &defaultValue) const {
    Q_ASSERT(settingsHandle_);

    if (key.isEmpty()) {
        qWarning() << "setting's key is empty or value is invalid.";
    }

    if (!settingsHandle_->contains(key)) {
        qWarning() << "setting file dosn't contain this key: " + key + ", return default val: " + defaultValue.toString();
    }

    return settingsHandle_->value(key, defaultValue);
}
