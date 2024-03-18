#ifndef __SETTINGS_HELPER_H_
#define __SETTINGS_HELPER_H_

#include <memory>

#include <QObject>
#include <QVariant>
#include <QSettings>

class SettingsHelper : public QObject {
    Q_OBJECT
  public:
    static SettingsHelper *instance();
    Q_INVOKABLE void init(const QString& iniFileName);
    Q_INVOKABLE void saveRender(const QVariant& render){setValue("render",render);}
    Q_INVOKABLE QVariant getRender(){return getValue("render", QVariant("software"));}
    Q_INVOKABLE void saveDarkMode(const int darkModel){setValue("darkMode",darkModel);}
    Q_INVOKABLE QVariant getDarkMode(){return getValue("darkMode",QVariant(0));}
    Q_INVOKABLE void saveVsync(const bool vsync){setValue("vsync",vsync);}
    Q_INVOKABLE QVariant getVsync(){return getValue("vsync",QVariant(true));}
    Q_INVOKABLE void saveUseSystemAppBar(const bool useSystemAppBar){setValue("useSystemAppBar",useSystemAppBar);}
    Q_INVOKABLE QVariant getUseSystemAppBar(){return getValue("useSystemAppBar",QVariant(false));}
  private:
    SettingsHelper(QObject* parent = nullptr) {}
    ~SettingsHelper() {}
    SettingsHelper(const SettingsHelper&) = delete;
    const SettingsHelper& opertor(const SettingsHelper&) = delete;

     void setValue(const QString &key, const QVariant &value);

     QVariant getValue(
        const QString &key, const QVariant &defaultValue = QVariant()) const;
    std::unique_ptr<QSettings> settingsHandle_;
    static SettingsHelper* helper_;
};

#endif //! __SETTINGS_HELPER_H_
