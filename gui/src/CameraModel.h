/**
 * @file CameraModel.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef CAMERAMODEL_H
#define CAMERAMODEL_H

#include <QAbstractListModel>
#include <QObject>
#include <QUrl>
#include <iostream>

#include <opencv2/opencv.hpp>

class CameraModel : public QAbstractListModel {
    Q_OBJECT
  public:
    enum INFO_TYPE { FileName = Qt::UserRole + 1 };
    explicit CameraModel(QObject *parent = nullptr);
    ~CameraModel();
    QHash<int, QByteArray> roleNames() const override;
    int rowCount(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    Q_INVOKABLE void recurseImg(const QString &folderUrl);
    Q_INVOKABLE void emplace_back(const QString &fileName);
    Q_INVOKABLE int erase(const QString &fileName);
    Q_INVOKABLE void erase(const int locIndex);
    Q_INVOKABLE const QList<QString> imgPaths() { return imgs_; }
    Q_INVOKABLE const QString curFolderPath() { return curFolderPath_; }
  signals:
    void updateImgs();
  private:
    QString curFolderPath_;
    QList<QString> imgs_;
    QHash<int, QByteArray> roleNames_;
};

#endif // CAMERAMODEL_H
