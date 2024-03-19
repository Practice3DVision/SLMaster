/**
 * @file cloudInputNode.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __CLOUD_INPUT_NODE_H_
#define __CLOUD_INPUT_NODE_H_

#include <thread>

#include "QuickQanava.h"

#include "flowNode.h"

class CloudInputNode : public FlowNode {
    Q_OBJECT
    Q_PROPERTY(CloudInputMode mode READ mode WRITE setMode NOTIFY modeChanged FINAL)
    Q_PROPERTY(QString filePath READ filePath WRITE setFilePath NOTIFY filePathChanged FINAL)
  public:
    enum class CloudInputMode {
      FromCamera = 0,
      FromFile,
    };
    Q_ENUM(CloudInputMode)
    CloudInputNode();
    ~CloudInputNode();
    Q_INVOKABLE void updateSource();
    Q_INVOKABLE CloudInputMode mode() { return mode_; }
    Q_INVOKABLE void setMode(const CloudInputMode mode);
    Q_INVOKABLE QString filePath() { return filePath_; }
    Q_INVOKABLE void setFilePath(const QString& filePath);
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
  signals:
    void modeChanged(const CloudInputMode mode);
    void filePathChanged(const QString& filePath);
  private:
    bool readPlyFile(QString path);
    CloudInputMode mode_;
    QString filePath_;
    std::thread workThread_;
};

QML_DECLARE_TYPE(CloudInputNode)
Q_DECLARE_METATYPE(CloudInputNode::CloudInputMode)

#endif // !__CLOUD_INPUT_NODE_H_
