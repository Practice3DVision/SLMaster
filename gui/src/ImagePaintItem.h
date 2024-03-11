#ifndef __IMAGE_PAINTITEM_H_
#define __IMAGE_PAINTITEM_H_

#include "typeDef.h"

#include <QImage>
#include <QQuickItem>
#include <QQuickWindow>
#include <QSGNode>
#include <QSGSimpleRectNode>
#include <QSGSimpleTextureNode>

class ImagePaintItem : public QQuickItem {
    Q_OBJECT
  public:
    explicit ImagePaintItem(QQuickItem *parent = nullptr);
    Q_PROPERTY_AUTO(QColor, color);
  public slots:
    void updateImage(const QImage &image);
    void updateBackground();
  signals:
    void renderImg();
  protected:
    QSGNode *updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) override;

  private:
    void render();
    bool hasRendered_;
    QImage imageThumb_;
    std::atomic_bool isRendererFinished_;
};

#endif // __IMAGE_PAINTITEM_H_
