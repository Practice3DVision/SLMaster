#include "ImagePaintItem.h"

ImagePaintItem::ImagePaintItem(QQuickItem *parent) : QQuickItem(parent), hasRendered_(false), isRendererFinished_(true) {
    setFlag(ItemHasContents, true);

    connect(this, &ImagePaintItem::colorChanged, this, &ImagePaintItem::updateBackground);
    connect(this, &ImagePaintItem::renderImg, this, &ImagePaintItem::render, Qt::AutoConnection);
}

void ImagePaintItem::render() {
    this->update();
}

void ImagePaintItem::updateBackground() {
    if(hasRendered_)
        return;

    imageThumb_ = QImage(200, 200, QImage::Format_RGB888);
    imageThumb_.fill(color_);

    emit renderImg();
}

void ImagePaintItem::updateImage(const QImage &image) {
    hasRendered_ = true;

    if(isRendererFinished_.load(std::memory_order_acquire)) {
        imageThumb_ = image;
        isRendererFinished_.store(false, std::memory_order_release);

        emit renderImg();
    }
}

QSGNode *ImagePaintItem::updatePaintNode(QSGNode *oldNode,
                                    QQuickItem::UpdatePaintNodeData *) {
    auto node = dynamic_cast<QSGSimpleTextureNode *>(oldNode);

    if (!node)
        node = new QSGSimpleTextureNode();

    QSGTexture *texture = window()->createTextureFromImage(imageThumb_, QQuickWindow::TextureIsOpaque);

    if(!texture)
        return node;
    node->setOwnsTexture(true);
    const int height = boundingRect().height();
    const int widht = boundingRect().height() * (float)texture->textureSize().width() / texture->textureSize().height();
    node->setRect((boundingRect().width() - widht) / 2, 0, widht, height);
    node->markDirty(QSGNode::DirtyForceUpdate);
    node->setTexture(texture);

    isRendererFinished_.store(true, std::memory_order_release);

    return node;
}
