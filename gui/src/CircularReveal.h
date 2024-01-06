#ifndef CIRCULARREVEAL_H
#define CIRCULARREVEAL_H

#include <QQuickItem>
#include <QQuickPaintedItem>
#include <QPainter>
#include <QPropertyAnimation>

#include "typeDef.h"

class CircularReveal : public QQuickPaintedItem
{
    Q_OBJECT
    Q_PROPERTY_AUTO(QQuickItem*,target)
    Q_PROPERTY_AUTO(int,radius)
public:
    CircularReveal(QQuickItem* parent = nullptr);
    void paint(QPainter* painter) override;
    Q_INVOKABLE void start(int w,int h,const QPoint& center,int radius);
    Q_SIGNAL void imageChanged();
    Q_SIGNAL void animationFinished();
    Q_SLOT void handleGrabResult();
private:
    QImage source_;
    QPropertyAnimation anim_;
    QPoint center_;
    QSharedPointer<QQuickItemGrabResult>  grabResult_;
};

#endif // CIRCULARREVEAL_H
