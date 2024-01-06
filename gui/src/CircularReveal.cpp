#include "CircularReveal.h"
#include <QGuiApplication>
#include <QQuickItemGrabResult>
#include <QPainterPath>

CircularReveal::CircularReveal(QQuickItem* parent) : QQuickPaintedItem(parent), anim_(this, "radius", this)
{
    setVisible(false);
    anim_.setDuration(333);
    anim_.setEasingCurve(QEasingCurve::OutCubic);
    connect(&anim_, &QPropertyAnimation::finished,this,[=](){
        update();
        setVisible(false);
        Q_EMIT animationFinished();
    });
    connect(this,&CircularReveal::radiusChanged,this,[=](){
        update();
    });
}

void CircularReveal::paint(QPainter* painter)
{
    painter->save();
    painter->drawImage(QRect(0, 0, static_cast<int>(width()), static_cast<int>(height())), source_);
    QPainterPath path;
    path.moveTo(center_.x(),center_.y());
    path.addEllipse(QPointF(center_.x(),center_.y()), radius_, radius_);
    painter->setCompositionMode(QPainter::CompositionMode_Clear);
    painter->fillPath(path, Qt::black);
    painter->restore();
}

void CircularReveal::start(int w,int h,const QPoint& center,int radius){
    anim_.setStartValue(0);
    anim_.setEndValue(radius);
    center_ = center;
    grabResult_ = target_->grabToImage(QSize(w,h));
    connect(grabResult_.data(), &QQuickItemGrabResult::ready, this, &CircularReveal::handleGrabResult);
}

void CircularReveal::handleGrabResult(){
    grabResult_.data()->image().swap(source_);
    update();
    setVisible(true);
    Q_EMIT imageChanged();
    anim_.start();
}
