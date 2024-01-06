#ifndef FLURECTANGLE_H
#define FLURECTANGLE_H

#include <QQuickItem>
#include <QQuickPaintedItem>
#include <QPainter>
#include "stdafx.h"

class FluRectangle : public QQuickPaintedItem
{
    Q_OBJECT
    Q_PROPERTY_AUTO(QColor,color)
    Q_PROPERTY_AUTO(QList<int>,radius)
    QML_NAMED_ELEMENT(FluRectangle)
public:
    explicit FluRectangle(QQuickItem *parent = nullptr);
    void paint(QPainter* painter) override;
};

#endif // FLURECTANGLE_H
