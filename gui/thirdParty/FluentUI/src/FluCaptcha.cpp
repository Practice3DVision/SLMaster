#include "FluCaptcha.h"
#include <QTime>
#include <QChar>
#include <QPainter>
#include <QRandomGenerator>
#include <qmath.h>

FluCaptcha::FluCaptcha(QQuickItem *parent):QQuickPaintedItem(parent){
    ignoreCase(true);
    QFont fontStype;
    fontStype.setPixelSize(28);
    fontStype.setBold(true);
    font(fontStype);
    setWidth(180);
    setHeight(80);
    refresh();
}

void FluCaptcha::paint(QPainter* painter){
    painter->save();
    painter->fillRect(boundingRect().toRect(),QColor(255,255,255,255));
    QPen pen;
    painter->setFont(_font);
    for(int i=0;i<100;i++)
    {
        pen = QPen(QColor(_generaNumber(256),_generaNumber(256),_generaNumber(256)));
        painter->setPen(pen);
        painter->drawPoint(_generaNumber(180),_generaNumber(80));
    }
    for(int i=0;i<5;i++)
    {
        pen = QPen(QColor(_generaNumber(256),_generaNumber(256),_generaNumber(256)));
        painter->setPen(pen);
        painter->drawLine(_generaNumber(180),_generaNumber(80),_generaNumber(180),_generaNumber(80));
    }
    for(int i=0;i<4;i++)
    {
        pen = QPen(QColor(_generaNumber(255),_generaNumber(255),_generaNumber(255)));
        painter->setPen(pen);
        painter->drawText(15+35*i,10+_generaNumber(15),30,40,Qt::AlignCenter, QString(_code[i]));
    }
    painter->restore();
}

int FluCaptcha::_generaNumber(int number){
    return QRandomGenerator::global()->bounded(0,number);
}

void FluCaptcha::refresh(){
    this->_code.clear();
    for(int i = 0;i < 4;++i)
    {
        int num = _generaNumber(3);
        if(num == 0)
        {
            this->_code += QString::number(_generaNumber(10));
        }
        else if(num == 1)
        {
            int temp = 'A';
            this->_code += static_cast<QChar>(temp + _generaNumber(26));
        }
        else if(num == 2)
        {
            int temp = 'a';
            this->_code += static_cast<QChar>(temp + _generaNumber(26));
        }
    }
    update();
}

bool FluCaptcha::verify(const QString& code){
    if(_ignoreCase){
        return this->_code.toUpper() == code.toUpper();
    }
    return this->_code == code;
}
