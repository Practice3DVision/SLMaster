/**
 * @file VtkRenderItem.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __VTK_RENDER_ITEM_H_
#define __VTK_RENDER_ITEM_H_

#include <future>
#include <queue>

#include <QQuickVTKRenderItem.h>
#include <QQuickVTKRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkUnsignedCharArray.h>

class VTKRenderItem : public QQuickVTKRenderItem {
    Q_OBJECT
    Q_PROPERTY(double fps READ fps NOTIFY fpsChanged FINAL)
public:
    VTKRenderItem();
    ~VTKRenderItem() = default;
    template<class F, class... Args>
    std::future<void>& pushCommandToQueue(F&& f, Args&&... args)
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared< std::packaged_task<return_type()> >(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                    );

        std::future<return_type> res = task->get_future();
        {
            commandQueue_.emplace([task]() { (*task)(); });
        }
        return res;
    }
    Q_INVOKABLE double fps();
signals:
    void fpsChanged();
public slots:
    void sync() override;
private:
    std::queue<std::function<void()>> commandQueue_;
};

#endif //__VTK_RENDER_ITEM_H_
