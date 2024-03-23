/**
 * @file projectorDlpc34xxDual.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __PROJECTOR_DLPC_34xx_DUAL_H_
#define __PROJECTOR_DLPC_34xx_DUAL_H_

#include "projector.h"

#include "dlpc347x_internal_patterns.h"
#include "dlpc_common.h"

#include <time.h>

/** @brief slmaster **/
namespace slmaster {
/** @brief 设备库 **/
namespace device {
/** @brief DLPC34xx dual系列投影仪 */
class DEVICE_API ProjectorDlpc34xxDual : public Projector {
  public:
    ProjectorDlpc34xxDual();
    ~ProjectorDlpc34xxDual();
    /**
     * @brief 获取投影仪信息
     *
     * @return ProjectorInfo 投影仪相关信息
     */
    ProjectorInfo getInfo() override;
    /**
     * @brief 连接
     *
     * @return true 成功
     * @return false 失败
     */
    bool connect() override;
    /**
     * @brief 断开连接
     *
     * @return true 成功
     * @return false 失败
     */
    bool disConnect() override;
    /**
     * @brief 断开连接
     *
     * @return true 成功
     * @return false 失败
     */
    bool isConnect() override;
    /**
     * @brief 从图案集制作投影序列
     *
     * @param table 投影图案集
     */
    bool
    populatePatternTableData(IN std::vector<PatternOrderSet> table) override;
    /**
     * @brief 投影
     *
     * @param isContinue 是否连续投影
     * @return true 成功
     * @return false 失败
     */
    bool project(IN const bool isContinue) override;
    /**
     * @brief 停止
     *
     * @return true 成功
     * @return false 失败
     */
    bool stop() override;
    /**
     * @brief 恢复投影
     *
     * @return true 成功
     * @return false 失败
     */
    bool resume() override;
    /**
     * @brief 暂停
     *
     * @return true 成功
     * @return false 失败
     */
    bool pause() override;
    /**
     * @brief 投影下一帧
     * @warning 仅在步进模式下使用
     *
     * @return true
     * @return false
     */
    bool step() override;
    /**
     * @brief 获取当前LED三色灯电流值
     *
     * @param r 红色电流值
     * @param g 绿色电流值
     * @param b 蓝色电流值
     * @return true 成功
     * @return false 失败
     */
    bool getLEDCurrent(OUT double &r, OUT double &g, OUT double &b) override;
    /**
     * @brief 设置当前LED三色灯电流值
     *
     * @param r 红色电流值
     * @param g 绿色电流值
     * @param b 蓝色电流值
     * @return true 成功
     * @return false 失败
     */
    bool setLEDCurrent(IN const double r, IN const double g,
                       IN const double b) override;
    /**
     * @brief 获取当前闪存图片数量
     *
     * @return int 图片数量
     */
    int getFlashImgsNum() override;
  private:
    /**
     * @brief 初始化Cypress USB-Serial和DLPC控制器
     *
     * @return bool 成功初始化
     */
    bool initConnectionAndCommandLayer();
    /**
     * @brief 从flash加载图案序列
     *
     */
    void loadPatternOrderTableEntryFromFlash();
    //是否已成功初始化
    bool isInitial_;
    //投影仪幅面列数
    uint16_t cols_;
    //投影仪幅面行数
    uint16_t rows_;
    //图案数量
    int numOfPatterns_;
    //图案集合数量
    int numOfPatternSets_;
};
} // namespace device
} // namespace slmaster

#endif // !__PROJECTOR_DLPC_34xx_DUAL_H_
