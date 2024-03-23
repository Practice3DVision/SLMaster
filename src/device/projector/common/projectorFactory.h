/**
 * @file projectorFactory.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __PROJECTORY_FACTORY_H_
#define __PROJECTORY_FACTORY_H_

#include <string>
#include <unordered_map>

#include "projector.h"
#include "projectorDlpc34xx.h"
#include "projectorDlpc34xxDual.h"

#include "typeDef.h"

/** @brief 结构光库 **/
namespace slmaster {
/** @brief 设备库 **/
namespace device {
/** @brief 投影仪工厂 **/
class DEVICE_API ProjectorFactory {
  public:
    ProjectorFactory(){};

    Projector *getProjector(const std::string dlpEvm) {
        Projector *projector = nullptr;

        if (projectoies_.count(dlpEvm)) {
            return projectoies_[dlpEvm];
        } else {
            if ("DLP4710" == dlpEvm) {
                projector = new ProjectorDlpc34xxDual();
                projectoies_[dlpEvm] = projector;
            }

            else if ("DLP3010" == dlpEvm) {
                projector = new ProjectorDlpc34xx();
                projectoies_[dlpEvm] = projector;
            }
            // TODO@Evans Liu:增加DLP6500支持
        }

        return projector;
    }

  private:
    std::unordered_map<std::string, Projector *> projectoies_;
}; // class ProjectorFactory
} // namespace device
} // namespace slmaster

#endif //__PROJECTORY_FACTORY_H_
