#ifndef __PROJECTORY_FACTORY_H_
#define __PROJECTORY_FACTORY_H_

#include <string>
#include <unordered_map>

#include "projector.h"
#include "projectorDlpc34xx.h"
#include "projectorDlpc34xxDual.h"

#include "typeDef.h"

/** @brief 结构光库 **/
namespace device {
/** @brief 投影仪库 **/
namespace projector {
/** @brief 投影仪工厂 **/
class DEVICE_API ProjectorFactory {
  public:
    ProjectorFactory() { };

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
            //TODO@LiuYunhuang:增加DLP6500支持
        }

        return projector;
    }
  private:
    std::unordered_map<std::string, Projector *> projectoies_;
}; // class ProjectorFactory
} // namespace projector
} // namespace device

#endif //__PROJECTORY_FACTORY_H_
