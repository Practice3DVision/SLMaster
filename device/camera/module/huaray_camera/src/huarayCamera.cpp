#include "huarayCamera.h"

#include <chrono>

namespace device {
namespace camera {

//相机取流回调函数
static void frameCallback(IMV_Frame *pFrame, void *pUser) {
    HuarayCammera *pCamera = reinterpret_cast<HuarayCammera *>(pUser);
    if (pFrame->pData != NULL) {
        //TODO@LiuYunhuang:增加各种格式的pack支持

        cv::Mat img;
        if(_IMV_EPixelType::gvspPixelMono8 == pFrame->frameInfo.pixelFormat) {
            img = cv::Mat(pFrame->frameInfo.height, pFrame->frameInfo.width, CV_8U, pFrame->pData).clone();
        }
        else if(_IMV_EPixelType::gvspPixelBayRG8 == pFrame->frameInfo.pixelFormat) {
            //TODO@LiuYunhuang:如果直接在这进行转码，将导致延迟
            img = cv::Mat(pFrame->frameInfo.height, pFrame->frameInfo.width, CV_8U, pFrame->pData).clone();
        }
        else if(_IMV_EPixelType::gvspPixelMono16 == pFrame->frameInfo.pixelFormat) {
            img = cv::Mat(pFrame->frameInfo.height, pFrame->frameInfo.width,
                CV_16U, pFrame->pData)
                .clone();
        }

        pCamera->getImgs().push(std::move(img));
    }
}

HuarayCammera::HuarayCammera(const std::string cameraUserId)
    : cameraUserId_(cameraUserId), pCamera_(nullptr) {}

HuarayCammera::~HuarayCammera() {}

CameraInfo HuarayCammera::getCameraInfo() {
    CameraInfo info;
    info.isFind_ = false;
    IMV_DeviceList deviceList;
    IMV_EnumDevices(&deviceList, IMV_EInterfaceType::interfaceTypeAll);
    for (size_t i = 0; i < deviceList.nDevNum; ++i) {
        if(cameraUserId_ == deviceList.pDevInfo[i].cameraName) {
            info.isFind_ = true;
            info.cameraKey_ = deviceList.pDevInfo[i].cameraKey;
            info.cameraUserId_ = deviceList.pDevInfo[i].cameraName;
            info.deviceType_ = deviceList.pDevInfo[i].nInterfaceType;
        }
    }
    return info;
}

bool HuarayCammera::connect() {
    auto ret =
        IMV_CreateHandle((void**)&pCamera_, IMV_ECreateHandleMode::modeByDeviceUserID,
                        (void*)cameraUserId_.data());

    if (IMV_OK != ret) {
        printf("create devHandle failed! userId[%s], ErrorCode[%d]\n",
               cameraUserId_.data(), ret);
        return false;
    }

    ret = IMV_Open(pCamera_);
    if (IMV_OK != ret) {
        printf("open camera failed! ErrorCode[%d]\n", ret);
        return false;
    }

    return true;
}

bool HuarayCammera::disConnect() {
    if (!pCamera_) {
        printf("close camera fail. No camera.\n");
        return false;
    }

    if (false == IMV_IsOpen(pCamera_)) {
        printf("camera is already close.\n");
        return false;
    }

    auto ret = IMV_Close(pCamera_);
    if (IMV_OK != ret) {
        printf("close camera failed! ErrorCode[%d]\n", ret);
        return false;
    }

    ret = IMV_DestroyHandle(pCamera_);
    if (IMV_OK != ret) {
        printf("destroy devHandle failed! ErrorCode[%d]\n", ret);
        return false;
    }

    pCamera_ = nullptr;

    return true;
}

SafeQueue<cv::Mat>& HuarayCammera::getImgs() {
    return imgs_;
}

bool HuarayCammera::pushImg(const cv::Mat &img) {
    imgs_.push(std::move(img));

    return true;
}

cv::Mat HuarayCammera::popImg() {
    cv::Mat img;
    bool isSucess = imgs_.try_move_pop(img);

    //std::cout << cameraUserId_ << " Size: " << imgs_.size() << " fps:  " << getFps() << " is sucess: " << isSucess << std::endl;

    return img;
}

bool HuarayCammera::clearImgs() {
    SafeQueue<cv::Mat> emptyQueue;
    imgs_.swap(emptyQueue);

    return true;
}

bool HuarayCammera::isConnect() { return IMV_IsOpen(pCamera_); }

cv::Mat HuarayCammera::capture() {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return cv::Mat();
    }

    const int preNums = imgs_.size();

    setTrigMode(TrigMode::trigSoftware);

    auto ret = IMV_ExecuteCommandFeature(pCamera_, "TriggerSoftware");
    if (IMV_OK != ret) {
        printf("ExecuteSoftTrig fail, ErrorCode[%d]\n", ret);
        return cv::Mat();
    }

    double exposureTime;
    getNumbericalAttribute("ExposureTime", exposureTime);
    auto timeBegin = std::chrono::system_clock::now();
    while(preNums == imgs_.size()) {
        auto timeEnd = std::chrono::system_clock::now();
        auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeBegin).count()* (double)std::chrono::milliseconds::period::num / std::chrono::milliseconds::period::den;
        if(timeElapsed > (exposureTime / 1000000.0 * 2))  {
            break;
        }
    };

    cv::Mat softWareCapturedImg = imgs_.back();

    setTrigMode(TrigMode::trigLine);

    return softWareCapturedImg;
}

bool HuarayCammera::start() {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    if (IMV_IsGrabbing(pCamera_)) {
        printf("camera is already grebbing.\n");
        return false;
    }

    auto ret = IMV_AttachGrabbing(pCamera_, frameCallback, this);

    if (IMV_OK != ret) {
        printf("Attach grabbing failed! ErrorCode[%d]\n", ret);
        return false;
    }

    ret = IMV_StartGrabbing(pCamera_);
    if (IMV_OK != ret) {
        printf("start grabbing failed! ErrorCode[%d]\n", ret);
        return false;
    }

    return true;
}

bool HuarayCammera::pause() {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    if (!IMV_IsGrabbing(pCamera_)) {
        printf("camera is already stop grubbing.\n");
        return false;
    }

    auto ret = IMV_StopGrabbing(pCamera_);
    if (IMV_OK != ret) {
        printf("Stop grubbing failed! ErrorCode[%d]\n", ret);
        return false;
    }

    return true;
}

bool HuarayCammera::setTrigMode(const TrigMode trigMode) {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    int ret = IMV_OK;
    if (trigContinous == trigMode) {
        ret = IMV_SetEnumFeatureSymbol(pCamera_, "TriggerMode", "Off");
        if (IMV_OK != ret) {
            printf("set TriggerMode value = Off fail, ErrorCode[%d]\n", ret);
            return false;
        }
    } else if (trigSoftware == trigMode) {
        ret = IMV_SetEnumFeatureSymbol(pCamera_, "TriggerMode", "On");
        if (IMV_OK != ret) {
            printf("set TriggerMode value = On fail, ErrorCode[%d]\n", ret);
            return false;
        }

        ret = IMV_SetEnumFeatureSymbol(pCamera_, "TriggerSource", "Software");
        if (IMV_OK != ret) {
            printf("set TriggerSource value = Software fail, ErrorCode[%d]\n",
                   ret);
            return false;
        }
    } else if (trigLine == trigMode) {
        ret = IMV_SetEnumFeatureSymbol(pCamera_, "TriggerMode", "On");
        if (IMV_OK != ret) {
            printf("set TriggerMode value = On fail, ErrorCode[%d]\n", ret);
            return false;
        }

        ret = IMV_SetEnumFeatureSymbol(pCamera_, "TriggerSource", "Line2");
        if (IMV_OK != ret) {
            printf("set TriggerSource value = Line1 fail, ErrorCode[%d]\n",
                   ret);
            return false;
        }
    }
    return true;
}

bool HuarayCammera::setEnumAttribute(const std::string attributeName,
                                     const std::string val) {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    return IMV_SetEnumFeatureSymbol(pCamera_, attributeName.data(),
                                    val.data()) == IMV_OK;
}

bool HuarayCammera::setStringAttribute(const std::string attributeName,
                                       const std::string val) {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    return IMV_SetStringFeatureValue(pCamera_, attributeName.data(),
                                     val.data()) == IMV_OK;
}

bool HuarayCammera::setNumberAttribute(const std::string attributeName,
                                           const double val) {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    return IMV_SetDoubleFeatureValue(pCamera_, attributeName.data(), val) == IMV_OK;
}

bool HuarayCammera::setBooleanAttribute(const std::string attributeName,
                                        const bool val) {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    return IMV_SetBoolFeatureValue(pCamera_, attributeName.data(), val) == IMV_OK;
}

int HuarayCammera::getFps() {
    IMV_StreamStatisticsInfo info;
    IMV_GetStatisticsInfo(pCamera_, &info);

    return info.u3vStatisticsInfo.fps;
}

bool HuarayCammera::getEnumAttribute(const std::string attributeName,
                                     std::string &val) {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    IMV_String data;
    IMV_GetEnumFeatureSymbol(pCamera_, attributeName.data(), &data);
    val = data.str;

    return true;
}

bool HuarayCammera::getStringAttribute(const std::string attributeName,
                                       std::string &val) {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    IMV_String data;
    IMV_GetStringFeatureValue(pCamera_, attributeName.data(), &data);
    val = data.str;

    return true;
}

bool HuarayCammera::getNumbericalAttribute(const std::string attributeName,
                                           double &val) {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    IMV_GetDoubleFeatureValue(pCamera_, attributeName.data(), &val);

    return true;
}

bool HuarayCammera::getBooleanAttribute(const std::string attributeName,
                                        bool &val) {
    if (!pCamera_) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    IMV_GetBoolFeatureValue(pCamera_, attributeName.data(), &val);

    return true;
}
} // namespace camera
} // namespace device
