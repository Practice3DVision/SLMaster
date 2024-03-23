#include "projectorDlpc34xx.h"

#include "common.hpp"

#include "CyUSBSerial.h"

namespace slmaster {
namespace device {

void ProjectorDlpc34xx::loadPatternOrderTableEntryFromFlash() {
    DLPC34XX_PatternOrderTableEntry_s PatternOrderTableEntry;

    DLPC34XX_WritePatternOrderTableEntry(DLPC34XX_WC_RELOAD_FROM_FLASH,
                                         &PatternOrderTableEntry);
}

bool ProjectorDlpc34xx::initConnectionAndCommandLayer() {
    DLPC_COMMON_InitCommandLibrary(s_WriteBuffer, sizeof(s_WriteBuffer),
                                   s_ReadBuffer, sizeof(s_ReadBuffer), writeI2C,
                                   readI2C);

    isInitial_ = CYPRESS_I2C_ConnectToCyI2C();

    return isInitial_;
}

ProjectorDlpc34xx::ProjectorDlpc34xx() : isInitial_(false) {
    cols_ = DLP3010_WIDTH;
    rows_ = DLP3010_HEIGHT;
}

bool ProjectorDlpc34xx::connect() {
    bool isInitSucess = initConnectionAndCommandLayer();
    if (!isInitSucess) {
        printf("init DLPC-USB connection error! \n");
        return false;
    }

    if (!CYPRESS_I2C_RequestI2CBusAccess()) {
        printf("Error request I2C bus access! \n");
        return false;
    }

    loadPatternOrderTableEntryFromFlash();

    //DLPC34XX_WriteInputImageSize(cols_, rows_);
    //DLPC34XX_WriteImageCrop(0, 0, cols_, rows_);
    //DLPC34XX_WriteDisplaySize(0, 0, cols_, rows_);

    DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_STOP, 0);
    DLPC34XX_WriteTriggerOutConfiguration(
        DLPC34XX_TT_TRIGGER1, DLPC34XX_TE_ENABLE, DLPC34XX_TI_NOT_INVERTED, 0);
    DLPC34XX_WriteTriggerOutConfiguration(
        DLPC34XX_TT_TRIGGER2, DLPC34XX_TE_ENABLE, DLPC34XX_TI_NOT_INVERTED, 0);
    DLPC34XX_WriteTriggerInConfiguration(DLPC34XX_TE_DISABLE,
                                         DLPC34XX_TP_ACTIVE_HI);
    DLPC34XX_WritePatternReadyConfiguration(DLPC34XX_TE_ENABLE,
                                            DLPC34XX_TP_ACTIVE_HI);
    DLPC34XX_WriteOperatingModeSelect(DLPC34XX_OM_SENS_INTERNAL_PATTERN);

    return true;
}

bool ProjectorDlpc34xx::disConnect() {
    if(!isInitial_) {
        return false;
    }

    uint8_t numDevices;
    CyGetListofDevices(&numDevices);
    for (size_t i = 0; i < numDevices; ++i) {
        CY_DEVICE_INFO deviceInfo;
        CyGetDeviceInfo(i, &deviceInfo);
        CY_HANDLE handle;
        unsigned char sig[6];
        if(deviceInfo.deviceType[i] == CY_DEVICE_TYPE::CY_TYPE_I2C && deviceInfo.deviceClass[i] == CY_CLASS_VENDOR) {
            if (CY_SUCCESS == CyOpen(i, 0, &handle)) {
                CyClose(handle);
                break;
            }
        }
    }
    
    return true;
}

bool ProjectorDlpc34xx::isConnect() {
    if(!isInitial_) {
        return false;
    }

    DLPC34XX_ShortStatus_s shortStatus;
    DLPC34XX_ReadShortStatus(&shortStatus);
    return shortStatus.SystemError == DLPC34XX_E_NO_ERROR;
}

bool ProjectorDlpc34xx::populatePatternTableData(
    std::vector<PatternOrderSet> table) {
    if (!isConnect()) {
        return false;
    }

    numOfPatternSets_ = table.size();
    numOfPatterns_ = 0;
    for (size_t i = 0; i < table.size(); ++i) {
        numOfPatterns_ += table[i].imgs_.size();
    }

    DLPC34XX_INT_PAT_PatternData_s *patterns =
        new DLPC34XX_INT_PAT_PatternData_s[numOfPatterns_];
    DLPC34XX_INT_PAT_PatternSet_s *patternSets =
        new DLPC34XX_INT_PAT_PatternSet_s[numOfPatternSets_];
    DLPC34XX_INT_PAT_PatternOrderTableEntry_s *patternOrderTableEntries =
        new DLPC34XX_INT_PAT_PatternOrderTableEntry_s[numOfPatternSets_];

    int indexOfPattern = 0;
    for (size_t i = 0; i < table.size(); ++i) {
        patternSets[i].BitDepth = table[i].isOneBit_ == true
                                      ? DLPC34XX_INT_PAT_BITDEPTH_ONE
                                      : DLPC34XX_INT_PAT_BITDEPTH_EIGHT;
        patternSets[i].Direction = table[i].isVertical_ == true
                                       ? DLPC34XX_INT_PAT_DIRECTION_VERTICAL
                                       : DLPC34XX_INT_PAT_DIRECTION_HORIZONTAL;
        patternSets[i].PatternArray = &patterns[indexOfPattern];
        patternSets[i].PatternCount = table[i].imgs_.size();

        for (size_t j = 0; j < table[i].imgs_.size(); ++j) {
            patterns[indexOfPattern].PixelArrayCount =
                table[i].patternArrayCounts_;

            table[i].imgs_[j] = table[i].isVertical_ ? table[i].imgs_[j] : table[i].imgs_[j].t();
            patterns[indexOfPattern].PixelArray = table[i].imgs_[j].data;
            ++indexOfPattern;
        }

        patternOrderTableEntries[i].PatternSetIndex = i;
        patternOrderTableEntries[i].NumDisplayPatterns =
            patternSets[i].PatternCount;
        patternOrderTableEntries[i].IlluminationSelect =
            (table[i].illumination_ == Red ? DLPC34XX_INT_PAT_ILLUMINATION_RED
             : table[i].illumination_ == Grren
                 ? DLPC34XX_INT_PAT_ILLUMINATION_GREEN
                 : table[i].illumination_ == Blue ? DLPC34XX_INT_PAT_ILLUMINATION_BLUE : DLPC34XX_INT_PAT_ILLUMINATION_RGB);
        patternOrderTableEntries[i].InvertPatterns = false;
        patternOrderTableEntries[i].IlluminationTimeInMicroseconds =
            table[i].exposureTime_;
        patternOrderTableEntries[i].PreIlluminationDarkTimeInMicroseconds =
            table[i].preExposureTime_;
        patternOrderTableEntries[i].PostIlluminationDarkTimeInMicroseconds =
            table[i].postExposureTime_;
    }

    DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_STOP, 0);

    s_StartProgramming = true;
    s_FlashProgramBufferPtr = 0;

    DLPC34XX_WriteFlashDataTypeSelect(
        DLPC34XX_FDTS_ENTIRE_SENS_PATTERN_DATA);
    DLPC34XX_WriteFlashErase();
    DLPC34XX_ShortStatus_s ShortStatus;
    do {
        DLPC34XX_ReadShortStatus(&ShortStatus);
    } while (ShortStatus.FlashEraseComplete == DLPC34XX_FE_NOT_COMPLETE);

    DLPC34XX_WriteFlashDataLength(sizeof(s_FlashProgramBuffer));

    DLPC34XX_INT_PAT_GeneratePatternDataBlock(
        DLPC34XX_INT_PAT_DMD_DLP4710, numOfPatternSets_, patternSets,
        numOfPatternSets_, patternOrderTableEntries,
        bufferPatternDataAndProgramToFlash, false, false);
    if (s_FlashProgramBufferPtr > 0) {
        DLPC34XX_WriteFlashDataLength(s_FlashProgramBufferPtr);

        programFlashWithDataInBuffer(s_FlashProgramBufferPtr);
    }

    loadPatternOrderTableEntryFromFlash();

    s_StartProgramming = false;

    delete[] patternOrderTableEntries;
    delete[] patternSets;
    delete[] patterns;

    return true;
}

bool ProjectorDlpc34xx::project(const bool isContinue) {
    if(!isInitial_) {
        return false;
    }

    if (isContinue) {
        DLPC34XX_WriteOperatingModeSelect(DLPC34XX_OM_SENS_INTERNAL_PATTERN);
        return DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_START, 0xFF) ==
               SUCCESS;
    } else {
        DLPC34XX_WriteOperatingModeSelect(DLPC34XX_OM_SENS_INTERNAL_PATTERN);
        return DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_START, 0x0) ==
               SUCCESS;
    }

    return true;
}

bool ProjectorDlpc34xx::stop() {
    if(!isInitial_) {
        return false;
    }

    return DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_STOP, 0) == SUCCESS;
}

bool ProjectorDlpc34xx::pause() {
    if(!isInitial_) {
        return false;
    }

    return DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_PAUSE, 0xff) ==
           SUCCESS;
}

bool ProjectorDlpc34xx::resume() {
    if(!isInitial_) {
        return false;
    }

    return DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_RESUME, 0xff) ==
           SUCCESS;
}

bool ProjectorDlpc34xx::step() {
    if(!isInitial_) {
        return false;
    }

    return DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_STEP, 0xff) ==
           SUCCESS;
}

bool ProjectorDlpc34xx::getLEDCurrent(OUT double &r, OUT double &g,
                                      OUT double &b) {
    if(!isInitial_) {
        return false;
    }

    uint16_t red, green, blue, maxRed, maxGreen, maxBlue;
    auto isSucess = DLPC34XX_ReadRgbLedCurrent(&red, &green, &blue);
    isSucess = DLPC34XX_ReadRgbLedMaxCurrent(&maxRed, &maxGreen, &maxBlue);
    r = (double)red / maxRed;
    g = (double)red / maxRed;
    b = (double)red / maxRed;

    return isSucess == SUCCESS;
}

bool ProjectorDlpc34xx::setLEDCurrent(IN const double r, IN const double g,
                                      IN const double b) {
    if(!isInitial_) {
        return false;
    }

    uint16_t maxRed, maxGreen, maxBlue;
    bool isSucess = DLPC34XX_WriteRgbLedEnable(true, true, true);
    isSucess = DLPC34XX_ReadRgbLedMaxCurrent(&maxRed, &maxGreen, &maxBlue);
    isSucess =
        DLPC34XX_WriteRgbLedCurrent(maxRed * r, maxGreen * g, maxBlue * b);

    return isSucess == SUCCESS;
}

int ProjectorDlpc34xx::getFlashImgsNum() {
    if(!isInitial_) {
        return false;
    }

    DLPC34XX_InternalPatternStatus_s status;
    DLPC34XX_ReadInternalPatternStatus(&status);

    return status.NumPatDisplayedFromPatSet;
}

ProjectorDlpc34xx::~ProjectorDlpc34xx() {
}

ProjectorInfo ProjectorDlpc34xx::getInfo() {
    ProjectorInfo projectorInfo;
    projectorInfo.dlpEvmType_ = "DLP4710";
    projectorInfo.width_ = cols_;
    projectorInfo.height_ = rows_;
    projectorInfo.isFind_ = false;

    uint8_t numDevices;
    CyGetListofDevices(&numDevices);
    for (size_t i = 0; i < numDevices; ++i) {
        CY_DEVICE_INFO deviceInfo;
        CyGetDeviceInfo(i, &deviceInfo);
        CY_HANDLE handle;
        unsigned char sig[6];
        if(deviceInfo.deviceType[i] == CY_DEVICE_TYPE::CY_TYPE_I2C && deviceInfo.deviceClass[i] == CY_CLASS_VENDOR) {
            projectorInfo.isFind_ = true;
            break;
        }
    }

    return projectorInfo;
}

} // namespace projector
} // namespace sl
