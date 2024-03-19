/**
 * @file common.hpp
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __PROJECTOR_COMMON_H_
#define __PROJECTOR_COMMON_H_

#include "typeDef.h"

#include "cypress_i2c.h"
#include "dlpc34xx.h"
#include "dlpc34xx_dual.h"
#include "math.h"
#include "stdio.h"
#include "time.h"

#define FLASH_WRITE_BLOCK_SIZE 1024
#define FLASH_READ_BLOCK_SIZE 256

#define MAX_WRITE_CMD_PAYLOAD (FLASH_WRITE_BLOCK_SIZE + 8)
#define MAX_READ_CMD_PAYLOAD (FLASH_READ_BLOCK_SIZE + 8)

static uint8_t s_WriteBuffer[MAX_WRITE_CMD_PAYLOAD];
static uint8_t s_ReadBuffer[MAX_READ_CMD_PAYLOAD];

static bool s_StartProgramming;
static uint8_t s_FlashProgramBuffer[FLASH_WRITE_BLOCK_SIZE];
static uint16_t s_FlashProgramBufferPtr;

/**
 * @brief 通过Cypress USB-Serial写入数据
 *
 * @param writeDataLength 写入数据长度
 * @param writeData 写入数据
 * @param protocolData 命令符
 * @return int32_t 成功与否
 */
static uint32_t writeI2C(IN uint16_t writeDataLength, IN uint8_t *writeData,
                         IN DLPC_COMMON_CommandProtocolData_s *protocolData) {
    bool Status = true;
    Status = CYPRESS_I2C_WriteI2C(writeDataLength, writeData);
    if (Status != true) {
        printf("Write I2C Error!!! \n");
        return FAIL;
    }

    return SUCCESS;
}
/**
 * @brief 通过Cypress USB-Serial读取数据
 *
 * @param writeDataLength 写入数据长度
 * @param writeData 写入数据
 * @param readDataLength 读取数据长度
 * @param readData 读取到的数据
 * @param protocolData 命令符
 * @return int32_t 成功与否
 */
static uint32_t readI2C(IN uint16_t writeDataLength, IN uint8_t *writeData,
                        IN uint16_t readDataLength, IN uint8_t *readData,
                        IN DLPC_COMMON_CommandProtocolData_s *protocolData) {
    bool Status = 0;
    Status = CYPRESS_I2C_WriteI2C(writeDataLength, writeData);
    if (Status != true) {
        printf("Write I2C Error!!! \n");
        return FAIL;
    }

    Status = CYPRESS_I2C_ReadI2C(readDataLength, readData);
    if (Status != true) {
        printf("Read I2C Error!!! \n");
        return FAIL;
    }

    return SUCCESS;
}
/**
 * @brief 等待
 *
 * @param seconds 等待的时间(s)
 */
static void waitForSeconds(IN uint32_t seconds) {
    uint32_t retTime = (uint32_t)(time(0)) + seconds;
    while (time(0) < retTime);
}

/**
 * @brief 拷贝数据至flash内存
 *
 * @param length 长度
 * @param pData 数据
 */
static void copyDataToFlashProgramBuffer(IN uint8_t *length,
                                         IN uint8_t **pData) {
    while ((*length >= 1) &&
           (s_FlashProgramBufferPtr < sizeof(s_FlashProgramBuffer))) {
        s_FlashProgramBuffer[s_FlashProgramBufferPtr] = **pData;
        s_FlashProgramBufferPtr++;
        (*pData)++;
        (*length)--;
    }
}

/**
 * @brief 烧录
 *
 * @param length 数据长度
 */
static void programDualFlashWithDataInBuffer(IN uint16_t length) {
    s_FlashProgramBufferPtr = 0;

    if (s_StartProgramming) {
        s_StartProgramming = false;
        DLPC34XX_DUAL_WriteFlashStart(length, s_FlashProgramBuffer);
    } else {
        DLPC34XX_DUAL_WriteFlashContinue(length, s_FlashProgramBuffer);
    }
}

/**
 * @brief 烧录
 *
 * @param length 数据长度
 */
static void programFlashWithDataInBuffer(IN uint16_t length) {
    s_FlashProgramBufferPtr = 0;

    if (s_StartProgramming) {
        s_StartProgramming = false;
        DLPC34XX_WriteFlashStart(length, s_FlashProgramBuffer);
    } else {
        DLPC34XX_WriteFlashContinue(length, s_FlashProgramBuffer);
    }
}

/**
 * @brief 打包数据并烧录
 *
 * @param length 数据长度
 * @param pData 数据
 */
static void bufferDualPatternDataAndProgramToFlash(IN uint8_t length,
                                               IN uint8_t *pData) {
    copyDataToFlashProgramBuffer(&length, &pData);

    if (s_FlashProgramBufferPtr >= sizeof(s_FlashProgramBuffer)) {
        programDualFlashWithDataInBuffer((uint16_t)sizeof(s_FlashProgramBuffer));
    }

    copyDataToFlashProgramBuffer(&length, &pData);
}

/**
 * @brief 打包数据并烧录
 *
 * @param length 数据长度
 * @param pData 数据
 */
static void bufferPatternDataAndProgramToFlash(IN uint8_t length,
                                               IN uint8_t *pData) {
    copyDataToFlashProgramBuffer(&length, &pData);

    if (s_FlashProgramBufferPtr >= sizeof(s_FlashProgramBuffer)) {
        programFlashWithDataInBuffer((uint16_t)sizeof(s_FlashProgramBuffer));
    }

    copyDataToFlashProgramBuffer(&length, &pData);
}

#endif // !__PROJECTOR_COMMON_H_
