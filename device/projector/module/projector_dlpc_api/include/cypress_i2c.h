/*------------------------------------------------------------------------------
 * Copyright (c) 2019 Texas Instruments Incorporated - http://www.ti.com/
 *------------------------------------------------------------------------------
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief  Sample code for I2C communication via Cypress USB-Serial 
 *         Bridge Controller.
 *
 */

#ifndef CYPRESS_I2C_H
#define CYPRESS_I2C_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stdbool.h"
#include "stdint.h"

bool CYPRESS_I2C_RequestI2CBusAccess();
bool CYPRESS_I2C_RelinquishI2CBusAccess();
bool CYPRESS_I2C_WriteI2C(uint32_t WriteDataLength, uint8_t* WriteData);
bool CYPRESS_I2C_ReadI2C(uint32_t ReadDataLength, uint8_t* ReadData);
bool CYPRESS_I2C_ConnectToCyI2C();
bool CYPRESS_I2C_GetCyGpio(uint8_t GpioNum, uint8_t* Value);
bool CYPRESS_I2C_SetCyGpio(uint8_t GpioNum, uint8_t Value);

#ifdef __cplusplus    /* matches __cplusplus construct above */
}
#endif
#endif /* CYPRESS_I2C_H */