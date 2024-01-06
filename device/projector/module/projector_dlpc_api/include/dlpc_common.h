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
 * \brief  Structs and functions to initialize the Command Library
 *
 */
 
#ifndef DLPC_COMMON_H
#define DLPC_COMMON_H
 
#ifdef __cplusplus
extern "C" {
#endif

#include "stdint.h"

#define DLPC_SUCCESS 0
#define SUCCESS                           0
#define FAIL                              1

#define DLP2010_WIDTH  854
#define DLP2010_HEIGHT 480
#define DLP230GP_WIDTH 960
#define DLP230GP_HEIGHT 540
#define DLP3010_WIDTH  1280
#define DLP3010_HEIGHT 720
#define DLP4710_WIDTH  1920
#define DLP4710_HEIGHT 1080

/**
* Additional information required for the command protocol. 
* Only used by the DLPC654x and DLPC754x controllers.
*/
typedef struct
{
    /** Command destination identifier */
    uint16_t CommandDestination;
    
    /** 
    * The number of bytes in the response data. 
    * The DLPC_COMMON_ReadCommandCallback should set this value for variable
    * length response data.
    */
    uint16_t BytesRead;

} DLPC_COMMON_CommandProtocolData_s;

/**
* The callback method that sends the specified number of bytes from the
* WriteBuffer to the controller
*
* \param[in] WriteLength   The number of bytes to transfer to the controller
* \param[in] WriteBuffer   The write buffer
* \param[in] ProtocolData  Additional information for the command protocol
*
* \return 0 if successful,
*         error code returned by the WriteCommandCallback otherwise
*/
typedef uint32_t(*DLPC_COMMON_WriteCommandCallback) (
    uint16_t                           WriteLength, 
    uint8_t*                           WriteBuffer, 
    DLPC_COMMON_CommandProtocolData_s* ProtocolData
);

/**
* The callback method that sends the specified number of bytes from the 
* WriteBuffer to the controller and reads the response data and populates
* it to the ReadBuffer
*
* \param[in] WriteLength   The number of bytes to transfer to the controller
* \param[in] WriteBuffer   The write buffer
* \param[in] ReadLength    The number of bytes to read from the controller.
*                          For variable length response, the value is 0xFFFF
* \param[in] ReadBuffer    The read buffer
* \param[in] ProtocolData  Additional information for the command protocol
*
* \return 0 if successful,
*         error code otherwise
*/
typedef uint32_t(*DLPC_COMMON_ReadCommandCallback) (
    uint16_t                           WriteLength, 
    uint8_t*                           WriteBuffer, 
    uint16_t                           ReadLength, 
    uint8_t*                           ReadBuffer,
    DLPC_COMMON_CommandProtocolData_s* ProtocolData
);

/**
* Initializes the read/write buffers and callbacks for the command APIs
* 
* A write command API packs command data to the WriteBuffer. A read command API
* packs the read request data to the WriteBuffer and unpacks the read response 
* data from the ReadBuffer. To avoid dynamic memory allocation or an arbitrarily 
* large static buffer allocation, the command APIs use the buffers provided the
* caller.
*
* The command APIs do not directly transmit data to the controller or receive data
* from the controller. Instead, they use the WriteCommandCallback and 
* ReadCommandCallback functions provided by the caller to send data to and receive
* data from the controller.
*
* \param[in] WriteBuffer          The write buffer
* \param[in] WriteBufferSize      The write buffer size in bytes
* \param[in] ReadBuffer           The read buffer
* \param[in] ReadBufferSize       The read buffer size in bytes
* \param[in] WriteCommandCallback The callback method that sends the specified 
*                                 number of bytes from the WriteBuffer to the 
*                                 controller 
* \param[in] ReadCommandCallback  The callback method that sends the specified 
*                                 number of bytes from the WriteBuffer to the 
*                                 controller and reads the response data and
*                                 populates it to the ReadBuffer
*/
void DLPC_COMMON_InitCommandLibrary(
    uint8_t*                         WriteBuffer,
    uint16_t                         WriteBufferSize,
    uint8_t*                         ReadBuffer,
    uint16_t                         ReadBufferSize,
    DLPC_COMMON_WriteCommandCallback WriteCommandCallback,
    DLPC_COMMON_ReadCommandCallback  ReadCommandCallback
);


#ifdef __cplusplus    /* matches __cplusplus construct above */
}
#endif
#endif /* DLPC_COMMON_H */
