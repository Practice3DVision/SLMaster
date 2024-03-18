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
 * \brief  Common private functions used by the command APIs
 */
 
#ifndef DLPC_COMMON_PRIVATE_H
#define DLPC_COMMON_PRIVATE_H
 
#ifdef __cplusplus
extern "C" {
#endif

#include "stdbool.h"
#include "stdint.h"
#include "stdlib.h"
#include "dlpc_common.h"

/**
* Invokes the WriteCommandCallback to send the specified number of bytes from the
* WriteBuffer to the controller. This function is called by the write command
* APIs after packing the command data in the WriteBuffer.
*
* \param[in] WriteLength    The number of bytes to transfer to the controller
*
* \return 0 if successful,
*         error code returned by the WriteCommandCallback otherwise
*/
uint32_t DLPC_COMMON_SendWrite();

/**
* Invokes the ReadCommandCallback to send the specified number of bytes from the
* WriteBuffer to the controller. This function is called by the read command
* APIs after packing the read request data in the WriteBuffer.
*
* \param[in] WriteLength    The number of bytes to transfer to the controller
* \param[in] ReadLength     The number of bytes to read from the controller.
*                           For variable length response, the value is 0xFFFF
*
* \return 0 if successful,
*         error code returned by the ReadCommandCallback otherwise
*/
uint32_t DLPC_COMMON_SendRead(uint16_t ReadLength);

/**
* Sets the write buffer to zeros to clear data from previous commands
* and resets the write buffer pointer
*/
void DLPC_COMMON_ClearWriteBuffer();

/**
* Sets the read buffer to zeros to clear data from previous commands
*  and resets the read buffer pointer
*/
void DLPC_COMMON_ClearReadBuffer();

/**
* Packs opcode to the WriteBuffer
*
* \param[in] Length  Length of the opcode in bytes
* \param[in] Opcode  The command opcode
*/
void DLPC_COMMON_PackOpcode(int32_t Length, uint16_t Opcode);

/**
* Moves the WriteBuffer pointer by specified number of bytes
*
* \param[in] Offset  Number of bytes to skip in the WriteBuffer 
*/
void DLPC_COMMON_MoveWriteBufferPointer(int32_t Offset);

/**
* Copies a byte to the WriteBuffer at the current WriteBuffer pointer 
* and increments the pointer by one
*
* \param[in] Data     The byte to copy
*/
void DLPC_COMMON_PackByte(uint8_t Data);


/**
* Copies a byte array of the given number of bytes to the WriteBuffer
* at the current WriteBuffer pointer and increments the pointer by 
* the specified Length
*
* \param[in] Data     Array to copy to the WriteBuffer
* \param[in] Length   Number of bytes to copy
*/
void DLPC_COMMON_PackBytes(uint8_t* Data, int32_t Length);

/**
* Converts a floating point value to fixed point value and packs it to
* the WriteBuffer. Also, increments the WriteBuffer pointer by the 
* specified Length.
*
* \param[in] Value  The floating point value to pack
* \param[in] Length The size of the fixed point value in bytes 
* \param[in] Scale  The scale value used to convert floating point value
*                   to fixed point value.
*                   fixed_pt = (int) floating_pt * scale
*/
void DLPC_COMMON_PackFloat(double Value, int32_t Length, uint32_t Scale);

/**
* Sets a bit-field to the given value
*
* \param[in] Value     The value to set
* \param[in] NumBits   Number of bits in the bit-field. The range is 1-64
* \param[in] BitOffset The starting bit of the bit-field. The range is 0-63
*/
void DLPC_COMMON_SetBits(int32_t Value, int32_t NumBits, int32_t BitOffset);

/**
* Moves the ReadBuffer pointer by specified number of bytes
*
* \param[in] Offset  Number of bytes to skip in the ReadBuffer
*/
void DLPC_COMMON_MoveReadBufferPointer(int32_t Offset);

/**
* Gets a byte array starting at the current ReadBuffer pointer and move the 
* pointer by the specified Length.
*
* \param[in] Length   Number of bytes to unpack
*/
uint8_t*  DLPC_COMMON_UnpackBytes(int32_t Length);

/**
* Unpacks a fixed point value of given length from the ReadBuffer and converts
* it to a floating point value. Also, increments the ReadBuffer pointer by the
* specified Length.
*
* \param[in] Length The length of the fixed point value in bytes
* \param[in] Scale  The scale value used to convert floating point value
*                   to fixed point value.
*                   fixed_pt = (int) floating_pt * scale
* \param[in] Signed A boolean value indicating if the floating point value is
*                   a signed value or not.
*
* \return Returns the floating point value
*/
double DLPC_COMMON_UnpackFloat(int32_t Length, uint32_t Scale, bool Signed);

/**
* Gets a bit-field value
*
* \param[in] NumBits   Number of bits in the bit-field. The range is 1-64
* \param[in] BitOffset The starting bit of the bit-field. The range is 0-63
* \param[in] Signed    A boolean value indicating if the bit-field value is 
*                      a signed value or not.
*
* \return Returns the bit-field value
*/
uint64_t DLPC_COMMON_GetBits(uint8_t NumBits, uint8_t BitOffset, bool Signed);


/**
* Sets the Command destination identifier for the next command
*
* \param[in] CommandDestination  Command destination identifier
*/
void DLPC_COMMON_SetCommandDestination(uint16_t CommandDestination);

/**
* Gets the number of bytes in the read response of the previous command.
* This function is used by the command APIs to know the response data
* length for commands with variable length response data.
*
* \return The length of the read response.
*/
uint16_t DLPC_COMMON_GetBytesRead();

#ifdef __cplusplus    /* matches __cplusplus construct above */
}
#endif
#endif /* DLPC_COMMON_PRIVATE_H */
