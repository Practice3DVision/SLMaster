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
  */

#include "dlpc_common.h"
#include "stdbool.h"
#include "string.h"

static uint8_t*                          s_WriteBuffer;
static uint16_t                          s_WriteBufferSize;
static uint16_t                          s_WriteBufferIndex;
static uint8_t*                          s_ReadBuffer;
static uint16_t                          s_ReadBufferSize;
static uint16_t                          s_ReadBufferIndex;
static DLPC_COMMON_WriteCommandCallback  s_WriteCommandCallback;
static DLPC_COMMON_ReadCommandCallback   s_ReadCommandCallback;
static DLPC_COMMON_CommandProtocolData_s s_ProtocolData;

void DLPC_COMMON_InitCommandLibrary(
    uint8_t*                         WriteBuffer,
    uint16_t                         WriteBufferSize,
    uint8_t*                         ReadBuffer,
    uint16_t                         ReadBufferSize,
    DLPC_COMMON_WriteCommandCallback WriteCommandCallback,
    DLPC_COMMON_ReadCommandCallback  ReadCommandCallback)
{
    s_WriteBuffer          = WriteBuffer;
    s_WriteBufferSize      = WriteBufferSize;
    s_ReadBuffer           = ReadBuffer;
    s_ReadBufferSize       = ReadBufferSize;
    s_WriteCommandCallback = WriteCommandCallback;
    s_ReadCommandCallback  = ReadCommandCallback;
}

uint32_t DLPC_COMMON_SendWrite()
{
    return s_WriteCommandCallback(s_WriteBufferIndex, s_WriteBuffer, &s_ProtocolData);
}

uint32_t DLPC_COMMON_SendRead(uint16_t ReadLength)
{
    return s_ReadCommandCallback(s_WriteBufferIndex,
                                 s_WriteBuffer,
                                 ReadLength,
                                 s_ReadBuffer,
                                 &s_ProtocolData);
}

void DLPC_COMMON_ClearWriteBuffer()
{
    memset(s_WriteBuffer, 0, s_WriteBufferSize);
    s_WriteBufferIndex = 0;
}

void DLPC_COMMON_ClearReadBuffer()
{
    memset(s_ReadBuffer, 0, s_ReadBufferSize);
    s_ReadBufferIndex = 0;
}

int64_t ConvertFloatToFixed(double Value, uint32_t Scale)
{
    return (int64_t)(Value * (double)Scale);
}

double ConvertFixedToFloat(int64_t Value, uint32_t Scale)
{
    return Value / (double)Scale;
}

uint64_t GetBitMask(uint8_t NumBits)
{
    return UINT64_MAX >> (64 - NumBits);
}

bool IsSignedBitSet(int64_t Value, int32_t NumBits)
{
    return (Value & (int64_t)(1 << (NumBits - 1))) != 0;
}

void DLPC_COMMON_PackOpcode(int32_t Length, uint16_t Opcode)
{
    memcpy(&s_WriteBuffer[s_WriteBufferIndex], &Opcode, Length);
    s_WriteBufferIndex += Length;
}

void DLPC_COMMON_MoveWriteBufferPointer(int32_t Offset)
{
    s_WriteBufferIndex += Offset;
}

void DLPC_COMMON_PackByte(uint8_t Data)
{
    s_WriteBuffer[s_WriteBufferIndex] = Data;
    s_WriteBufferIndex++;
}

void DLPC_COMMON_PackBytes(uint8_t* Data, int32_t Length)
{
    memcpy(&s_WriteBuffer[s_WriteBufferIndex], Data, Length);
    s_WriteBufferIndex += Length;
}

void DLPC_COMMON_PackFloat(double Value, int32_t Length, uint32_t Scale)
{
    int64_t FixedValue = ConvertFloatToFixed(Value, Scale);
    DLPC_COMMON_PackBytes((uint8_t*)&FixedValue, Length);
}

void DLPC_COMMON_SetBits(int32_t Value, int32_t NumBits, int32_t BitOffset)
{
    uint32_t StartBit  = BitOffset % 8;
    uint32_t StartByte = s_WriteBufferIndex + (BitOffset / 8);
    uint32_t EndByte   = StartByte + ((NumBits + 7) / 8);
    uint32_t Index;
    uint64_t BitMask;

    for (Index = StartByte; Index < EndByte; Index++)
    {
        BitMask = GetBitMask(NumBits > 8 ? 8 : NumBits);
        
        s_WriteBuffer[Index] &= (uint8_t)(~(BitMask << StartBit));
        s_WriteBuffer[Index] |= (uint8_t)((Value & BitMask) << StartBit);
        
        Value    = Value >> (8 - StartBit);
        NumBits  = NumBits - (8 - StartBit);
        StartBit = 0;
    }
}

void DLPC_COMMON_MoveReadBufferPointer(int32_t Length)
{
    s_ReadBufferIndex += Length;
}

uint8_t* DLPC_COMMON_UnpackBytes(int32_t Length)
{
    uint16_t CurReadBufferIndex = s_ReadBufferIndex;
    s_ReadBufferIndex += Length;

    return &s_ReadBuffer[CurReadBufferIndex];
}

double DLPC_COMMON_UnpackFloat(int32_t Length, uint32_t Scale, bool Signed)
{
    uint8_t NumBits    = (uint8_t)(8 * Length);
    int64_t FixedValue = *((int64_t*)DLPC_COMMON_UnpackBytes(Length));

    FixedValue &= GetBitMask(NumBits);
    if (Signed && IsSignedBitSet(FixedValue, NumBits))
    {
        FixedValue |= (int64_t)(UINT64_MAX << NumBits);
    }

    return ConvertFixedToFloat(FixedValue, Scale);
}

uint64_t DLPC_COMMON_GetBits(uint8_t NumBits, uint8_t BitOffset, bool Signed)
{
    uint32_t StartBit  = BitOffset % 8;
    uint32_t StartByte = s_ReadBufferIndex + (BitOffset / 8);
    uint32_t EndByte   = StartByte + ((NumBits + 7) / 8);
    uint64_t Value     = 0;
    uint32_t Index;
    uint64_t BitMask;
    uint8_t  Shift;

    for (Index = StartByte; Index < EndByte; Index++)
    {
        BitMask = GetBitMask(NumBits > 8 ? 8 : NumBits);

        Shift = 8 * (Index - StartByte);
        Value |= (((s_ReadBuffer[Index] >> StartBit) & BitMask) << Shift);

        NumBits  = NumBits - (8 - StartBit);
        StartBit = 0;
    }

    return Value;
}

void DLPC_COMMON_SetCommandDestination(uint16_t CommandDestination)
{
    s_ProtocolData.CommandDestination = CommandDestination;
}

uint16_t DLPC_COMMON_GetBytesRead()
{
    return s_ProtocolData.BytesRead;
}