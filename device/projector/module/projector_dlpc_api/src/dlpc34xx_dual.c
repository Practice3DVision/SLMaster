/*------------------------------------------------------------------------------
 * Copyright (c) 2020 Texas Instruments Incorporated - http://www.ti.com/
 *------------------------------------------------------------------------------
 *
 * NOTE: This file is auto generated from a command definition file.
 *       Please do not modify the file directly.                    
 *
 * Command Spec Version : 1.0
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the
 *   distribution.
 *
 *   Neither the name of Texas Instruments Incorporated nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * \file
 * \brief DLPC347x Dual Commands
 */

#include "stdlib.h"
#include "string.h"
#include "dlpc34xx_dual.h"
#include "dlpc_common_private.h"

static uint32_t s_Index;

uint32_t DLPC34XX_DUAL_WriteOperatingModeSelect(DLPC34XX_DUAL_OperatingMode_e OperatingMode)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x5);
    DLPC_COMMON_PackBytes((uint8_t*)&OperatingMode, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadOperatingModeSelect(DLPC34XX_DUAL_OperatingMode_e *OperatingMode)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x6);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *OperatingMode = (DLPC34XX_DUAL_OperatingMode_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteSplashScreenSelect(uint8_t SplashScreenIndex)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xD);
    DLPC_COMMON_PackBytes((uint8_t*)&SplashScreenIndex, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadSplashScreenSelect(uint8_t *SplashScreenIndex)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *SplashScreenIndex = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteSplashScreenExecute()
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x35);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadSplashScreenHeader(uint8_t SplashScreenIndex, DLPC34XX_DUAL_SplashScreenHeader_s *SplashScreenHeader)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xF);
    DLPC_COMMON_PackBytes((uint8_t*)&SplashScreenIndex, 1);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(13);
    if (Status == 0)
    {
        SplashScreenHeader->WidthInPixels = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SplashScreenHeader->HeightInPixels = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SplashScreenHeader->SizeInBytes = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        SplashScreenHeader->PixelFormat = (DLPC34XX_DUAL_PixelFormats_e)*(DLPC_COMMON_UnpackBytes(1));
        SplashScreenHeader->CompressionType = (DLPC34XX_DUAL_CompressionTypes_e)*(DLPC_COMMON_UnpackBytes(1));
        SplashScreenHeader->ColorOrder = (DLPC34XX_DUAL_ColorOrders_e)*(DLPC_COMMON_UnpackBytes(1));
        SplashScreenHeader->ChromaOrder = (DLPC34XX_DUAL_ChromaOrders_e)*(DLPC_COMMON_UnpackBytes(1));
        SplashScreenHeader->ByteOrder = (DLPC34XX_DUAL_ByteOrders_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteExternalVideoSourceFormatSelect(DLPC34XX_DUAL_ExternalVideoFormat_e VideoFormat)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x7);
    DLPC_COMMON_PackBytes((uint8_t*)&VideoFormat, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadExternalVideoSourceFormatSelect(DLPC34XX_DUAL_ExternalVideoFormat_e *VideoFormat)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x8);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *VideoFormat = (DLPC34XX_DUAL_ExternalVideoFormat_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteVideoChromaProcessingSelect(DLPC34XX_DUAL_ChromaInterpolationMethod_e ChromaInterpolationMethod, DLPC34XX_DUAL_ChromaChannelSwap_e ChromaChannelSwap, uint8_t CscCoefficientSet)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x9);
    DLPC_COMMON_SetBits((int32_t)ChromaInterpolationMethod, 1, 4);
    DLPC_COMMON_SetBits((int32_t)ChromaChannelSwap, 1, 2);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)CscCoefficientSet, 2, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadVideoChromaProcessingSelect(DLPC34XX_DUAL_ChromaInterpolationMethod_e *ChromaInterpolationMethod, DLPC34XX_DUAL_ChromaChannelSwap_e *ChromaChannelSwap, uint8_t *CscCoefficientSet)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xA);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *ChromaInterpolationMethod = (DLPC34XX_DUAL_ChromaInterpolationMethod_e)DLPC_COMMON_GetBits(1, 4, false);
        *ChromaChannelSwap = (DLPC34XX_DUAL_ChromaChannelSwap_e)DLPC_COMMON_GetBits(1, 3, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        *CscCoefficientSet = (uint8_t)DLPC_COMMON_GetBits(2, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_Write3DControl(DLPC34XX_DUAL_ThreeDDominance_e ThreeDFrameDominance, DLPC34XX_DUAL_ThreeDReferencePolarity_e ThreeDReferencePolarity)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x20);
    DLPC_COMMON_SetBits(1, 1, 1);     // ThreeDSource
    DLPC_COMMON_SetBits((int32_t)ThreeDFrameDominance, 1, 5);
    DLPC_COMMON_SetBits((int32_t)ThreeDReferencePolarity, 1, 6);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_Read3DControl(DLPC34XX_DUAL_ThreeDModes_e *ThreeDMode, DLPC34XX_DUAL_ThreeDDominance_e *ThreeDFrameDominance, DLPC34XX_DUAL_ThreeDReferencePolarity_e *ThreeDReferencePolarity)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x21);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *ThreeDMode = (DLPC34XX_DUAL_ThreeDModes_e)DLPC_COMMON_GetBits(1, 0, false);
        *ThreeDFrameDominance = (DLPC34XX_DUAL_ThreeDDominance_e)DLPC_COMMON_GetBits(1, 5, false);
        *ThreeDReferencePolarity = (DLPC34XX_DUAL_ThreeDReferencePolarity_e)DLPC_COMMON_GetBits(1, 6, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteInputImageSize(uint16_t PixelsPerLine, uint16_t LinesPerFrame)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x2E);
    DLPC_COMMON_PackBytes((uint8_t*)&PixelsPerLine, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&LinesPerFrame, 2);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadInputImageSize(uint16_t *PixelsPerLine, uint16_t *LinesPerFrame)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x2F);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *PixelsPerLine = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *LinesPerFrame = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteDisplaySize(uint16_t PixelsPerLine, uint16_t LinesPerFrame)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x12);
    DLPC_COMMON_PackBytes((uint8_t*)&PixelsPerLine, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&LinesPerFrame, 2);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadDisplaySize(uint16_t *PixelsPerLine, uint16_t *LinesPerFrame)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x13);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *PixelsPerLine = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *LinesPerFrame = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteDisplayImageOrientation(DLPC34XX_DUAL_ImageFlip_e LongAxisImageFlip, DLPC34XX_DUAL_ImageFlip_e ShortAxisImageFlip)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x14);
    DLPC_COMMON_SetBits((int32_t)LongAxisImageFlip, 1, 1);
    DLPC_COMMON_SetBits((int32_t)ShortAxisImageFlip, 1, 2);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadDisplayImageOrientation(DLPC34XX_DUAL_ImageFlip_e *LongAxisImageFlip, DLPC34XX_DUAL_ImageFlip_e *ShortAxisImageFlip)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x15);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *LongAxisImageFlip = (DLPC34XX_DUAL_ImageFlip_e)DLPC_COMMON_GetBits(1, 1, false);
        *ShortAxisImageFlip = (DLPC34XX_DUAL_ImageFlip_e)DLPC_COMMON_GetBits(1, 2, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteDisplayImageCurtain(DLPC34XX_DUAL_ImageCurtainEnable_e Enable, DLPC34XX_DUAL_Color_e Color)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x16);
    DLPC_COMMON_SetBits((int32_t)Enable, 1, 0);
    DLPC_COMMON_SetBits((int32_t)Color, 3, 1);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadDisplayImageCurtain(DLPC34XX_DUAL_ImageCurtainEnable_e *Enable, DLPC34XX_DUAL_Color_e *Color)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x17);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Enable = (DLPC34XX_DUAL_ImageCurtainEnable_e)DLPC_COMMON_GetBits(1, 0, false);
        *Color = (DLPC34XX_DUAL_Color_e)DLPC_COMMON_GetBits(3, 1, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteImageFreeze(bool Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1A);
    DLPC_COMMON_PackBytes((uint8_t*)&Enable, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadImageFreeze(bool *Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1B);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Enable = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteBorderColor(DLPC34XX_DUAL_Color_e DisplayBorderColor)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB2);
    DLPC_COMMON_PackBytes((uint8_t*)&DisplayBorderColor, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadBorderColor(DLPC34XX_DUAL_Color_e *DisplayBorderColor, DLPC34XX_DUAL_BorderColorSource_e *PillarBoxBorderColorSource)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB3);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *DisplayBorderColor = (DLPC34XX_DUAL_Color_e)DLPC_COMMON_GetBits(3, 0, false);
        *PillarBoxBorderColorSource = (DLPC34XX_DUAL_BorderColorSource_e)DLPC_COMMON_GetBits(1, 7, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteSolidField(DLPC34XX_DUAL_BorderEnable_e Border, DLPC34XX_DUAL_Color_e ForegroundColor)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB);
    DLPC_COMMON_SetBits(0, 4, 0);     // PatternSelect
    DLPC_COMMON_SetBits((int32_t)Border, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)ForegroundColor, 3, 4);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteHorizontalRamp(DLPC34XX_DUAL_BorderEnable_e Border, DLPC34XX_DUAL_Color_e ForegroundColor, uint8_t StartValue, uint8_t EndValue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB);
    DLPC_COMMON_SetBits(1, 4, 0);     // PatternSelect
    DLPC_COMMON_SetBits((int32_t)Border, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)ForegroundColor, 3, 4);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&StartValue, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&EndValue, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteVerticalRamp(DLPC34XX_DUAL_BorderEnable_e Border, DLPC34XX_DUAL_Color_e ForegroundColor, uint8_t StartValue, uint8_t EndValue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB);
    DLPC_COMMON_SetBits(2, 4, 0);     // PatternSelect
    DLPC_COMMON_SetBits((int32_t)Border, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)ForegroundColor, 3, 4);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&StartValue, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&EndValue, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteHorizontalLines(DLPC34XX_DUAL_HorizontalLines_s *HorizontalLines)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB);
    DLPC_COMMON_SetBits(3, 4, 0);     // PatternSelect
    DLPC_COMMON_SetBits((int32_t)HorizontalLines->Border, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)HorizontalLines->BackgroundColor, 3, 0);
    DLPC_COMMON_SetBits((int32_t)HorizontalLines->ForegroundColor, 3, 4);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&HorizontalLines->ForegroundLineWidth, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&HorizontalLines->BackgroundLineWidth, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteDiagonalLines(DLPC34XX_DUAL_DiagonalLines_s *DiagonalLines)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB);
    DLPC_COMMON_SetBits(4, 4, 0);     // PatternSelect
    DLPC_COMMON_SetBits((int32_t)DiagonalLines->Border, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)DiagonalLines->BackgroundColor, 3, 0);
    DLPC_COMMON_SetBits((int32_t)DiagonalLines->ForegroundColor, 3, 4);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&DiagonalLines->HorizontalSpacing, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&DiagonalLines->VerticalSpacing, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteVerticalLines(DLPC34XX_DUAL_VerticalLines_s *VerticalLines)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB);
    DLPC_COMMON_SetBits(5, 4, 0);     // PatternSelect
    DLPC_COMMON_SetBits((int32_t)VerticalLines->Border, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)VerticalLines->BackgroundColor, 3, 0);
    DLPC_COMMON_SetBits((int32_t)VerticalLines->ForegroundColor, 3, 4);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&VerticalLines->ForegroundLineWidth, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&VerticalLines->BackgroundLineWidth, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteGridLines(DLPC34XX_DUAL_GridLines_s *GridLines)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB);
    DLPC_COMMON_SetBits(6, 4, 0);     // PatternSelect
    DLPC_COMMON_SetBits((int32_t)GridLines->Border, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)GridLines->BackgroundColor, 3, 0);
    DLPC_COMMON_SetBits((int32_t)GridLines->ForegroundColor, 3, 4);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&GridLines->HorizontalForegroundLineWidth, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&GridLines->HorizontalBackgroundLineWidth, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&GridLines->VerticalForegroundLineWidth, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&GridLines->VerticalBackgroundLineWidth, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteCheckerboard(DLPC34XX_DUAL_Checkerboard_s *Checkerboard)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB);
    DLPC_COMMON_SetBits(7, 4, 0);     // PatternSelect
    DLPC_COMMON_SetBits((int32_t)Checkerboard->Border, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)Checkerboard->BackgroundColor, 3, 0);
    DLPC_COMMON_SetBits((int32_t)Checkerboard->ForegroundColor, 3, 4);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&Checkerboard->HorizontalCheckerCount, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&Checkerboard->VerticalCheckerCount, 2);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteColorbars(DLPC34XX_DUAL_BorderEnable_e Border)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB);
    DLPC_COMMON_SetBits(8, 4, 0);     // PatternSelect
    DLPC_COMMON_SetBits((int32_t)Border, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadTestPatternSelect(DLPC34XX_DUAL_TestPatternSelect_s *TestPatternSelect)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        TestPatternSelect->PatternSelect = (DLPC34XX_DUAL_TestPattern_e)DLPC_COMMON_GetBits(4, 0, false);
        TestPatternSelect->Border = (DLPC34XX_DUAL_BorderEnable_e)DLPC_COMMON_GetBits(1, 7, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        TestPatternSelect->BackgroundColor = (DLPC34XX_DUAL_Color_e)DLPC_COMMON_GetBits(4, 0, false);
        TestPatternSelect->ForegroundColor = (DLPC34XX_DUAL_Color_e)DLPC_COMMON_GetBits(4, 4, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        TestPatternSelect->StartValue = (uint8_t)DLPC_COMMON_GetBits(8, 0, false);
        TestPatternSelect->EndValue = (uint8_t)DLPC_COMMON_GetBits(8, 8, false);
        TestPatternSelect->ForegroundLineWidth = (uint8_t)DLPC_COMMON_GetBits(8, 0, false);
        TestPatternSelect->BackgroundLineWidth = (uint8_t)DLPC_COMMON_GetBits(8, 8, false);
        TestPatternSelect->HorizontalSpacing = (uint8_t)DLPC_COMMON_GetBits(8, 0, false);
        TestPatternSelect->VerticalSpacing = (uint8_t)DLPC_COMMON_GetBits(8, 8, false);
        TestPatternSelect->HorizontalForegroundLineWidth = (uint8_t)DLPC_COMMON_GetBits(8, 0, false);
        TestPatternSelect->HorizontalBackgroundLineWidth = (uint8_t)DLPC_COMMON_GetBits(8, 8, false);
        TestPatternSelect->HorizontalCheckerCount = (uint16_t)DLPC_COMMON_GetBits(11, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(2);
        TestPatternSelect->VerticalForegroundLineWidth = (uint8_t)DLPC_COMMON_GetBits(8, 0, false);
        TestPatternSelect->VerticalBackgroundLineWidth = (uint8_t)DLPC_COMMON_GetBits(8, 8, false);
        TestPatternSelect->VerticalCheckerCount = (uint16_t)DLPC_COMMON_GetBits(11, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(2);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteExecuteFlashBatchFile(uint8_t BatchFileNumber)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x2D);
    DLPC_COMMON_PackBytes((uint8_t*)&BatchFileNumber, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteBatchFileDelay(uint16_t DelayInMicroseconds)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xDB);
    DLPC_COMMON_PackBytes((uint8_t*)&DelayInMicroseconds, 2);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteLedOutputControlMethod(DLPC34XX_DUAL_LedControlMethod_e LedControlMethod)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x50);
    DLPC_COMMON_PackBytes((uint8_t*)&LedControlMethod, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadLedOutputControlMethod(DLPC34XX_DUAL_LedControlMethod_e *LedControlMethod)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x51);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *LedControlMethod = (DLPC34XX_DUAL_LedControlMethod_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteRgbLedEnable(bool RedLedEnable, bool GreenLedEnable, bool BlueLedEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x52);
    DLPC_COMMON_SetBits((int32_t)RedLedEnable, 1, 0);
    DLPC_COMMON_SetBits((int32_t)GreenLedEnable, 1, 1);
    DLPC_COMMON_SetBits((int32_t)BlueLedEnable, 1, 2);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadRgbLedEnable(bool *RedLedEnable, bool *GreenLedEnable, bool *BlueLedEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x53);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *RedLedEnable = DLPC_COMMON_GetBits(1, 0, false);
        *GreenLedEnable = DLPC_COMMON_GetBits(1, 1, false);
        *BlueLedEnable = DLPC_COMMON_GetBits(1, 2, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteRgbLedCurrent(uint16_t RedLedCurrent, uint16_t GreenLedCurrent, uint16_t BlueLedCurrent)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x54);
    DLPC_COMMON_PackBytes((uint8_t*)&RedLedCurrent, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&GreenLedCurrent, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&BlueLedCurrent, 2);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadRgbLedCurrent(uint16_t *RedLedCurrent, uint16_t *GreenLedCurrent, uint16_t *BlueLedCurrent)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x55);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        *RedLedCurrent = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *GreenLedCurrent = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *BlueLedCurrent = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadCaicLedMaxAvailablePower(double *MaxLedPower)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x57);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *MaxLedPower = DLPC_COMMON_UnpackFloat(2, 100, false);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteRgbLedMaxCurrent(uint16_t MaxRedLedCurrent, uint16_t MaxGreenLedCurrent, uint16_t MaxBlueLedCurrent)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x5C);
    DLPC_COMMON_PackBytes((uint8_t*)&MaxRedLedCurrent, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&MaxGreenLedCurrent, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&MaxBlueLedCurrent, 2);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadRgbLedMaxCurrent(uint16_t *MaxRedLedCurrent, uint16_t *MaxGreenLedCurrent, uint16_t *MaxBlueLedCurrent)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x5D);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        *MaxRedLedCurrent = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *MaxGreenLedCurrent = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *MaxBlueLedCurrent = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadCaicRgbLedCurrent(uint16_t *RedLedCurrent, uint16_t *GreenLedCurrent, uint16_t *BlueLedCurrent)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x5F);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        *RedLedCurrent = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *GreenLedCurrent = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *BlueLedCurrent = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteLookSelect(uint8_t LookNumber)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x22);
    DLPC_COMMON_PackBytes((uint8_t*)&LookNumber, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadLookSelect(uint8_t *LookNumber, uint8_t *SequenceIndex, double *SequenceFrameTime)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x23);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        *LookNumber = *(DLPC_COMMON_UnpackBytes(1));
        *SequenceIndex = *(DLPC_COMMON_UnpackBytes(1));
        *SequenceFrameTime = DLPC_COMMON_UnpackFloat(4, 15, false);
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadSequenceHeaderAttributes(DLPC34XX_DUAL_SequenceHeaderAttributes_s *SequenceHeaderAttributes)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x26);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(30);
    if (Status == 0)
    {
        SequenceHeaderAttributes->LookRedDutyCycle = DLPC_COMMON_UnpackFloat(2, 255, false);
        SequenceHeaderAttributes->LookGreenDutyCycle = DLPC_COMMON_UnpackFloat(2, 255, false);
        SequenceHeaderAttributes->LookBlueDutyCycle = DLPC_COMMON_UnpackFloat(2, 255, false);
        SequenceHeaderAttributes->LookMaxFrameTime = DLPC_COMMON_UnpackFloat(4, 15, false);
        SequenceHeaderAttributes->LookMinFrameTime = DLPC_COMMON_UnpackFloat(4, 15, false);
        SequenceHeaderAttributes->LookMaxSequenceVectors = *(DLPC_COMMON_UnpackBytes(1));
        SequenceHeaderAttributes->SeqRedDutyCycle = DLPC_COMMON_UnpackFloat(2, 255, false);
        SequenceHeaderAttributes->SeqGreenDutyCycle = DLPC_COMMON_UnpackFloat(2, 255, false);
        SequenceHeaderAttributes->SeqBlueDutyCycle = DLPC_COMMON_UnpackFloat(2, 255, false);
        SequenceHeaderAttributes->SeqMaxFrameTime = DLPC_COMMON_UnpackFloat(4, 15, false);
        SequenceHeaderAttributes->SeqMinFrameTime = DLPC_COMMON_UnpackFloat(4, 15, false);
        SequenceHeaderAttributes->SeqMaxSequenceVectors = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteLocalAreaBrightnessBoostControl(DLPC34XX_DUAL_LabbControl_e LabbControl, uint8_t SharpnessStrength, uint8_t LabbStrengthSetting)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x80);
    DLPC_COMMON_SetBits((int32_t)LabbControl, 2, 0);
    DLPC_COMMON_SetBits((int32_t)SharpnessStrength, 4, 4);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&LabbStrengthSetting, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadLocalAreaBrightnessBoostControl(DLPC34XX_DUAL_LabbControl_e *LabbControl, uint8_t *SharpnessStrength, uint8_t *LabbStrengthSetting, uint8_t *LabbGainValue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x81);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(3);
    if (Status == 0)
    {
        *LabbControl = (DLPC34XX_DUAL_LabbControl_e)DLPC_COMMON_GetBits(2, 0, false);
        *SharpnessStrength = (uint8_t)DLPC_COMMON_GetBits(4, 4, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        *LabbStrengthSetting = *(DLPC_COMMON_UnpackBytes(1));
        *LabbGainValue = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteCaicImageProcessingControl(DLPC34XX_DUAL_CaicGainDisplayScale_e CaicGainDisplayScale, bool CaicGainDisplayEnable, double CaicMaxLumensGain, double CaicClippingThreshold)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x84);
    DLPC_COMMON_SetBits((int32_t)CaicGainDisplayScale, 1, 6);
    DLPC_COMMON_SetBits((int32_t)CaicGainDisplayEnable, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackFloat(CaicMaxLumensGain, 1, 31);
    DLPC_COMMON_PackFloat(CaicClippingThreshold, 1, 63);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadCaicImageProcessingControl(DLPC34XX_DUAL_CaicGainDisplayScale_e *CaicGainDisplayScale, bool *CaicGainDisplayEnable, double *CaicMaxLumensGain, double *CaicClippingThreshold)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x85);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(3);
    if (Status == 0)
    {
        *CaicGainDisplayScale = (DLPC34XX_DUAL_CaicGainDisplayScale_e)DLPC_COMMON_GetBits(1, 6, false);
        *CaicGainDisplayEnable = DLPC_COMMON_GetBits(1, 7, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        *CaicMaxLumensGain = DLPC_COMMON_UnpackFloat(1, 31, false);
        *CaicClippingThreshold = DLPC_COMMON_UnpackFloat(1, 63, false);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteColorCoordinateAdjustmentControl(bool CcaEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x86);
    DLPC_COMMON_PackBytes((uint8_t*)&CcaEnable, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadColorCoordinateAdjustmentControl(bool *CcaEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x87);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *CcaEnable = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadShortStatus(DLPC34XX_DUAL_ShortStatus_s *ShortStatus)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xD0);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        ShortStatus->SystemInitialized = (DLPC34XX_DUAL_SystemInit_e)DLPC_COMMON_GetBits(1, 0, false);
        ShortStatus->CommunicationError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 1, false);
        ShortStatus->SystemError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 3, false);
        ShortStatus->FlashEraseComplete = (DLPC34XX_DUAL_FlashErase_e)DLPC_COMMON_GetBits(1, 4, false);
        ShortStatus->FlashError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 5, false);
        ShortStatus->SensingSequenceError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 6, false);
        ShortStatus->Application = (DLPC34XX_DUAL_Application_e)DLPC_COMMON_GetBits(1, 7, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadSystemStatus(DLPC34XX_DUAL_SystemStatus_s *SystemStatus)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xD1);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        SystemStatus->DmdDeviceError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 0, false);
        SystemStatus->DmdInterfaceError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 1, false);
        SystemStatus->DmdTrainingError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 1, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        SystemStatus->RedLedEnableState = (DLPC34XX_DUAL_LedState_e)DLPC_COMMON_GetBits(1, 0, false);
        SystemStatus->GreenLedEnableState = (DLPC34XX_DUAL_LedState_e)DLPC_COMMON_GetBits(1, 1, false);
        SystemStatus->BlueLedEnableState = (DLPC34XX_DUAL_LedState_e)DLPC_COMMON_GetBits(1, 2, false);
        SystemStatus->RedLedError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 3, false);
        SystemStatus->GreenLedError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 4, false);
        SystemStatus->BlueLedError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 5, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        SystemStatus->SequenceAbortError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 0, false);
        SystemStatus->SequenceError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 1, false);
        SystemStatus->DcPowerSupply = (DLPC34XX_DUAL_PowerSupply_e)DLPC_COMMON_GetBits(1, 2, false);
        SystemStatus->SensingError = (DLPC34XX_DUAL_SensingError_e)DLPC_COMMON_GetBits(5, 3, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        SystemStatus->ControllerConfiguration = (DLPC34XX_DUAL_ControllerConfiguration_e)DLPC_COMMON_GetBits(1, 2, false);
        SystemStatus->MasterOrSlaveOperation = (DLPC34XX_DUAL_MasterOrSlaveOperation_e)DLPC_COMMON_GetBits(1, 3, false);
        SystemStatus->ProductConfigurationError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 4, false);
        SystemStatus->WatchdogTimerTimeout = (DLPC34XX_DUAL_WatchdogTimeout_e)DLPC_COMMON_GetBits(1, 5, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadCommunicationStatus(DLPC34XX_DUAL_CommunicationStatus_s *CommunicationStatus)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xD3);
    DLPC_COMMON_PackByte(0x2);     // CommandBusStatusSelection

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        CommunicationStatus->InvalidCommandError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 0, false);
        CommunicationStatus->InvalidCommandParameterValue = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 1, false);
        CommunicationStatus->CommandProcessingError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 2, false);
        CommunicationStatus->FlashBatchFileError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 3, false);
        CommunicationStatus->ReadCommandError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 4, false);
        CommunicationStatus->InvalidNumberOfCommandParameters = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 5, false);
        CommunicationStatus->BusTimeoutByDisplayError = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 6, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        CommunicationStatus->AbortedOpCode = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadSystemSoftwareVersion(uint16_t *PatchVersion, uint8_t *MinorVersion, uint8_t *MajorVersion)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xD2);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *PatchVersion = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *MinorVersion = *(DLPC_COMMON_UnpackBytes(1));
        *MajorVersion = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadControllerDeviceId(DLPC34XX_DUAL_ControllerDeviceId_e *DeviceId)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xD4);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *DeviceId = (DLPC34XX_DUAL_ControllerDeviceId_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadDmdDeviceId(DLPC34XX_DUAL_DmdDataSelection_e DmdDataSelection, uint32_t *DeviceId)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xD5);
    DLPC_COMMON_PackBytes((uint8_t*)&DmdDataSelection, 1);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *DeviceId = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadFirmwareBuildVersion(uint16_t *PatchVersion, uint8_t *MinorVersion, uint8_t *MajorVersion)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xD9);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *PatchVersion = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *MinorVersion = *(DLPC_COMMON_UnpackBytes(1));
        *MajorVersion = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadSystemTemperature(double *Temperature)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xD6);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *Temperature = DLPC_COMMON_UnpackFloat(2, 10, false);
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadFlashUpdatePrecheck(uint32_t FlashUpdatePackageSize, DLPC34XX_DUAL_Error_e *PackageSizeStatus, DLPC34XX_DUAL_Error_e *PacakgeConfigurationCollapsed, DLPC34XX_DUAL_Error_e *PacakgeConfigurationIdentifier)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xDD);
    DLPC_COMMON_PackBytes((uint8_t*)&FlashUpdatePackageSize, 4);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *PackageSizeStatus = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 0, false);
        *PacakgeConfigurationCollapsed = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 1, false);
        *PacakgeConfigurationIdentifier = (DLPC34XX_DUAL_Error_e)DLPC_COMMON_GetBits(1, 2, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteFlashDataTypeSelect(DLPC34XX_DUAL_FlashDataTypeSelect_e FlashSelect)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xDE);
    DLPC_COMMON_PackBytes((uint8_t*)&FlashSelect, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteFlashDataLength(uint16_t FlashDataLength)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xDF);
    DLPC_COMMON_PackBytes((uint8_t*)&FlashDataLength, 2);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteFlashErase()
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE0);
    DLPC_COMMON_PackByte(0xAA);     // Signature
    DLPC_COMMON_PackByte(0xBB);     // Signature
    DLPC_COMMON_PackByte(0xCC);     // Signature
    DLPC_COMMON_PackByte(0xDD);     // Signature

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteFlashStart(uint16_t DataLength, uint8_t* Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE1);
    DLPC_COMMON_PackBytes((uint8_t*)Data, DataLength);     // Data

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadFlashStart(uint16_t Length, uint8_t *Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE3);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(Length);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < Length; s_Index++)
        {
            Data[s_Index] = *(DLPC_COMMON_UnpackBytes(1));
        }
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteFlashContinue(uint16_t DataLength, uint8_t* Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE2);
    DLPC_COMMON_PackBytes((uint8_t*)Data, DataLength);     // Data

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadFlashContinue(uint16_t Length, uint8_t *Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE4);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(Length);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < Length; s_Index++)
        {
            Data[s_Index] = *(DLPC_COMMON_UnpackBytes(1));
        }
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadSequenceBinaryVersion(uint8_t *PatchVersion, uint8_t *MinorVersion, uint8_t *MajorVersion)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x9B);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        DLPC_COMMON_MoveReadBufferPointer(1); // Reserved
        *PatchVersion = *(DLPC_COMMON_UnpackBytes(1));
        *MinorVersion = *(DLPC_COMMON_UnpackBytes(1));
        *MajorVersion = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteInternalPatternControl(DLPC34XX_DUAL_PatternControl_e PatternControl, uint8_t RepeatCount)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x9E);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternControl, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&RepeatCount, 1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadValidateExposureTime(DLPC34XX_DUAL_PatternMode_e PatternMode, DLPC34XX_DUAL_SequenceType_e BitDepth, uint32_t ExposureTime, DLPC34XX_DUAL_ValidateExposureTime_s *ValidateExposureTime)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x9D);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternMode, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&BitDepth, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&ExposureTime, 4);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(13);
    if (Status == 0)
    {
        ValidateExposureTime->ExposureTimeSupported = (DLPC34XX_DUAL_ExposureTimeSupported_e)DLPC_COMMON_GetBits(1, 0, false);
        ValidateExposureTime->ZeroDarkTimeSupported = (DLPC34XX_DUAL_ZeroDarkTimeSupported_e)DLPC_COMMON_GetBits(1, 4, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        ValidateExposureTime->MinimumExposureTime = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        ValidateExposureTime->PreExposureDarkTime = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        ValidateExposureTime->PostExposureDarkTime = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteTriggerInConfiguration(DLPC34XX_DUAL_TriggerEnable_e TriggerEnable, DLPC34XX_DUAL_TriggerPolarity_e TriggerPolarity)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x90);
    DLPC_COMMON_SetBits((int32_t)TriggerEnable, 1, 0);
    DLPC_COMMON_SetBits((int32_t)TriggerPolarity, 1, 1);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadTriggerInConfiguration(DLPC34XX_DUAL_TriggerEnable_e *TriggerEnable, DLPC34XX_DUAL_TriggerPolarity_e *TriggerPolarity)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x91);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *TriggerEnable = (DLPC34XX_DUAL_TriggerEnable_e)DLPC_COMMON_GetBits(1, 0, false);
        *TriggerPolarity = (DLPC34XX_DUAL_TriggerPolarity_e)DLPC_COMMON_GetBits(1, 1, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WriteTriggerOutConfiguration(DLPC34XX_DUAL_TriggerType_e TriggerType, DLPC34XX_DUAL_TriggerEnable_e TriggerEnable, DLPC34XX_DUAL_TriggerInversion_e TriggerInversion, int32_t Delay)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x92);
    DLPC_COMMON_SetBits((int32_t)TriggerType, 1, 0);
    DLPC_COMMON_SetBits((int32_t)TriggerEnable, 1, 1);
    DLPC_COMMON_SetBits((int32_t)TriggerInversion, 1, 2);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&Delay, 4);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadTriggerOutConfiguration(DLPC34XX_DUAL_TriggerType_e Trigger, DLPC34XX_DUAL_TriggerEnable_e *TriggerEnable, DLPC34XX_DUAL_TriggerInversion_e *TriggerInversion, int32_t *Delay)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x93);
    DLPC_COMMON_PackBytes((uint8_t*)&Trigger, 1);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(5);
    if (Status == 0)
    {
        *TriggerEnable = (DLPC34XX_DUAL_TriggerEnable_e)DLPC_COMMON_GetBits(1, 0, false);
        *TriggerInversion = (DLPC34XX_DUAL_TriggerInversion_e)DLPC_COMMON_GetBits(1, 1, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        *Delay = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WritePatternReadyConfiguration(DLPC34XX_DUAL_TriggerEnable_e TriggerEnable, DLPC34XX_DUAL_TriggerPolarity_e TriggerPolarity)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x94);
    DLPC_COMMON_SetBits((int32_t)TriggerEnable, 1, 0);
    DLPC_COMMON_SetBits((int32_t)TriggerPolarity, 1, 1);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadPatternReadyConfiguration(DLPC34XX_DUAL_TriggerEnable_e *TriggerEnable, DLPC34XX_DUAL_TriggerPolarity_e *TriggerPolarity)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x95);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *TriggerEnable = (DLPC34XX_DUAL_TriggerEnable_e)DLPC_COMMON_GetBits(1, 0, false);
        *TriggerPolarity = (DLPC34XX_DUAL_TriggerPolarity_e)DLPC_COMMON_GetBits(1, 1, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WritePatternConfiguration(DLPC34XX_DUAL_PatternConfiguration_s *PatternConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x96);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternConfiguration->SequenceType, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternConfiguration->NumberOfPatterns, 1);
    DLPC_COMMON_SetBits((int32_t)PatternConfiguration->RedIlluminator, 1, 0);
    DLPC_COMMON_SetBits((int32_t)PatternConfiguration->GreenIlluminator, 1, 1);
    DLPC_COMMON_SetBits((int32_t)PatternConfiguration->BlueIlluminator, 1, 2);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternConfiguration->IlluminationTime, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternConfiguration->PreIlluminationDarkTime, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternConfiguration->PostIlluminationDarkTime, 4);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadPatternConfiguration(DLPC34XX_DUAL_PatternConfiguration_s *PatternConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x97);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(15);
    if (Status == 0)
    {
        PatternConfiguration->SequenceType = (DLPC34XX_DUAL_SequenceType_e)*(DLPC_COMMON_UnpackBytes(1));
        PatternConfiguration->NumberOfPatterns = *(DLPC_COMMON_UnpackBytes(1));
        PatternConfiguration->RedIlluminator = (DLPC34XX_DUAL_IlluminatorEnable_e)DLPC_COMMON_GetBits(1, 0, false);
        PatternConfiguration->GreenIlluminator = (DLPC34XX_DUAL_IlluminatorEnable_e)DLPC_COMMON_GetBits(1, 1, false);
        PatternConfiguration->BlueIlluminator = (DLPC34XX_DUAL_IlluminatorEnable_e)DLPC_COMMON_GetBits(1, 2, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        PatternConfiguration->IlluminationTime = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        PatternConfiguration->PreIlluminationDarkTime = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        PatternConfiguration->PostIlluminationDarkTime = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}

uint32_t DLPC34XX_DUAL_WritePatternOrderTableEntry(DLPC34XX_DUAL_WriteControl_e WriteControl, DLPC34XX_DUAL_PatternOrderTableEntry_s *PatternOrderTableEntry)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x98);
    DLPC_COMMON_PackBytes((uint8_t*)&WriteControl, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternOrderTableEntry->PatSetIndex, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternOrderTableEntry->NumberOfPatternsToDisplay, 1);
    DLPC_COMMON_SetBits((int32_t)PatternOrderTableEntry->RedIlluminator, 1, 0);
    DLPC_COMMON_SetBits((int32_t)PatternOrderTableEntry->GreenIlluminator, 1, 1);
    DLPC_COMMON_SetBits((int32_t)PatternOrderTableEntry->BlueIlluminator, 1, 2);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternOrderTableEntry->PatternInvertLsword, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternOrderTableEntry->PatternInvertMsword, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternOrderTableEntry->IlluminationTime, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternOrderTableEntry->PreIlluminationDarkTime, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternOrderTableEntry->PostIlluminationDarkTime, 4);

    DLPC_COMMON_SetCommandDestination(0);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadPatternOrderTableEntry(uint8_t PatternOrderTableEntryIndex, DLPC34XX_DUAL_PatternOrderTableEntry_s *PatternOrderTableEntry)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x99);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternOrderTableEntryIndex, 1);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(23);
    if (Status == 0)
    {
        PatternOrderTableEntry->PatSetIndex = *(DLPC_COMMON_UnpackBytes(1));
        PatternOrderTableEntry->NumberOfPatternsToDisplay = *(DLPC_COMMON_UnpackBytes(1));
        PatternOrderTableEntry->RedIlluminator = (DLPC34XX_DUAL_IlluminatorEnable_e)DLPC_COMMON_GetBits(1, 0, false);
        PatternOrderTableEntry->GreenIlluminator = (DLPC34XX_DUAL_IlluminatorEnable_e)DLPC_COMMON_GetBits(1, 1, false);
        PatternOrderTableEntry->BlueIlluminator = (DLPC34XX_DUAL_IlluminatorEnable_e)DLPC_COMMON_GetBits(1, 2, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        PatternOrderTableEntry->PatternInvertLsword = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        PatternOrderTableEntry->PatternInvertMsword = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        PatternOrderTableEntry->IlluminationTime = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        PatternOrderTableEntry->PreIlluminationDarkTime = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        PatternOrderTableEntry->PostIlluminationDarkTime = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}
 
uint32_t DLPC34XX_DUAL_ReadInternalPatternStatus(DLPC34XX_DUAL_InternalPatternStatus_s *InternalPatternStatus)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x9F);

    DLPC_COMMON_SetCommandDestination(0);

    Status = DLPC_COMMON_SendRead(7);
    if (Status == 0)
    {
        InternalPatternStatus->PatternReadyStatus = (DLPC34XX_DUAL_PatternReadyStatus_e)*(DLPC_COMMON_UnpackBytes(1));
        InternalPatternStatus->NumPatOrderTableEntries = *(DLPC_COMMON_UnpackBytes(1));
        InternalPatternStatus->CurrentPatOrderEntryIndex = *(DLPC_COMMON_UnpackBytes(1));
        InternalPatternStatus->CurrentPatSetIndex = *(DLPC_COMMON_UnpackBytes(1));
        InternalPatternStatus->NumPatInCurrentPatSet = *(DLPC_COMMON_UnpackBytes(1));
        InternalPatternStatus->NumPatDisplayedFromPatSet = *(DLPC_COMMON_UnpackBytes(1));
        InternalPatternStatus->NextPatSetIndex = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
