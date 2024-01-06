/*------------------------------------------------------------------------------
 * Copyright (c) 2020 Texas Instruments Incorporated - http://www.ti.com/
 *------------------------------------------------------------------------------
 *
 * NOTE: This file is auto generated from a command definition file.
 *       Please do not modify the file directly.                    
 *
 * Command Spec Version : 0.1
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
 * \brief Bootloader commands, Projector Control, Formatter Only Commands
 */

#include "stdlib.h"
#include "string.h"
#include "dlpc654x.h"
#include "dlpc_common_private.h"

static uint32_t s_Index;
 
uint32_t DLPC654X_ReadMode(DLPC654X_CmdModeT_e *AppMode, DLPC654X_CmdControllerConfigT_e *ControllerConfig)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x0);

    DLPC_COMMON_SetCommandDestination(1);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *AppMode = (DLPC654X_CmdModeT_e)DLPC_COMMON_GetBits(1, 0, false);
        *ControllerConfig = (DLPC654X_CmdControllerConfigT_e)DLPC_COMMON_GetBits(1, 1, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}
 
uint32_t DLPC654X_ReadVersion(DLPC654X_Version_s *Version)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1);

    DLPC_COMMON_SetCommandDestination(1);

    Status = DLPC_COMMON_SendRead(8);
    if (Status == 0)
    {
        Version->AppMajor = *(DLPC_COMMON_UnpackBytes(1));
        Version->AppMinor = *(DLPC_COMMON_UnpackBytes(1));
        Version->AppPatch = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Version->ApiMajor = *(DLPC_COMMON_UnpackBytes(1));
        Version->ApiMinor = *(DLPC_COMMON_UnpackBytes(1));
        Version->ApiPatch = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteSwitchMode(DLPC654X_CmdSwitchTypeT_e SwitchMode)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x2);
    DLPC_COMMON_PackBytes((uint8_t*)&SwitchMode, 1);

    DLPC_COMMON_SetCommandDestination(1);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadBootHoldReason(uint8_t *ReasonCode)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x12);

    DLPC_COMMON_SetCommandDestination(1);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *ReasonCode = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadFlashId(uint16_t *ManfId, uint64_t *DevId)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x20);

    DLPC_COMMON_SetCommandDestination(1);

    Status = DLPC_COMMON_SendRead(10);
    if (Status == 0)
    {
        *ManfId = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *DevId = *((uint64_t*)DLPC_COMMON_UnpackBytes(8));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadGetFlashSectorInformation(DLPC654X_SectorInfo_s *SectorInfo[])
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x21);

    DLPC_COMMON_SetCommandDestination(1);

    Status = DLPC_COMMON_SendRead(0xFFFF);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < (uint32_t)(DLPC_COMMON_GetBytesRead() / 6); s_Index++)
        {
            SectorInfo[s_Index]->SectorSize = (uint32_t)DLPC_COMMON_GetBits(32, 0, false);
            SectorInfo[s_Index]->NumSectors = (uint16_t)DLPC_COMMON_GetBits(16, 32, false);
            DLPC_COMMON_MoveReadBufferPointer(6);
        }
    }
    return Status;
}

uint32_t DLPC654X_WriteUnlockFlashForUpdate(DLPC654X_CmdFlashUpdateT_e Unlock)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x22);
    DLPC_COMMON_PackBytes((uint8_t*)&Unlock, 4);

    DLPC_COMMON_SetCommandDestination(1);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadUnlockFlashForUpdate(uint8_t *IsUnlocked)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x22);

    DLPC_COMMON_SetCommandDestination(1);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *IsUnlocked = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteEraseSector(uint32_t SectorAddress)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x23);
    DLPC_COMMON_PackBytes((uint8_t*)&SectorAddress, 4);

    DLPC_COMMON_SetCommandDestination(1);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC654X_WriteInitializeFlashReadWriteSettings(uint32_t StartAddress, uint32_t NumBytes)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x24);
    DLPC_COMMON_PackBytes((uint8_t*)&StartAddress, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&NumBytes, 4);

    DLPC_COMMON_SetCommandDestination(1);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC654X_WriteFlashWrite(uint16_t DataLength, uint8_t* Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x25);
    DLPC_COMMON_PackBytes((uint8_t*)Data, DataLength);     // Data

    DLPC_COMMON_SetCommandDestination(1);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadFlashWrite(uint16_t NumBytesToRead, uint8_t *Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x25);
    DLPC_COMMON_PackBytes((uint8_t*)&NumBytesToRead, 2);

    DLPC_COMMON_SetCommandDestination(1);

    Status = DLPC_COMMON_SendRead(0xFFFF);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < 1; s_Index++)
        {
            Data[s_Index] = *(DLPC_COMMON_UnpackBytes(1));
        }
    }
    return Status;
}
 
uint32_t DLPC654X_ReadChecksum(uint32_t StartAddress, uint32_t NumBytes, uint32_t *SimpleChecksum, uint32_t *SumofSumChecksum)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x26);
    DLPC_COMMON_PackBytes((uint8_t*)&StartAddress, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&NumBytes, 4);

    DLPC_COMMON_SetCommandDestination(1);

    Status = DLPC_COMMON_SendRead(8);
    if (Status == 0)
    {
        *SimpleChecksum = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        *SumofSumChecksum = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}

uint32_t DLPC654X_WriteMemory(uint32_t Address, uint32_t Value)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x10);
    DLPC_COMMON_PackBytes((uint8_t*)&Address, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&Value, 4);

    DLPC_COMMON_SetCommandDestination(1);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadMemory(uint32_t Address, uint32_t *Value)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x10);
    DLPC_COMMON_PackBytes((uint8_t*)&Address, 4);

    DLPC_COMMON_SetCommandDestination(1);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *Value = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}

uint32_t DLPC654X_WriteMemoryArray(uint16_t DataLength, DLPC654X_MemoryArray_s *MemoryArray)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x11);
    DLPC_COMMON_PackBytes((uint8_t*)&MemoryArray->StartAddress, 4);
    DLPC_COMMON_SetBits((int32_t)MemoryArray->AddressIncrement, 6, 0);
    DLPC_COMMON_SetBits((int32_t)MemoryArray->AccessWidth, 2, 6);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&MemoryArray->NumberOfWords, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&MemoryArray->NumberOfBytesPerWord, 1);
    DLPC_COMMON_PackBytes((uint8_t*)MemoryArray->Data, DataLength);     // Data

    DLPC_COMMON_SetCommandDestination(1);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadMemoryArray(uint32_t StartAddress, uint8_t AddressIncrement, DLPC654X_CmdAccessWidthT_e AccessWidth, uint16_t NumberOfWords, uint8_t NumberOfBytesPerWord, uint8_t *Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x11);
    DLPC_COMMON_PackBytes((uint8_t*)&StartAddress, 4);
    DLPC_COMMON_SetBits((int32_t)AddressIncrement, 6, 0);
    DLPC_COMMON_SetBits((int32_t)AccessWidth, 2, 6);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&NumberOfWords, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&NumberOfBytesPerWord, 1);

    DLPC_COMMON_SetCommandDestination(1);

    Status = DLPC_COMMON_SendRead(0xFFFF);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < 1; s_Index++)
        {
            Data[s_Index] = *(DLPC_COMMON_UnpackBytes(1));
        }
    }
    return Status;
}

uint32_t DLPC654X_WriteSystemLook(uint16_t System)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x13);
    DLPC_COMMON_PackBytes((uint8_t*)&System, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadSystemLook(uint16_t *System)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x13);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *System = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteAspectRatio(DLPC654X_DispfmtAspectRatioT_e AspectRatio)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x20);
    DLPC_COMMON_PackBytes((uint8_t*)&AspectRatio, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadAspectRatio(DLPC654X_DispfmtAspectRatioT_e *AspectRatio)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x20);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *AspectRatio = (DLPC654X_DispfmtAspectRatioT_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadSfgResolution(uint16_t *HorizontalResolution, uint16_t *VerticalResolution)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x19);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *HorizontalResolution = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *VerticalResolution = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteTpgPredefinedPattern(uint8_t PatternNumber)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x14);
    DLPC_COMMON_PackBytes((uint8_t*)&PatternNumber, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadTpgPredefinedPattern(uint8_t *PatternNumber)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x14);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *PatternNumber = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadControllerInformation(uint8_t *ControllerId)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x0);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *ControllerId = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadDmdDeviceId(uint32_t *DmdId)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *DmdId = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadDmdResolution(uint16_t *Width, uint16_t *Height)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x2);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *Width = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *Height = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadFlashVersion(uint8_t *FlashVersionMajor, uint8_t *FlashVersionMinor, uint8_t *FlashVersionSubminor)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x3);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(3);
    if (Status == 0)
    {
        *FlashVersionMajor = *(DLPC_COMMON_UnpackBytes(1));
        *FlashVersionMinor = *(DLPC_COMMON_UnpackBytes(1));
        *FlashVersionSubminor = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadFlashLayoutVersion(uint16_t *FlashCfgLayoutVersion, uint8_t* *FlashCfgLayoutHash, uint16_t *AppCfgLayoutVersion, uint8_t* *AppCfgLayoutHash)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(68);
    if (Status == 0)
    {
        *FlashCfgLayoutVersion = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *FlashCfgLayoutHash = (DLPC_COMMON_UnpackBytes(32));
        *AppCfgLayoutVersion = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *AppCfgLayoutHash = (DLPC_COMMON_UnpackBytes(32));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadSystemStatus(DLPC654X_SystemStatus_s *SystemStatus)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x6);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(12);
    if (Status == 0)
    {
        SystemStatus->CwSpinning = DLPC_COMMON_GetBits(1, 0, false);
        SystemStatus->CwPhaselock = DLPC_COMMON_GetBits(1, 1, false);
        SystemStatus->CwFreqlock = DLPC_COMMON_GetBits(1, 2, false);
        SystemStatus->Lamplit = DLPC_COMMON_GetBits(1, 3, false);
        SystemStatus->MemTstPassed = DLPC_COMMON_GetBits(1, 4, false);
        SystemStatus->FrameRateConvEn = DLPC_COMMON_GetBits(1, 10, false);
        SystemStatus->SeqPhaselock = DLPC_COMMON_GetBits(1, 11, false);
        SystemStatus->SeqFreqlock = DLPC_COMMON_GetBits(1, 12, false);
        SystemStatus->SeqSearch = DLPC_COMMON_GetBits(1, 13, false);
        SystemStatus->ScpcalEnable = DLPC_COMMON_GetBits(1, 29, false);
        SystemStatus->VicalEnable = DLPC_COMMON_GetBits(1, 30, false);
        SystemStatus->BccalEnable = DLPC_COMMON_GetBits(1, 31, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
        SystemStatus->SequenceErr = DLPC_COMMON_GetBits(1, 0, false);
        SystemStatus->PixclkOor = DLPC_COMMON_GetBits(1, 1, false);
        SystemStatus->SyncvalStat = DLPC_COMMON_GetBits(1, 2, false);
        SystemStatus->UartPort0CommErr = DLPC_COMMON_GetBits(1, 6, false);
        SystemStatus->UartPort1CommErr = DLPC_COMMON_GetBits(1, 7, false);
        SystemStatus->UartPort2CommErr = DLPC_COMMON_GetBits(1, 8, false);
        SystemStatus->SspPort0CommErr = DLPC_COMMON_GetBits(1, 9, false);
        SystemStatus->SspPort1CommErr = DLPC_COMMON_GetBits(1, 10, false);
        SystemStatus->SspPort2CommErr = DLPC_COMMON_GetBits(1, 11, false);
        SystemStatus->I2CPort0CommErr = DLPC_COMMON_GetBits(1, 12, false);
        SystemStatus->I2CPort1CommErr = DLPC_COMMON_GetBits(1, 13, false);
        SystemStatus->I2CPort2CommErr = DLPC_COMMON_GetBits(1, 14, false);
        SystemStatus->DlpcInitErr = DLPC_COMMON_GetBits(1, 15, false);
        SystemStatus->LampHwErr = DLPC_COMMON_GetBits(1, 16, false);
        SystemStatus->LampPprftout = DLPC_COMMON_GetBits(1, 17, false);
        SystemStatus->NoFreqBinErr = DLPC_COMMON_GetBits(1, 19, false);
        SystemStatus->Dlpa3000CommErr = DLPC_COMMON_GetBits(1, 20, false);
        SystemStatus->UmcRefreshBwUnderflowErr = DLPC_COMMON_GetBits(1, 21, false);
        SystemStatus->DmdInitErr = DLPC_COMMON_GetBits(1, 22, false);
        SystemStatus->DmdPwrDownErr = DLPC_COMMON_GetBits(1, 23, false);
        SystemStatus->SrcdefNotpresent = DLPC_COMMON_GetBits(1, 24, false);
        SystemStatus->SeqbinNotpresent = DLPC_COMMON_GetBits(1, 25, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
        SystemStatus->EepromInitFail = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteGeneralDelayCommand(uint32_t DelayInMilliseconds)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x8);
    DLPC_COMMON_PackBytes((uint8_t*)&DelayInMilliseconds, 4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC654X_WriteEepromInvalidate(DLPC654X_EepromInvalidate_s *EepromInvalidate)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xA);
    DLPC_COMMON_SetBits((int32_t)EepromInvalidate->InvalidateSettings, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)EepromInvalidate->InvalidateColorwheelLampData, 1, 0);
    DLPC_COMMON_SetBits((int32_t)EepromInvalidate->InvalidateSsiCalibrationData, 1, 1);
    DLPC_COMMON_SetBits((int32_t)EepromInvalidate->InvalidateAdcCalibrationData, 1, 2);
    DLPC_COMMON_SetBits((int32_t)EepromInvalidate->InvalidateWpcSensorCalibrationData, 1, 3);
    DLPC_COMMON_SetBits((int32_t)EepromInvalidate->InvalidateWpcBrightnessTableData, 1, 4);
    DLPC_COMMON_SetBits((int32_t)EepromInvalidate->InvalidateXprCalibrationData, 1, 5);
    DLPC_COMMON_SetBits((int32_t)EepromInvalidate->InvalidateXprWaveformCalibrationData, 1, 6);
    DLPC_COMMON_SetBits((int32_t)EepromInvalidate->InvalidateSurfaceCorrectionData, 1, 7);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC654X_WriteSplashCapture()
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadSplashCaptureStatus(DLPC654X_SplashCaptureStateT_e *CaptureState, uint8_t *CaptureCompletionPercentage)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *CaptureState = (DLPC654X_SplashCaptureStateT_e)*(DLPC_COMMON_UnpackBytes(1));
        *CaptureCompletionPercentage = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteTerminateSplashCapture()
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xD);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC654X_WriteInitializeOnTheFlyLoadSplashImage(uint8_t* SplashHeader, uint16_t Width, uint16_t Height)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xA9);
    DLPC_COMMON_PackBytes((uint8_t*)SplashHeader, 20);      // SplashHeader
    DLPC_COMMON_PackBytes((uint8_t*)&Width, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&Height, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC654X_WriteLoadSplashImageOnTheFly(uint16_t ImageDataLength, uint8_t* ImageData)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xAA);
    DLPC_COMMON_PackBytes((uint8_t*)ImageData, ImageDataLength);     // ImageData

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC654X_WriteEnableThreeD(bool Enable3D)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB1);
    DLPC_COMMON_SetBits((int32_t)Enable3D, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadEnableThreeD(bool *Enable3D)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Enable3D = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteLeftRightSignalSource(bool UseLeftRightSignalOnGpio)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB2);
    DLPC_COMMON_SetBits((int32_t)UseLeftRightSignalOnGpio, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadLeftRightSignalSource(bool *UseLeftRightSignalOnGpio)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB2);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *UseLeftRightSignalOnGpio = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteLeftRightSignalPolarity(bool Invert)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB3);
    DLPC_COMMON_SetBits((int32_t)Invert, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadLeftRightSignalPolarity(bool *Invert)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB3);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Invert = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteXpr4WayOrientation(uint8_t XprOrientationNumber)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB4);
    DLPC_COMMON_PackBytes((uint8_t*)&XprOrientationNumber, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXpr4WayOrientation(uint8_t *XprOrientationNumber)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB4);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *XprOrientationNumber = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteXprActuatorWaveformControlParameter(DLPC654X_Xpr4WayCommand_e XprCommand, uint8_t AwcChannel, uint32_t Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackBytes((uint8_t*)&XprCommand, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&AwcChannel, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&Data, 4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprActuatorWaveformControlParameter(DLPC654X_Xpr4WayCommand_e XprCommand, uint8_t AwcChannel, uint32_t *Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackBytes((uint8_t*)&XprCommand, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&AwcChannel, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *Data = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}

uint32_t DLPC654X_WriteDbAperturePosition(uint16_t Position)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB7);
    DLPC_COMMON_PackBytes((uint8_t*)&Position, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDbAperturePosition(uint16_t *Position)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB7);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *Position = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteDbApertureMinMax(uint16_t AptMin, uint16_t AptMax)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB8);
    DLPC_COMMON_PackBytes((uint8_t*)&AptMin, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&AptMax, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDbApertureMinMax(uint16_t *AptMin, uint16_t *AptMax)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB8);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *AptMin = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *AptMax = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteDbManualMode(uint8_t Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB9);
    DLPC_COMMON_PackBytes((uint8_t*)&Enable, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDbManualMode(uint8_t *Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB9);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Enable = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteDbBorderConfiguration(uint16_t Top, uint16_t Bottom, uint16_t Left, uint16_t Right)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xBB);
    DLPC_COMMON_PackBytes((uint8_t*)&Top, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&Bottom, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&Left, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&Right, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDbBorderConfiguration(uint16_t *Top, uint16_t *Bottom, uint16_t *Left, uint16_t *Right)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xBB);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(8);
    if (Status == 0)
    {
        *Top = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *Bottom = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *Left = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *Right = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteDbBorderWeight(DLPC654X_DbPixelWeightT_e BorderWeight)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xBC);
    DLPC_COMMON_PackBytes((uint8_t*)&BorderWeight, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDbBorderWeight(DLPC654X_DbPixelWeightT_e *BorderWeight)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xBC);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *BorderWeight = (DLPC654X_DbPixelWeightT_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteDbClipPixels(uint16_t ClipPixels)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xBD);
    DLPC_COMMON_PackBytes((uint8_t*)&ClipPixels, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDbClipPixels(uint16_t *ClipPixels)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xBD);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *ClipPixels = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteDbGain(double DbGain)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xBE);
    DLPC_COMMON_PackFloat(DbGain, 2, 4096);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDbGain(double *DbGain)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xBE);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *DbGain = DLPC_COMMON_UnpackFloat(2, 4096, false);
    }
    return Status;
}

uint32_t DLPC654X_WriteDbNumSteps(uint16_t Steps)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xBF);
    DLPC_COMMON_PackBytes((uint8_t*)&Steps, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDbNumSteps(uint16_t *Steps)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xBF);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *Steps = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteDbApertureSpeed(uint16_t Speed)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC0);
    DLPC_COMMON_PackBytes((uint8_t*)&Speed, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDbApertureSpeed(uint16_t *Speed)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC0);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *Speed = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteDbStrength(uint8_t Strength)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC1);
    DLPC_COMMON_PackBytes((uint8_t*)&Strength, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDbStrength(uint8_t *Strength)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Strength = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadDbHistogram(uint8_t *HistPtr)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC2);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(136);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < 136; s_Index++)
        {
            HistPtr[s_Index] = *(DLPC_COMMON_UnpackBytes(1));
        }
    }
    return Status;
}
 
uint32_t DLPC654X_ReadDbCurrentApertPos(uint32_t *Position)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC3);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *Position = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadCurrentLedColorPoint(double *ChromaticX, double *ChromaticY, uint32_t *LuminenceY)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC4);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(8);
    if (Status == 0)
    {
        *ChromaticX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        *ChromaticY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        *LuminenceY = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}

uint32_t DLPC654X_WriteWpcOptimalDutyCycle()
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC5);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadWpcOptimalDutyCycle(DLPC654X_WpcOptimalDutyCycle_s *WpcOptimalDutyCycle)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xC5);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(12);
    if (Status == 0)
    {
        WpcOptimalDutyCycle->RedIdealDutyCycle = DLPC_COMMON_UnpackFloat(2, 256, false);
        WpcOptimalDutyCycle->GreenIdealDutyCycle = DLPC_COMMON_UnpackFloat(2, 256, false);
        WpcOptimalDutyCycle->BlueIdealDutyCycle = DLPC_COMMON_UnpackFloat(2, 256, false);
        WpcOptimalDutyCycle->RedOptimalDutyCycle = DLPC_COMMON_UnpackFloat(2, 256, false);
        WpcOptimalDutyCycle->GreenOptimalDutyCycle = DLPC_COMMON_UnpackFloat(2, 256, false);
        WpcOptimalDutyCycle->BlueOptimalDutyCycle = DLPC_COMMON_UnpackFloat(2, 256, false);
    }
    return Status;
}
 
uint32_t DLPC654X_ReadWpcSensorOutput(uint32_t *Red, uint32_t *Green, uint32_t *Blue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xCD);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(12);
    if (Status == 0)
    {
        *Red = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        *Green = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        *Blue = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}

uint32_t DLPC654X_WriteMaximumSsiDriveLevel(DLPC654X_SsiDrvColorT_e Color, uint16_t DriveLevel)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xCE);
    DLPC_COMMON_PackBytes((uint8_t*)&Color, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&DriveLevel, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadMaximumSsiDriveLevel(DLPC654X_SsiDrvColorT_e Color, uint16_t *DriveLevel)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xCE);
    DLPC_COMMON_PackBytes((uint8_t*)&Color, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *DriveLevel = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteDebugMessageMask(DLPC654X_DebugMessageMask_s *DebugMessageMask)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE0);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Comm, 1, 11);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->ThreeD, 1, 13);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->MessageService, 1, 14);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->I2C, 1, 15);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->ClosedCaptioning, 1, 17);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->DdcCi, 1, 18);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Gui, 1, 19);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Environment, 1, 20);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Illumination, 1, 21);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->System, 1, 22);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Eeprom, 1, 23);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Datapath, 1, 24);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Autolock, 1, 25);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->ProjectorCtl, 1, 26);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Peripheral, 1, 27);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Ir, 1, 28);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Usb, 1, 29);
    DLPC_COMMON_SetBits((int32_t)DebugMessageMask->Mailbox, 1, 30);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDebugMessageMask(DLPC654X_DebugMessageMask_s *DebugMessageMask)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE0);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        DebugMessageMask->Comm = DLPC_COMMON_GetBits(1, 11, false);
        DebugMessageMask->ThreeD = DLPC_COMMON_GetBits(1, 13, false);
        DebugMessageMask->MessageService = DLPC_COMMON_GetBits(1, 14, false);
        DebugMessageMask->I2C = DLPC_COMMON_GetBits(1, 15, false);
        DebugMessageMask->ClosedCaptioning = DLPC_COMMON_GetBits(1, 17, false);
        DebugMessageMask->DdcCi = DLPC_COMMON_GetBits(1, 18, false);
        DebugMessageMask->Gui = DLPC_COMMON_GetBits(1, 19, false);
        DebugMessageMask->Environment = DLPC_COMMON_GetBits(1, 20, false);
        DebugMessageMask->Illumination = DLPC_COMMON_GetBits(1, 21, false);
        DebugMessageMask->System = DLPC_COMMON_GetBits(1, 22, false);
        DebugMessageMask->Eeprom = DLPC_COMMON_GetBits(1, 23, false);
        DebugMessageMask->Datapath = DLPC_COMMON_GetBits(1, 24, false);
        DebugMessageMask->Autolock = DLPC_COMMON_GetBits(1, 25, false);
        DebugMessageMask->ProjectorCtl = DLPC_COMMON_GetBits(1, 26, false);
        DebugMessageMask->Peripheral = DLPC_COMMON_GetBits(1, 27, false);
        DebugMessageMask->Ir = DLPC_COMMON_GetBits(1, 28, false);
        DebugMessageMask->Usb = DLPC_COMMON_GetBits(1, 29, false);
        DebugMessageMask->Mailbox = DLPC_COMMON_GetBits(1, 30, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteEnableUsbDebugLog(uint8_t Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE1);
    DLPC_COMMON_PackBytes((uint8_t*)&Enable, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC654X_WriteEepromMemory(uint16_t Index, uint16_t Size, uint32_t Pwd, uint8_t* Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE2);
    DLPC_COMMON_PackBytes((uint8_t*)&Index, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&Size, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&Pwd, 4);
    DLPC_COMMON_PackBytes((uint8_t*)Data, Size);      // Data

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadEepromMemory(uint16_t Index, uint16_t Size, uint8_t *Data)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE2);
    DLPC_COMMON_PackBytes((uint8_t*)&Index, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&Size, 2);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(Size);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < Size; s_Index++)
        {
            Data[s_Index] = *(DLPC_COMMON_UnpackBytes(1));
        }
    }
    return Status;
}

uint32_t DLPC654X_WriteDlpa3005Register(uint8_t RegisterAddress, uint8_t RegisterValue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE3);
    DLPC_COMMON_PackBytes((uint8_t*)&RegisterAddress, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&RegisterValue, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDlpa3005Register(uint8_t RegisterAddress, uint8_t *RegisterValue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE3);
    DLPC_COMMON_PackBytes((uint8_t*)&RegisterAddress, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *RegisterValue = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteTiActuatorInterfaceDebug(uint8_t TiActQueryType, uint16_t TiActAddr, uint16_t TiActNumData)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE4);
    DLPC_COMMON_PackBytes((uint8_t*)&TiActQueryType, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&TiActAddr, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&TiActNumData, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadTiActuatorInterfaceDebug(uint8_t *ActuatorData)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE4);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(32);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < 32; s_Index++)
        {
            ActuatorData[s_Index] = *(DLPC_COMMON_UnpackBytes(1));
        }
    }
    return Status;
}

uint32_t DLPC654X_WriteDmdPower(bool Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE8);
    DLPC_COMMON_SetBits((int32_t)Enable, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDmdPower(bool *Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE8);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Enable = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteDmdPark(bool Park)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE9);
    DLPC_COMMON_SetBits((int32_t)Park, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDmdPark(bool *Park)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xE9);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Park = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteDmdTrueGlobalReset(bool GlobalResetEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xEB);
    DLPC_COMMON_SetBits((int32_t)GlobalResetEnable, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDmdTrueGlobalReset(bool *GlobalResetEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xEB);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *GlobalResetEnable = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}
 
uint32_t DLPC654X_ReadIntStack(uint32_t *Ssize, uint32_t *Sused, uint32_t *Sfree)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xF0);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(12);
    if (Status == 0)
    {
        *Ssize = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        *Sused = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        *Sfree = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}

uint32_t DLPC654X_WritePrintAllTaskInformation()
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xF1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadResource(DLPC654X_Resource_s *Resource)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xF2);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(12);
    if (Status == 0)
    {
        Resource->HighCountTasks = *(DLPC_COMMON_UnpackBytes(1));
        Resource->HighCountEvts = *(DLPC_COMMON_UnpackBytes(1));
        Resource->HighCountGrpEvts = *(DLPC_COMMON_UnpackBytes(1));
        Resource->HighCountMbxs = *(DLPC_COMMON_UnpackBytes(1));
        Resource->HighCountMemPools = *(DLPC_COMMON_UnpackBytes(1));
        Resource->HighCountSems = *(DLPC_COMMON_UnpackBytes(1));
        Resource->CurrCountTasks = *(DLPC_COMMON_UnpackBytes(1));
        Resource->CurrCountEvts = *(DLPC_COMMON_UnpackBytes(1));
        Resource->CurrCountGrpEvts = *(DLPC_COMMON_UnpackBytes(1));
        Resource->CurrCountMbxs = *(DLPC_COMMON_UnpackBytes(1));
        Resource->CurrCountMemPools = *(DLPC_COMMON_UnpackBytes(1));
        Resource->CurrCountSems = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadVx1HwStatus(DLPC654X_Vx1HwStatus_s *Vx1HwStatus)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x3A);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(30);
    if (Status == 0)
    {
        Vx1HwStatus->Appl = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->Alpf = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->TpplLargest = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->Tlpf = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->TpplSmallest = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->Vfp = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->Vbp = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->Vsw = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->Hfp = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->Hbp = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->Hsw = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->Hs2Vs = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->Vs2Hs = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        Vx1HwStatus->HSyncPolarityIsPositive = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        Vx1HwStatus->VSyncPolarityIsPositive = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        Vx1HwStatus->FreqCaptured = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WritePower()
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x10);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadPower(DLPC654X_PowerstateEnum_e *PowerState)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x10);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *PowerState = (DLPC654X_PowerstateEnum_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteDisplay(DLPC654X_CmdProjectionModes_e Source)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x11);
    DLPC_COMMON_PackBytes((uint8_t*)&Source, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDisplay(DLPC654X_CmdProjectionModes_e *Source)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x11);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Source = (DLPC654X_CmdProjectionModes_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteTpgBorder(uint8_t Width, uint16_t BorderColorRed, uint16_t BorderColorGreen, uint16_t BorderColorBlue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x15);
    DLPC_COMMON_PackBytes((uint8_t*)&Width, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&BorderColorRed, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&BorderColorGreen, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&BorderColorBlue, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadTpgBorder(uint8_t *Width, uint16_t *BorderColorRed, uint16_t *BorderColorGreen, uint16_t *BorderColorBlue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x15);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(7);
    if (Status == 0)
    {
        *Width = *(DLPC_COMMON_UnpackBytes(1));
        *BorderColorRed = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *BorderColorGreen = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *BorderColorBlue = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteTpgResolution(uint16_t HorizontalResolution, uint16_t VerticalResolution)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x16);
    DLPC_COMMON_PackBytes((uint8_t*)&HorizontalResolution, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&VerticalResolution, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadTpgResolution(uint16_t *HorizontalResolution, uint16_t *VerticalResolution)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x16);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *HorizontalResolution = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *VerticalResolution = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteTpgFrameRate(uint8_t FrameRate)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x17);
    DLPC_COMMON_PackBytes((uint8_t*)&FrameRate, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadTpgFrameRate(uint8_t *FrameRate)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x17);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *FrameRate = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteSfgColor(uint16_t Red, uint16_t Green, uint16_t Blue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x18);
    DLPC_COMMON_PackBytes((uint8_t*)&Red, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&Green, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&Blue, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadSfgColor(uint16_t *Red, uint16_t *Green, uint16_t *Blue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x18);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        *Red = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *Green = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *Blue = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteCurtainColor(DLPC654X_DispBackgroundColorT_e Color)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1A);
    DLPC_COMMON_PackBytes((uint8_t*)&Color, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadCurtainColor(DLPC654X_DispBackgroundColorT_e *Color)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1A);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Color = (DLPC654X_DispBackgroundColorT_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteSplashLoadImage(uint8_t Index)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1B);
    DLPC_COMMON_PackBytes((uint8_t*)&Index, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadSplashLoadImage(uint8_t *Index)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1B);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Index = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteEnableImageFlip(bool VertFlip, bool HorzFlip)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1C);
    DLPC_COMMON_SetBits((int32_t)VertFlip, 1, 0);
    DLPC_COMMON_SetBits((int32_t)HorzFlip, 1, 1);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadEnableImageFlip(bool *VertFlip, bool *HorzFlip)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1C);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *VertFlip = DLPC_COMMON_GetBits(1, 0, false);
        *HorzFlip = DLPC_COMMON_GetBits(1, 1, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteEnableFreeze(bool Freeze)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1D);
    DLPC_COMMON_SetBits((int32_t)Freeze, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadEnableFreeze(bool *Freeze)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1D);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Freeze = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteKeystoneAngles(double Pitch, double Yaw, double Roll)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1E);
    DLPC_COMMON_PackFloat(Pitch, 2, 256);
    DLPC_COMMON_PackFloat(Yaw, 2, 256);
    DLPC_COMMON_PackFloat(Roll, 2, 256);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadKeystoneAngles(double *Pitch, double *Yaw, double *Roll)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1E);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        *Pitch = DLPC_COMMON_UnpackFloat(2, 256, true);
        *Yaw = DLPC_COMMON_UnpackFloat(2, 256, true);
        *Roll = DLPC_COMMON_UnpackFloat(2, 256, true);
    }
    return Status;
}

uint32_t DLPC654X_WriteKeystoneConfigOverride(double ThrowRatio, double VerticalOffset)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1F);
    DLPC_COMMON_PackFloat(ThrowRatio, 2, 256);
    DLPC_COMMON_PackFloat(VerticalOffset, 2, 256);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadKeystoneConfigOverride(double *ThrowRatio, double *VerticalOffset)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x1F);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *ThrowRatio = DLPC_COMMON_UnpackFloat(2, 256, false);
        *VerticalOffset = DLPC_COMMON_UnpackFloat(2, 256, true);
    }
    return Status;
}

uint32_t DLPC654X_WriteDisplayImageSize(DLPC654X_DisplayImageSize_s *DisplayImageSize)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x21);
    DLPC_COMMON_PackBytes((uint8_t*)&DisplayImageSize->CroppedAreaFirstPixel, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&DisplayImageSize->CroppedAreaFirstLine, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&DisplayImageSize->CroppedAreaPixelsPerLine, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&DisplayImageSize->CroppedAreaLinesPerFrame, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&DisplayImageSize->DisplayAreaFirstPixel, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&DisplayImageSize->DisplayAreaFirstLine, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&DisplayImageSize->DisplayAreaPixelsPerLine, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&DisplayImageSize->DisplayAreaLinesPerFrame, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDisplayImageSize(DLPC654X_DisplayImageSize_s *DisplayImageSize)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x21);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(16);
    if (Status == 0)
    {
        DisplayImageSize->CroppedAreaFirstPixel = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        DisplayImageSize->CroppedAreaFirstLine = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        DisplayImageSize->CroppedAreaPixelsPerLine = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        DisplayImageSize->CroppedAreaLinesPerFrame = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        DisplayImageSize->DisplayAreaFirstPixel = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        DisplayImageSize->DisplayAreaFirstLine = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        DisplayImageSize->DisplayAreaPixelsPerLine = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        DisplayImageSize->DisplayAreaLinesPerFrame = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteSourceConfiguration(DLPC654X_SourceConfiguration_s *SourceConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x22);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->VSyncConfiguration, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->HSyncConfiguration, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->TopFieldConfiguration, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->DownSampleConfiguration, 1);
    DLPC_COMMON_SetBits((int32_t)SourceConfiguration->IsThreeD, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)SourceConfiguration->IsClockPolarityPositive, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->PixelFormat, 1);
    DLPC_COMMON_SetBits((int32_t)SourceConfiguration->IsExternalDataEnable, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)SourceConfiguration->IsInterlaced, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)SourceConfiguration->IsOffsetBinary, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)SourceConfiguration->IsTopFieldInvertedAtScaler, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->TotalAreaPixelsPerLine, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->TotalAreaLinesPerFrame, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ActiveAreaFirstPixel, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ActiveAreaFirstLine, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ActiveAreaPixelsPerLine, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ActiveAreaLinesPerFrame, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->BottomFieldFirstLine, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->PixelClockFreqInKiloHertz, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ColorSpaceConvCoeffs0, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ColorSpaceConvCoeffs1, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ColorSpaceConvCoeffs2, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ColorSpaceConvCoeffs3, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ColorSpaceConvCoeffs4, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ColorSpaceConvCoeffs5, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ColorSpaceConvCoeffs6, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ColorSpaceConvCoeffs7, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->ColorSpaceConvCoeffs8, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->OffsetRed, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->OffsetGreen, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->OffsetBlue, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->IsVideo, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&SourceConfiguration->IsHighDefinitionVideo, 1);
    DLPC_COMMON_PackFloat(SourceConfiguration->FrameRate, 4, 65536);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadSourceConfiguration(DLPC654X_SourceConfiguration_s *SourceConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x22);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(59);
    if (Status == 0)
    {
        SourceConfiguration->VSyncConfiguration = (DLPC654X_SrcSyncConfigT_e)*(DLPC_COMMON_UnpackBytes(1));
        SourceConfiguration->HSyncConfiguration = (DLPC654X_SrcSyncConfigT_e)*(DLPC_COMMON_UnpackBytes(1));
        SourceConfiguration->TopFieldConfiguration = (DLPC654X_SrcSyncConfigT_e)*(DLPC_COMMON_UnpackBytes(1));
        SourceConfiguration->DownSampleConfiguration = (DLPC654X_SrcDownSampleConfigT_e)*(DLPC_COMMON_UnpackBytes(1));
        SourceConfiguration->IsThreeD = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        SourceConfiguration->IsClockPolarityPositive = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        SourceConfiguration->PixelFormat = (DLPC654X_SrcPixelFormatT_e)*(DLPC_COMMON_UnpackBytes(1));
        SourceConfiguration->IsExternalDataEnable = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        SourceConfiguration->IsInterlaced = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        SourceConfiguration->IsOffsetBinary = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        SourceConfiguration->IsTopFieldInvertedAtScaler = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        SourceConfiguration->TotalAreaPixelsPerLine = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->TotalAreaLinesPerFrame = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ActiveAreaFirstPixel = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ActiveAreaFirstLine = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ActiveAreaPixelsPerLine = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ActiveAreaLinesPerFrame = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->BottomFieldFirstLine = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->PixelClockFreqInKiloHertz = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        SourceConfiguration->ColorSpaceConvCoeffs0 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ColorSpaceConvCoeffs1 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ColorSpaceConvCoeffs2 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ColorSpaceConvCoeffs3 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ColorSpaceConvCoeffs4 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ColorSpaceConvCoeffs5 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ColorSpaceConvCoeffs6 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ColorSpaceConvCoeffs7 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->ColorSpaceConvCoeffs8 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->OffsetRed = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->OffsetGreen = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->OffsetBlue = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SourceConfiguration->IsVideo = *(DLPC_COMMON_UnpackBytes(1));
        SourceConfiguration->IsHighDefinitionVideo = *(DLPC_COMMON_UnpackBytes(1));
        SourceConfiguration->FrameRate = DLPC_COMMON_UnpackFloat(4, 65536, false);
    }
    return Status;
}

uint32_t DLPC654X_WriteAutolockSetup(uint8_t Action)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x24);
    DLPC_COMMON_PackBytes((uint8_t*)&Action, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDatapathScanStatus(DLPC654X_DpScanStatus_e *ScanStatus)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x25);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *ScanStatus = (DLPC654X_DpScanStatus_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteVboConfiguration(DLPC654X_SrcVboDataMapModeT_e DataMapMode, DLPC654X_SrcVboByteModeT_e ByteMode, uint8_t NumberOfLanes, bool EnablePixelRepeat)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x30);
    DLPC_COMMON_PackBytes((uint8_t*)&DataMapMode, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&ByteMode, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&NumberOfLanes, 1);
    DLPC_COMMON_SetBits((int32_t)EnablePixelRepeat, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadVboConfiguration(DLPC654X_SrcVboDataMapModeT_e *DataMapMode, DLPC654X_SrcVboByteModeT_e *ByteMode, uint8_t *NumberOfLanes, bool *EnablePixelRepeat)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x30);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *DataMapMode = (DLPC654X_SrcVboDataMapModeT_e)*(DLPC_COMMON_UnpackBytes(1));
        *ByteMode = (DLPC654X_SrcVboByteModeT_e)*(DLPC_COMMON_UnpackBytes(1));
        *NumberOfLanes = *(DLPC_COMMON_UnpackBytes(1));
        *EnablePixelRepeat = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteFpdConfiguration(DLPC654X_FpdConfiguration_s *FpdConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x31);
    DLPC_COMMON_PackBytes((uint8_t*)&FpdConfiguration->FpdMode, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&FpdConfiguration->DataInterfaceMode, 1);
    DLPC_COMMON_SetBits((int32_t)FpdConfiguration->Enable3DRef, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)FpdConfiguration->EnableField, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)FpdConfiguration->EnablePixelRepeat, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadFpdConfiguration(DLPC654X_FpdConfiguration_s *FpdConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x31);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(5);
    if (Status == 0)
    {
        FpdConfiguration->FpdMode = (DLPC654X_SrcFpdDataMapModeT_e)*(DLPC_COMMON_UnpackBytes(1));
        FpdConfiguration->DataInterfaceMode = (DLPC654X_SrcFpdDataInterfaceModeT_e)*(DLPC_COMMON_UnpackBytes(1));
        FpdConfiguration->Enable3DRef = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        FpdConfiguration->EnableField = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        FpdConfiguration->EnablePixelRepeat = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteManualWarpTable(uint16_t WarpTableIndex, uint16_t WarpPointsLength, uint16_t WarpPoints[])
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x34);
    DLPC_COMMON_PackBytes((uint8_t*)&WarpTableIndex, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadManualWarpTable(uint16_t WarpTableIndex, uint16_t NumEntries, uint16_t *WarpPoints[])
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x34);
    DLPC_COMMON_PackBytes((uint8_t*)&WarpTableIndex, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&NumEntries, 2);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(0xFFFF);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < (uint32_t)(DLPC_COMMON_GetBytesRead() / 2); s_Index++)
        {
            WarpPoints[s_Index] = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        }
    }
    return Status;
}

uint32_t DLPC654X_WriteManualWarpControlPoints(uint16_t HorizontalCtrlPointsLength, uint16_t VerticalCtrlPointsLength, DLPC654X_ManualWarpControlPoints_s *ManualWarpControlPoints)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x35);
    DLPC_COMMON_PackBytes((uint8_t*)&ManualWarpControlPoints->ControlPointsDefinedByArray, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&ManualWarpControlPoints->InputWidth, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&ManualWarpControlPoints->InputHeight, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&ManualWarpControlPoints->WarpColumns, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&ManualWarpControlPoints->WarpRows, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadManualWarpControlPoints(uint16_t *HorizontalCtrlPoints, uint16_t *VerticalCtrlPoints)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x35);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(0xFFFF);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < (uint32_t)(DLPC_COMMON_GetBytesRead() / 2); s_Index++)
        {
            HorizontalCtrlPoints[s_Index] = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        }
        for (s_Index = 0; s_Index < (uint32_t)(DLPC_COMMON_GetBytesRead() / 2); s_Index++)
        {
            VerticalCtrlPoints[s_Index] = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        }
    }
    return Status;
}

uint32_t DLPC654X_WriteApplyManualWarping(uint8_t Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x36);
    DLPC_COMMON_PackBytes((uint8_t*)&Enable, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadManualWarpingEnabled(uint8_t *Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x37);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Enable = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteConfigureSmoothWarp(uint16_t WarpPointsLength, uint16_t WarpPoints[])
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x38);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC654X_WriteManualWarpTableUpdateMode(uint8_t Mode)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x39);
    DLPC_COMMON_PackBytes((uint8_t*)&Mode, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadManualWarpTableUpdateMode(uint8_t *Mode)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x39);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Mode = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteIlluminationEnable(uint8_t Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x80);
    DLPC_COMMON_PackBytes((uint8_t*)&Enable, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadIlluminationEnable(uint8_t *Enable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x80);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Enable = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteDlpa3005IlluminationCurrent(uint16_t DriveLevelRed, uint16_t DriveLevelGreen, uint16_t DriveLevelBlue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x84);
    DLPC_COMMON_PackBytes((uint8_t*)&DriveLevelRed, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&DriveLevelGreen, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&DriveLevelBlue, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadDlpa3005IlluminationCurrent(uint16_t *DriveLevelRed, uint16_t *DriveLevelGreen, uint16_t *DriveLevelBlue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x84);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        *DriveLevelRed = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *DriveLevelGreen = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *DriveLevelBlue = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteSsiDriveLevels(uint8_t PwmGroup, DLPC654X_SsiDriveLevels_s *SsiDriveLevels)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x8C);
    DLPC_COMMON_PackBytes((uint8_t*)&PwmGroup, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&SsiDriveLevels->DriveLevelRed, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SsiDriveLevels->DriveLevelGreen, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SsiDriveLevels->DriveLevelBlue, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SsiDriveLevels->DriveLevelC1, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SsiDriveLevels->DriveLevelC2, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&SsiDriveLevels->DriveLevelSense, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadSsiDriveLevels(uint8_t PwmGroup, DLPC654X_SsiDriveLevels_s *SsiDriveLevels)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x8C);
    DLPC_COMMON_PackBytes((uint8_t*)&PwmGroup, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(12);
    if (Status == 0)
    {
        SsiDriveLevels->DriveLevelRed = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SsiDriveLevels->DriveLevelGreen = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SsiDriveLevels->DriveLevelBlue = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SsiDriveLevels->DriveLevelC1 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SsiDriveLevels->DriveLevelC2 = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        SsiDriveLevels->DriveLevelSense = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteImageAlgorithmEnable(DLPC654X_ImageAlgorithmEnable_s *ImageAlgorithmEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x40);
    DLPC_COMMON_SetBits((int32_t)ImageAlgorithmEnable->ChromaTransientImprovementEnableBit, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)ImageAlgorithmEnable->GammaCorrectionEnableBit, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)ImageAlgorithmEnable->ColorCoordinateAdjustmentEnableBit, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)ImageAlgorithmEnable->BrilliantColorEnableBit, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)ImageAlgorithmEnable->WhitePointCorrectionEnableBit, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)ImageAlgorithmEnable->DynamicBlackEnableBit, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)ImageAlgorithmEnable->HdrEnableBit, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageAlgorithmEnable(DLPC654X_ImageAlgorithmEnable_s *ImageAlgorithmEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x40);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(7);
    if (Status == 0)
    {
        ImageAlgorithmEnable->ChromaTransientImprovementEnableBit = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        ImageAlgorithmEnable->GammaCorrectionEnableBit = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        ImageAlgorithmEnable->ColorCoordinateAdjustmentEnableBit = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        ImageAlgorithmEnable->BrilliantColorEnableBit = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        ImageAlgorithmEnable->WhitePointCorrectionEnableBit = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        ImageAlgorithmEnable->DynamicBlackEnableBit = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        ImageAlgorithmEnable->HdrEnableBit = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteImageBrightness(double BrightnessAdjustment)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x41);
    DLPC_COMMON_PackFloat(BrightnessAdjustment, 2, 4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageBrightness(double *BrightnessAdjustment)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x41);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *BrightnessAdjustment = DLPC_COMMON_UnpackFloat(2, 4, true);
    }
    return Status;
}

uint32_t DLPC654X_WriteImageContrast(uint16_t Contrast)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x42);
    DLPC_COMMON_PackBytes((uint8_t*)&Contrast, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageContrast(uint16_t *Contrast)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x42);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *Contrast = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteImageHueAndColorControl(int8_t HueAdjustmentAngle, uint16_t ColorControlGain)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x43);
    DLPC_COMMON_PackBytes((uint8_t*)&HueAdjustmentAngle, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&ColorControlGain, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageHueAndColorControl(int8_t *HueAdjustmentAngle, uint16_t *ColorControlGain)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x43);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(3);
    if (Status == 0)
    {
        *HueAdjustmentAngle = *(DLPC_COMMON_UnpackBytes(1));
        *ColorControlGain = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteImageSharpness(uint8_t Sharpness)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x44);
    DLPC_COMMON_PackBytes((uint8_t*)&Sharpness, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageSharpness(uint8_t *Sharpness)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x44);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Sharpness = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteImageRgbOffset(double RedOffset, double GreenOffset, double BlueOffset)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x45);
    DLPC_COMMON_PackFloat(RedOffset, 2, 4);
    DLPC_COMMON_PackFloat(GreenOffset, 2, 4);
    DLPC_COMMON_PackFloat(BlueOffset, 2, 4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageRgbOffset(double *RedOffset, double *GreenOffset, double *BlueOffset)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x45);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        *RedOffset = DLPC_COMMON_UnpackFloat(2, 4, true);
        *GreenOffset = DLPC_COMMON_UnpackFloat(2, 4, true);
        *BlueOffset = DLPC_COMMON_UnpackFloat(2, 4, true);
    }
    return Status;
}

uint32_t DLPC654X_WriteImageRgbGain(uint16_t RedGain, uint16_t GreenGain, uint16_t BlueGain)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x46);
    DLPC_COMMON_PackBytes((uint8_t*)&RedGain, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&GreenGain, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&BlueGain, 2);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageRgbGain(uint16_t *RedGain, uint16_t *GreenGain, uint16_t *BlueGain)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x46);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        *RedGain = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *GreenGain = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        *BlueGain = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteCscTable(DLPC654X_SrcCscTablesT_e Index)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x47);
    DLPC_COMMON_PackBytes((uint8_t*)&Index, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadCscTable(DLPC654X_SrcCscTablesT_e *Index)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x47);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *Index = (DLPC654X_SrcCscTablesT_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteImageCcaCoordinates(DLPC654X_ImageCcaCoordinates_s *ImageCcaCoordinates)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x48);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsRedX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsRedY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsRedLum, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsGreenX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsGreenY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsGreenLum, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsBlueX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsBlueY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsBlueLum, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsWhiteX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsWhiteY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsWhiteLum, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsC1X, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsC1Y, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsC1Lum, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsC2X, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsC2Y, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsC2Lum, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsDraAX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsDraAY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsDraALum, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsDraBX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsDraBY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsDraBLum, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsDraCX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsDraCY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->OrigCoordsDraCLum, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsRedX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsRedY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsRedGain, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsGreenX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsGreenY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsGreenGain, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsBlueX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsBlueY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsBlueGain, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsCyanX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsCyanY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsCyanGain, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsMagentaX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsMagentaY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsMagentaGain, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsYellowX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsYellowY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsYellowGain, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsWhiteX, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsWhiteY, 2, 32768);
    DLPC_COMMON_PackFloat(ImageCcaCoordinates->TargetCoordsWhiteGain, 2, 32768);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageCcaCoordinates(DLPC654X_ImageCcaCoordinates_s *ImageCcaCoordinates)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x48);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(96);
    if (Status == 0)
    {
        ImageCcaCoordinates->OrigCoordsRedX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsRedY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsRedLum = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsGreenX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsGreenY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsGreenLum = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsBlueX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsBlueY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsBlueLum = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsWhiteX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsWhiteY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsWhiteLum = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsC1X = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsC1Y = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsC1Lum = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsC2X = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsC2Y = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsC2Lum = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsDraAX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsDraAY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsDraALum = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsDraBX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsDraBY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsDraBLum = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsDraCX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsDraCY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->OrigCoordsDraCLum = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsRedX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsRedY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsRedGain = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsGreenX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsGreenY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsGreenGain = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsBlueX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsBlueY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsBlueGain = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsCyanX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsCyanY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsCyanGain = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsMagentaX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsMagentaY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsMagentaGain = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsYellowX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsYellowY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsYellowGain = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsWhiteX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsWhiteY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        ImageCcaCoordinates->TargetCoordsWhiteGain = DLPC_COMMON_UnpackFloat(2, 32768, false);
    }
    return Status;
}

uint32_t DLPC654X_WriteImageHsg(DLPC654X_ImageHsg_s *ImageHsg)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x49);
    DLPC_COMMON_PackFloat(ImageHsg->HsgRedGain, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgRedSaturation, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgRedHue, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgGreenGain, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgGreenSaturation, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgGreenHue, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgBlueGain, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgBlueSaturation, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgBlueHue, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgCyanGain, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgCyanSaturation, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgCyanHue, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgMagentaGain, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgMagentaSaturation, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgMagentaHue, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgYellowGain, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgYellowSaturation, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgYellowHue, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgWhiteRedGain, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgWhiteGreenGain, 2, 16384);
    DLPC_COMMON_PackFloat(ImageHsg->HsgWhiteBlueGain, 2, 16384);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageHsg(DLPC654X_ImageHsg_s *ImageHsg)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x49);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(42);
    if (Status == 0)
    {
        ImageHsg->HsgRedGain = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgRedSaturation = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgRedHue = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgGreenGain = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgGreenSaturation = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgGreenHue = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgBlueGain = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgBlueSaturation = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgBlueHue = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgCyanGain = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgCyanSaturation = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgCyanHue = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgMagentaGain = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgMagentaSaturation = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgMagentaHue = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgYellowGain = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgYellowSaturation = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgYellowHue = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgWhiteRedGain = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgWhiteGreenGain = DLPC_COMMON_UnpackFloat(2, 16384, true);
        ImageHsg->HsgWhiteBlueGain = DLPC_COMMON_UnpackFloat(2, 16384, true);
    }
    return Status;
}

uint32_t DLPC654X_WriteImageGammaLut(uint8_t GammaLutNumber)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4A);
    DLPC_COMMON_PackBytes((uint8_t*)&GammaLutNumber, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageGammaLut(uint8_t *GammaLutNumber)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4A);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *GammaLutNumber = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteImageGammaCurveShift(int8_t RedShift, int8_t GreenShift, int8_t BlueShift, int8_t AllColorShift)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4B);
    DLPC_COMMON_PackBytes((uint8_t*)&RedShift, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&GreenShift, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&BlueShift, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&AllColorShift, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImageGammaCurveShift(int8_t *RedShift, int8_t *GreenShift, int8_t *BlueShift, int8_t *AllColorShift)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4B);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *RedShift = *(DLPC_COMMON_UnpackBytes(1));
        *GreenShift = *(DLPC_COMMON_UnpackBytes(1));
        *BlueShift = *(DLPC_COMMON_UnpackBytes(1));
        *AllColorShift = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteImgWhitePeakingFactor(uint8_t WhitePeakingVal)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4C);
    DLPC_COMMON_PackBytes((uint8_t*)&WhitePeakingVal, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadImgWhitePeakingFactor(uint8_t *WhitePeakingVal)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4C);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *WhitePeakingVal = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteHdrSourceConfiguration(DLPC654X_HdrSourceConfiguration_s *HdrSourceConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4E);
    DLPC_COMMON_PackBytes((uint8_t*)&HdrSourceConfiguration->TransferFunction, 1);
    DLPC_COMMON_PackFloat(HdrSourceConfiguration->MasterDisplayBlackLevel, 4, 65536);
    DLPC_COMMON_PackFloat(HdrSourceConfiguration->MasterDisplayWhiteLevel, 4, 65536);
    DLPC_COMMON_PackFloat(HdrSourceConfiguration->MasterDisplayColorGamutRedX, 2, 32768);
    DLPC_COMMON_PackFloat(HdrSourceConfiguration->MasterDisplayColorGamutRedY, 2, 32768);
    DLPC_COMMON_PackFloat(HdrSourceConfiguration->MasterDisplayColorGamutGreenX, 2, 32768);
    DLPC_COMMON_PackFloat(HdrSourceConfiguration->MasterDisplayColorGamutGreenY, 2, 32768);
    DLPC_COMMON_PackFloat(HdrSourceConfiguration->MasterDisplayColorGamutBlueX, 2, 32768);
    DLPC_COMMON_PackFloat(HdrSourceConfiguration->MasterDisplayColorGamutBlueY, 2, 32768);
    DLPC_COMMON_PackFloat(HdrSourceConfiguration->MasterDisplayColorGamutWhiteX, 2, 32768);
    DLPC_COMMON_PackFloat(HdrSourceConfiguration->MasterDisplayColorGamutWhiteY, 2, 32768);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadHdrSourceConfiguration(DLPC654X_HdrSourceConfiguration_s *HdrSourceConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4E);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(25);
    if (Status == 0)
    {
        HdrSourceConfiguration->TransferFunction = (DLPC654X_HdrTransferFnT_e)*(DLPC_COMMON_UnpackBytes(1));
        HdrSourceConfiguration->MasterDisplayBlackLevel = DLPC_COMMON_UnpackFloat(4, 65536, false);
        HdrSourceConfiguration->MasterDisplayWhiteLevel = DLPC_COMMON_UnpackFloat(4, 65536, false);
        HdrSourceConfiguration->MasterDisplayColorGamutRedX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        HdrSourceConfiguration->MasterDisplayColorGamutRedY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        HdrSourceConfiguration->MasterDisplayColorGamutGreenX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        HdrSourceConfiguration->MasterDisplayColorGamutGreenY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        HdrSourceConfiguration->MasterDisplayColorGamutBlueX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        HdrSourceConfiguration->MasterDisplayColorGamutBlueY = DLPC_COMMON_UnpackFloat(2, 32768, false);
        HdrSourceConfiguration->MasterDisplayColorGamutWhiteX = DLPC_COMMON_UnpackFloat(2, 32768, false);
        HdrSourceConfiguration->MasterDisplayColorGamutWhiteY = DLPC_COMMON_UnpackFloat(2, 32768, false);
    }
    return Status;
}

uint32_t DLPC654X_WriteHdrStrengthSetting(uint8_t HdrStrength)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4F);
    DLPC_COMMON_PackBytes((uint8_t*)&HdrStrength, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadHdrStrengthSetting(uint8_t *HdrStrength)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x4F);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *HdrStrength = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteSystemBrightnessRangeSetting(double MinBrightness, double MaxBrightness)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x50);
    DLPC_COMMON_PackFloat(MinBrightness, 4, 65536);
    DLPC_COMMON_PackFloat(MaxBrightness, 4, 65536);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadSystemBrightnessRangeSetting(double *MinBrightness, double *MaxBrightness)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x50);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(8);
    if (Status == 0)
    {
        *MinBrightness = DLPC_COMMON_UnpackFloat(4, 65536, false);
        *MaxBrightness = DLPC_COMMON_UnpackFloat(4, 65536, false);
    }
    return Status;
}

uint32_t DLPC654X_WriteImageColorProfile(uint8_t ColorProfile)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x51);
    DLPC_COMMON_PackBytes((uint8_t*)&ColorProfile, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}

uint32_t DLPC654X_WriteGpioPinConfig(uint8_t PinNumber, bool Output, bool LogicVal, bool OpenDrain)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x60);
    DLPC_COMMON_PackBytes((uint8_t*)&PinNumber, 1);
    DLPC_COMMON_SetBits((int32_t)Output, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)LogicVal, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_SetBits((int32_t)OpenDrain, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadGpioPinConfig(uint8_t PinNumber, bool *Output, bool *LogicVal, bool *OpenDrain)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x60);
    DLPC_COMMON_PackBytes((uint8_t*)&PinNumber, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(3);
    if (Status == 0)
    {
        *Output = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        *LogicVal = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        *OpenDrain = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteGpioPin(uint8_t PinNumber, bool LogicVal)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x61);
    DLPC_COMMON_PackBytes((uint8_t*)&PinNumber, 1);
    DLPC_COMMON_SetBits((int32_t)LogicVal, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadGpioPin(uint8_t PinNumber, bool *LogicVal)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x61);
    DLPC_COMMON_PackBytes((uint8_t*)&PinNumber, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *LogicVal = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WriteGenPurpseClockEnable(uint8_t Clk, uint8_t Enabled, uint32_t Freq)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x63);
    DLPC_COMMON_PackBytes((uint8_t*)&Clk, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&Enabled, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&Freq, 4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadGenPurpseClockEnable(uint8_t Clk, uint8_t *IsEnabled)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x63);
    DLPC_COMMON_PackBytes((uint8_t*)&Clk, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *IsEnabled = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}
 
uint32_t DLPC654X_ReadGenPurpseClockFrequency(uint8_t Clk, uint32_t *Freq)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x64);
    DLPC_COMMON_PackBytes((uint8_t*)&Clk, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *Freq = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
    }
    return Status;
}

uint32_t DLPC654X_WritePwmOutputConfiguration(DLPC654X_Pwmport_e Port, uint32_t Frequency, uint8_t DutyCycle, bool OutputEnabled)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x65);
    DLPC_COMMON_PackBytes((uint8_t*)&Port, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&Frequency, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&DutyCycle, 1);
    DLPC_COMMON_SetBits((int32_t)OutputEnabled, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadPwmOutputConfiguration(DLPC654X_Pwmport_e Port, uint32_t *Frequency, uint8_t *DutyCycle, bool *OutputEnabledBit)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x65);
    DLPC_COMMON_PackBytes((uint8_t*)&Port, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(6);
    if (Status == 0)
    {
        *Frequency = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        *DutyCycle = *(DLPC_COMMON_UnpackBytes(1));
        *OutputEnabledBit = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
    }
    return Status;
}

uint32_t DLPC654X_WritePwmInputConfiguration(DLPC654X_PwmIncounter_e Port, uint32_t SampleRate, uint8_t InCounterEnabled)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x66);
    DLPC_COMMON_PackBytes((uint8_t*)&Port, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&SampleRate, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&InCounterEnabled, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadPwmInputConfiguration(DLPC654X_PwmIncounter_e Port, DLPC654X_PwmInputConfiguration_s *PwmInputConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x66);
    DLPC_COMMON_PackBytes((uint8_t*)&Port, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(10);
    if (Status == 0)
    {
        PwmInputConfiguration->SampleRate = *((uint32_t*)DLPC_COMMON_UnpackBytes(4));
        PwmInputConfiguration->InCounterEnabled = *(DLPC_COMMON_UnpackBytes(1));
        PwmInputConfiguration->HighPulseWidth = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        PwmInputConfiguration->LowPulseWidth = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
        PwmInputConfiguration->DutyCycle = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteI2CPassthrough(uint16_t DataBytesLength, DLPC654X_I2CPassthrough_s *I2CPassthrough)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x67);
    DLPC_COMMON_PackBytes((uint8_t*)&I2CPassthrough->Port, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&I2CPassthrough->Is7Bit, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&I2CPassthrough->HasSubaddr, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&I2CPassthrough->ClockRate, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&I2CPassthrough->DeviceAddress, 2);
    DLPC_COMMON_PackBytes((uint8_t*)I2CPassthrough->SubAddr, I2CPassthrough->HasSubaddr);      // SubAddr
    DLPC_COMMON_PackBytes((uint8_t*)I2CPassthrough->DataBytes, DataBytesLength);     // DataBytes

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadI2CPassthrough(DLPC654X_I2CPortT_e Port, uint8_t Is7Bit, uint8_t HasSubaddr, uint32_t ClockRate, uint16_t DeviceAddress, uint16_t ByteCount, uint8_t* SubAddr, uint8_t *DataBytes)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x67);
    DLPC_COMMON_PackBytes((uint8_t*)&Port, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&Is7Bit, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&HasSubaddr, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&ClockRate, 4);
    DLPC_COMMON_PackBytes((uint8_t*)&DeviceAddress, 2);
    DLPC_COMMON_PackBytes((uint8_t*)&ByteCount, 2);
    DLPC_COMMON_PackBytes((uint8_t*)SubAddr, HasSubaddr);      // SubAddr

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(ByteCount);
    if (Status == 0)
    {
        for (s_Index = 0; s_Index < ByteCount; s_Index++)
        {
            DataBytes[s_Index] = *(DLPC_COMMON_UnpackBytes(1));
        }
    }
    return Status;
}
 
uint32_t DLPC654X_ReadDmdTemperature(uint16_t *Temperature)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x69);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(2);
    if (Status == 0)
    {
        *Temperature = *((uint16_t*)DLPC_COMMON_UnpackBytes(2));
    }
    return Status;
}

uint32_t DLPC654X_WriteEepromLockState(uint8_t LockState)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x6C);
    DLPC_COMMON_PackBytes((uint8_t*)&LockState, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadEepromLockState(uint8_t *LockState)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x6C);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(1);
    if (Status == 0)
    {
        *LockState = *(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteUartConfiguration(DLPC654X_UrtPortT_e Port, DLPC654X_UartConfiguration_s *UartConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0x6D);
    DLPC_COMMON_PackBytes((uint8_t*)&Port, 1);
    DLPC_COMMON_SetBits((int32_t)UartConfiguration->Enable, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(1);
    DLPC_COMMON_PackBytes((uint8_t*)&UartConfiguration->BaudRate, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&UartConfiguration->DataBits, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&UartConfiguration->StopBits, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&UartConfiguration->Parity, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&UartConfiguration->FlowControl, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&UartConfiguration->RxTrigLevel, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&UartConfiguration->TxTrigLevel, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&UartConfiguration->RxDataPolarity, 1);
    DLPC_COMMON_PackBytes((uint8_t*)&UartConfiguration->RxDataSource, 1);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadUartConfiguration(DLPC654X_UrtPortT_e Port, DLPC654X_UartConfiguration_s *UartConfiguration)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0x6D);
    DLPC_COMMON_PackBytes((uint8_t*)&Port, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(10);
    if (Status == 0)
    {
        UartConfiguration->Enable = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(1);
        UartConfiguration->BaudRate = (DLPC654X_UrtBaudRateT_e)*(DLPC_COMMON_UnpackBytes(1));
        UartConfiguration->DataBits = (DLPC654X_UrtDataBitsT_e)*(DLPC_COMMON_UnpackBytes(1));
        UartConfiguration->StopBits = (DLPC654X_UrtStopBitsT_e)*(DLPC_COMMON_UnpackBytes(1));
        UartConfiguration->Parity = (DLPC654X_UrtParityT_e)*(DLPC_COMMON_UnpackBytes(1));
        UartConfiguration->FlowControl = (DLPC654X_UrtFlowControlT_e)*(DLPC_COMMON_UnpackBytes(1));
        UartConfiguration->RxTrigLevel = (DLPC654X_UrtFifoTriggerLevelT_e)*(DLPC_COMMON_UnpackBytes(1));
        UartConfiguration->TxTrigLevel = (DLPC654X_UrtFifoTriggerLevelT_e)*(DLPC_COMMON_UnpackBytes(1));
        UartConfiguration->RxDataPolarity = (DLPC654X_UrtRxDataPolarityT_e)*(DLPC_COMMON_UnpackBytes(1));
        UartConfiguration->RxDataSource = (DLPC654X_UrtRxDataSourceT_e)*(DLPC_COMMON_UnpackBytes(1));
    }
    return Status;
}

uint32_t DLPC654X_WriteXprFixedOutputEnable(uint8_t ChannelNum, bool FixedOutputEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x0);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_SetBits((int32_t)FixedOutputEnable, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprFixedOutputEnable(uint8_t ChannelNum, bool *FixedOutputEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x0);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *FixedOutputEnable = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprDacGain(uint8_t ChannelNum, double DacGain)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x1);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprDacGain(uint8_t ChannelNum, double *DacGain)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x1);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprSubframeDelay(uint8_t ChannelNum, uint32_t SubframeDelay)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x2);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_SetBits((int32_t)SubframeDelay, 32, 0);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprSubframeDelay(uint8_t ChannelNum, uint32_t *SubframeDelay)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x2);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *SubframeDelay = (uint32_t)DLPC_COMMON_GetBits(32, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}
 
uint32_t DLPC654X_ReadXprActuatorType(uint8_t *ActuatorType)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x3);     // CommandNum
    DLPC_COMMON_PackByte(0x0);     // ChannelNum

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *ActuatorType = (uint8_t)DLPC_COMMON_GetBits(8, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprOutputEnable(uint8_t ChannelNum, bool OutputEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x4);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_SetBits((int32_t)OutputEnable, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprOutputEnable(uint8_t ChannelNum, bool *OutputEnable)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x4);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *OutputEnable = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprClockWidth(uint8_t ChannelNum, uint32_t ClockWidth)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x5);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_SetBits((int32_t)ClockWidth, 32, 0);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprClockWidth(uint8_t ChannelNum, uint32_t *ClockWidth)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x5);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *ClockWidth = (uint32_t)DLPC_COMMON_GetBits(32, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprDacOffset(uint8_t ChannelNum, int8_t DacOffset)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x6);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprDacOffset(uint8_t ChannelNum, int8_t *DacOffset)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x6);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprNumberOfSegments(uint8_t ChannelNum, uint8_t NumberOfSegments)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x7);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_SetBits((int32_t)NumberOfSegments, 8, 0);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprNumberOfSegments(uint8_t ChannelNum, uint8_t *NumberOfSegments)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x7);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *NumberOfSegments = (uint8_t)DLPC_COMMON_GetBits(8, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprSegmentLength(uint8_t ChannelNum, uint16_t SegmentLength)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x8);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_SetBits((int32_t)SegmentLength, 16, 0);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprSegmentLength(uint8_t ChannelNum, uint16_t *SegmentLength)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x8);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *SegmentLength = (uint16_t)DLPC_COMMON_GetBits(16, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprInvertPwmA(uint8_t ChannelNum, bool InvertPwmA)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x9);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_SetBits((int32_t)InvertPwmA, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprInvertPwmA(uint8_t ChannelNum, bool *InvertPwmA)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0x9);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *InvertPwmA = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprInvertPwmB(uint8_t ChannelNum, bool InvertPwmB)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0xA);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_SetBits((int32_t)InvertPwmB, 1, 0);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprInvertPwmB(uint8_t ChannelNum, bool *InvertPwmB)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0xA);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *InvertPwmB = DLPC_COMMON_GetBits(1, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprSubframeFilterValue(uint8_t ChannelNum, uint8_t SubframeFilterValue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0xB);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_SetBits((int32_t)SubframeFilterValue, 8, 0);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprSubframeFilterValue(uint8_t ChannelNum, uint8_t *SubframeFilterValue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0xB);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *SubframeFilterValue = (uint8_t)DLPC_COMMON_GetBits(8, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprSubframeWatchDog(uint8_t ChannelNum, uint16_t SubframeWatchDog)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0xC);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_SetBits((int32_t)SubframeWatchDog, 16, 0);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprSubframeWatchDog(uint8_t ChannelNum, uint16_t *SubframeWatchDog)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0xC);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        *SubframeWatchDog = (uint16_t)DLPC_COMMON_GetBits(16, 0, false);
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}

uint32_t DLPC654X_WriteXprFixedOutputValue(uint8_t ChannelNum, int8_t FixedOutputValue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0xD);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);
    DLPC_COMMON_MoveWriteBufferPointer(4);

    DLPC_COMMON_SetCommandDestination(4);
    Status = DLPC_COMMON_SendWrite();
    return Status;
}
 
uint32_t DLPC654X_ReadXprFixedOutputValue(uint8_t ChannelNum, int8_t *FixedOutputValue)
{
    uint32_t Status = 0;

    DLPC_COMMON_ClearWriteBuffer();
    DLPC_COMMON_ClearReadBuffer();

    DLPC_COMMON_PackOpcode(1, 0xB5);
    DLPC_COMMON_PackByte(0xD);     // CommandNum
    DLPC_COMMON_PackBytes((uint8_t*)&ChannelNum, 1);

    DLPC_COMMON_SetCommandDestination(4);

    Status = DLPC_COMMON_SendRead(4);
    if (Status == 0)
    {
        DLPC_COMMON_MoveReadBufferPointer(4);
    }
    return Status;
}
