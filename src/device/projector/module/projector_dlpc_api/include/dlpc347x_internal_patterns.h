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
 * \brief  Defines the APIs for creating internal pattern data for the 347x
 *         controllers (DLP2010, DLP3010, and DLP4710).
 */

#ifndef DLPC34XX_INT_PAT_H
#define DLPC34XX_INT_PAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stdint.h"
#include "stdbool.h"

#define ERR_UNSUPPORTED_DMD 100

typedef enum
{
    DLPC34XX_INT_PAT_BITDEPTH_ONE = 1,
    DLPC34XX_INT_PAT_BITDEPTH_EIGHT = 8
} DLPC34XX_INT_PAT_BitDepth_e;

typedef enum
{
    DLPC34XX_INT_PAT_DIRECTION_HORIZONTAL,
    DLPC34XX_INT_PAT_DIRECTION_VERTICAL
} DLPC34XX_INT_PAT_Direction_e;

typedef enum
{
    DLPC34XX_INT_PAT_ILLUMINATION_NONE = 0,
    DLPC34XX_INT_PAT_ILLUMINATION_RED = 1,
    DLPC34XX_INT_PAT_ILLUMINATION_GREEN = 2,
    DLPC34XX_INT_PAT_ILLUMINATION_BLUE = 4,
    DLPC34XX_INT_PAT_ILLUMINATION_RGB = 7
} DLPC34XX_INT_PAT_IlluminationSelect_e;

typedef enum {
    DLPC34XX_INT_PAT_DMD_DLP2010,
    DLPC34XX_INT_PAT_DMD_DLP3010,
    DLPC34XX_INT_PAT_DMD_DLP4710
} DLPC34XX_INT_PAT_DMD_e;

typedef struct
{
    /**
     * Number of bytes in the pixel array 
     */
    uint32_t PixelArrayCount;

    /** 
     * The 1-D pixel data array. 
     *
     * For 1-bit patterns, each byte should 0 or 1. 
     * For 8-bit patterns, each byte can be 0-255.
     * For horizontal patterns, the number of bytes should be equal to the DMD_HEIGHT
     * For vertical patterns, the number of bytes should be equal to the DMD_WIDTH
     * If the number of bytes is less than the expected value, the missing pixel
     * data will be filled with zeros.
     * If the number of bytes is greater than the expected value, the excess bytes
     * are ignored.
     */
    uint8_t* PixelArray;
} DLPC34XX_INT_PAT_PatternData_s;

typedef struct
{
    DLPC34XX_INT_PAT_BitDepth_e     BitDepth;
    DLPC34XX_INT_PAT_Direction_e    Direction;
    uint32_t                        PatternCount;
    DLPC34XX_INT_PAT_PatternData_s* PatternArray;
} DLPC34XX_INT_PAT_PatternSet_s;

typedef struct
{
    uint8_t                               PatternSetIndex;
    uint8_t                               NumDisplayPatterns;
    DLPC34XX_INT_PAT_IlluminationSelect_e IlluminationSelect;
    bool                                  InvertPatterns;
    uint32_t                              IlluminationTimeInMicroseconds;
    uint32_t                              PreIlluminationDarkTimeInMicroseconds;
    uint32_t                              PostIlluminationDarkTimeInMicroseconds;
} DLPC34XX_INT_PAT_PatternOrderTableEntry_s;

/**
 * The callback used to transfer pattern data to the caller. The pattern generation
 * function may transfer one or more bytes at a time.
 *
 * \param[in] Length Number of bytes transferred
 * \param[in] Data   Pointer to the data bytes
 */
typedef void(*DLPC34XX_INT_PAT_WritePatternDataCallback)(uint8_t Length, uint8_t* Data);

/**
 * Generates the pattern data block from the given inputs. In order to avoid
 * dynamic memory allocation, this function uses a callback to transfer data to
 * the caller as it generates pattern data. This function calls the 
 * WritePatternDataCallback function several times with one or more bytes of 
 * pattern data transfered to the caller at a time. The function is synchronous
 * and waits for the callback to finish execution before continuing generation
 * of the rest of the data.
 *
 * \param[in] DMD                      The DMD for which pattern data is being
 *                                     generated
 * \param[in] PatternSetCount          Number of pattern sets
 * \param[in] PatternSetArray          An array of DLPC34XX_INT_PAT_PatternSet_s
 * \param[in] PatternOrderTableCount   Number of rows in the pattern order table
 * \param[in] PatternOrderTable        An array of DLPC34XX_INT_PAT_PatternOrderTableEntry_s
 * \param[in] WritePatternDataCallback The callback used to transfer data to 
 *                                     the caller
 * \param[in] EastWestFlip             Whether to E/W flip pattern data
 * \param[in] LongAxisFlip             Whether to flip pattern data along the long axis
 *
 * \return DLPC_SUCCESS         if successful
 *         ERR_UNSUPPORTED_DMD  if the DMD is not supported
 */
uint32_t DLPC34XX_INT_PAT_GeneratePatternDataBlock(
    DLPC34XX_INT_PAT_DMD_e                     DMD,
    uint32_t                                   PatternSetCount,
    DLPC34XX_INT_PAT_PatternSet_s*             PatternSetArray,
    uint32_t                                   PatternOrderTableCount,
    DLPC34XX_INT_PAT_PatternOrderTableEntry_s* PatternOrderTable,
    DLPC34XX_INT_PAT_WritePatternDataCallback  WritePatternDataCallback,
    bool                                       EastWestFlip,
    bool                                       LongAxisFlip
);

/**
 * Gets the size of the pattern data block in bytes for the given inputs
 *
 * \param[in] DMD                    The DMD for which pattern data is being 
 *                                   generated
 * \param[in] PatternSetCount        Number of pattern sets
 * \param[in] PatternSetArray        An array of DLPC34XX_INT_PAT_PatternSet_s
 * \param[in] PatternOrderTableCount Number of rows in the pattern order table
 * \param[in] PatternOrderTable      An array of DLPC34XX_INT_PAT_PatternOrderTableEntry_s
 *
 * \return UINT32_MAX if the DMD is not supported
 *         otherwise, the size of the pattern data block in bytes
 */
uint32_t DLPC34XX_INT_PAT_GetPatternDataBlockSize(
    DLPC34XX_INT_PAT_DMD_e                     DMD,
    uint32_t                                   PatternSetCount,
    DLPC34XX_INT_PAT_PatternSet_s*             PatternSetArray,
    uint32_t                                   PatternOrderTableCount,
    DLPC34XX_INT_PAT_PatternOrderTableEntry_s* PatternOrderTable
);

#ifdef __cplusplus    /* matches __cplusplus construct above */
}
#endif
#endif /* DLPC34XX_INT_PAT_H */