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
 * \brief  Bootloader commands, Projector Control, Formatter Only Commands
 */

#ifndef DLPC654X_H
#define DLPC654X_H

#ifdef __cplusplus
extern "C" {
#endif

#include "dlpc_common.h"
#include "stdbool.h"
#include "stdint.h"

typedef enum
{
    DLPC654X_SCST_CS0 = 0x0,                                          /**< Cs0 */
    DLPC654X_SCST_CS1 = 0x1,                                          /**< Cs1 */
    DLPC654X_SCST_CS2 = 0x2,                                          /**< Cs2 */
    DLPC654X_SCST_CS3 = 0x3,                                          /**< Cs3 */
    DLPC654X_SCST_CS4 = 0x4,                                          /**< Cs4 */
    DLPC654X_SCST_CS5 = 0x5,                                          /**< Cs5 */
    DLPC654X_SCST_CS6 = 0x6,                                          /**< Cs6 */
    DLPC654X_SCST_BCSZ = 0x7,                                         /**< Bcsz */
    DLPC654X_SCST_CS_GPIO = 0x8,                                      /**< Cs Gpio */
    DLPC654X_SCST_CSMAX = 0x9,                                        /**< Csmax */
} DLPC654X_SspChipSelectT_e;

typedef enum
{
    DLPC654X_SDLET_DLP_READY = 0x0,                                   /**< Single colored line(s) encoding. Must specify with @ref SRC 3D Signature t. */
    DLPC654X_SDLET_NO_ENCODING = 0x1,                                 /**< No Encoding */
    DLPC654X_SDLET_LR7525_ENCODING = 0x2,                             /**< Lr 75 25 Encoding */
} DLPC654X_Src3DLrEncodingT_e;

typedef enum
{
    DLPC654X_SCTT_TABLE0 = 0x0,                                       /**< Table0 */
    DLPC654X_SCTT_TABLE1 = 0x1,                                       /**< Table1 */
    DLPC654X_SCTT_TABLE2 = 0x2,                                       /**< Table2 */
    DLPC654X_SCTT_TABLE3 = 0x3,                                       /**< Table3 */
    DLPC654X_SCTT_TABLE4 = 0x4,                                       /**< Table4 */
    DLPC654X_SCTT_TABLE5 = 0x5,                                       /**< Table5 */
    DLPC654X_SCTT_TABLE6 = 0x6,                                       /**< Table6 */
    DLPC654X_SCTT_TABLE7 = 0x7,                                       /**< Table7 */
    DLPC654X_SCTT_MAXTABLE = 0x8,                                     /**< Maxtable */
} DLPC654X_SrcCscTablesT_e;

typedef enum
{
    DLPC654X_SCST_TERMINATED = 0x0,                                   /**< Image Capture Terminated because of error or Timeout */
    DLPC654X_SCST_BUFFER_WRITE_IN_PROGRESS = 0x1,                     /**< External Image is being written into the internal DRAM Splash buffer */
    DLPC654X_SCST_BUFFER_WRITE_COMPLETE = 0x2,                        /**< Image is successfully captured into internal DRAM Splash buffer */
    DLPC654X_SCST_FLASH_WRITE_IN_PROGRESS = 0x3,                      /**< Image is being programmed into the Flash memory */
    DLPC654X_SCST_FLASH_WRITE_COMPLETE = 0x4,                         /**< Image is successfully programmed into Flash memory */
} DLPC654X_SplashCaptureStateT_e;

typedef enum
{
    DLPC654X_AFST_ANR_STRENGTH1 = 0x0,                                /**< Filtering Strength 1 (lowest strength) */
    DLPC654X_AFST_ANR_STRENGTH2 = 0x1,                                /**< Filtering Strength 2 */
    DLPC654X_AFST_ANR_STRENGTH3 = 0x2,                                /**< Filtering Strength 3 */
    DLPC654X_AFST_ANR_STRENGTH4 = 0x3,                                /**< Filtering Strength 4 */
    DLPC654X_AFST_ANR_STRENGTH5 = 0x4,                                /**< Filtering Strength 5 */
    DLPC654X_AFST_ANR_STRENGTH6 = 0x5,                                /**< Filtering Strength 6 */
    DLPC654X_AFST_ANR_STRENGTH7 = 0x6,                                /**< Filtering Strength 7 */
    DLPC654X_AFST_ANR_STRENGTH8 = 0x7,                                /**< Filtering Strength 8 */
    DLPC654X_AFST_ANR_STRENGTH9 = 0x8,                                /**< Filtering Strength 9 */
    DLPC654X_AFST_ANR_STRENGTH10 = 0x9,                               /**< Filtering Strength 10 */
    DLPC654X_AFST_ANR_STRENGTH11 = 0xA,                               /**< Filtering Strength 11 */
    DLPC654X_AFST_ANR_STRENGTH12 = 0xB,                               /**< Filtering Strength 12 */
    DLPC654X_AFST_ANR_STRENGTH13 = 0xC,                               /**< Filtering Strength 13 */
    DLPC654X_AFST_ANR_STRENGTH14 = 0xD,                               /**< Filtering Strength 14 */
    DLPC654X_AFST_ANR_STRENGTH15 = 0xE,                               /**< Filtering Strength 15 */
    DLPC654X_AFST_ANR_STRENGTH16 = 0xF,                               /**< Filtering Strength 16 (highest strength) */
} DLPC654X_AnrFilterStrengthT_e;

typedef enum
{
    DLPC654X_SDT_PAD = 0x0,                                           /**< Pad */
    DLPC654X_SDT_PMD = 0x1,                                           /**< Pmd */
} DLPC654X_SspDeviceT_e;

typedef enum
{
    DLPC654X_UFTLT_ONE_EIGHTH_FULL = 0x0,                             /**< One Eighth Full */
    DLPC654X_UFTLT_ONE_FOURTH_FULL = 0x1,                             /**< One Fourth Full */
    DLPC654X_UFTLT_ONE_HALF_FULL = 0x2,                               /**< One Half Full */
    DLPC654X_UFTLT_THREE_FOURTHS_FULL = 0x3,                          /**< Three Fourths Full */
    DLPC654X_UFTLT_SEVEN_EIGHTHS_FULL = 0x4,                          /**< Seven Eighths Full */
} DLPC654X_UrtFifoTriggerLevelT_e;

typedef enum
{
    DLPC654X_GPT_PIN0 = 0x0,                                          /**< Pin0 */
    DLPC654X_GPT_PIN1 = 0x1,                                          /**< Pin1 */
    DLPC654X_GPT_PIN2 = 0x2,                                          /**< Pin2 */
    DLPC654X_GPT_PIN3 = 0x3,                                          /**< Pin3 */
    DLPC654X_GPT_PIN4 = 0x4,                                          /**< Pin4 */
    DLPC654X_GPT_PIN5 = 0x5,                                          /**< Pin5 */
    DLPC654X_GPT_PIN6 = 0x6,                                          /**< Pin6 */
    DLPC654X_GPT_PIN7 = 0x7,                                          /**< Pin7 */
    DLPC654X_GPT_PIN8 = 0x8,                                          /**< Pin8 */
    DLPC654X_GPT_PIN9 = 0x9,                                          /**< Pin9 */
    DLPC654X_GPT_PIN10 = 0xA,                                         /**< Pin10 */
    DLPC654X_GPT_PIN11 = 0xB,                                         /**< Pin11 */
    DLPC654X_GPT_PIN12 = 0xC,                                         /**< Pin12 */
    DLPC654X_GPT_PIN13 = 0xD,                                         /**< Pin13 */
    DLPC654X_GPT_PIN14 = 0xE,                                         /**< Pin14 */
    DLPC654X_GPT_PIN15 = 0xF,                                         /**< Pin15 */
    DLPC654X_GPT_PIN16 = 0x10,                                        /**< Pin16 */
    DLPC654X_GPT_PIN17 = 0x11,                                        /**< Pin17 */
    DLPC654X_GPT_PIN18 = 0x12,                                        /**< Pin18 */
    DLPC654X_GPT_PIN19 = 0x13,                                        /**< Pin19 */
    DLPC654X_GPT_PIN20 = 0x14,                                        /**< Pin20 */
    DLPC654X_GPT_PIN21 = 0x15,                                        /**< Pin21 */
    DLPC654X_GPT_PIN22 = 0x16,                                        /**< Pin22 */
    DLPC654X_GPT_PIN23 = 0x17,                                        /**< Pin23 */
    DLPC654X_GPT_PIN24 = 0x18,                                        /**< Pin24 */
    DLPC654X_GPT_PIN25 = 0x19,                                        /**< Pin25 */
    DLPC654X_GPT_PIN26 = 0x1A,                                        /**< Pin26 */
    DLPC654X_GPT_PIN27 = 0x1B,                                        /**< Pin27 */
    DLPC654X_GPT_PIN28 = 0x1C,                                        /**< Pin28 */
    DLPC654X_GPT_PIN29 = 0x1D,                                        /**< Pin29 */
    DLPC654X_GPT_PIN30 = 0x1E,                                        /**< Pin30 */
    DLPC654X_GPT_PIN31 = 0x1F,                                        /**< Pin31 */
    DLPC654X_GPT_PIN32 = 0x20,                                        /**< Pin32 */
    DLPC654X_GPT_PIN33 = 0x21,                                        /**< Pin33 */
    DLPC654X_GPT_PIN34 = 0x22,                                        /**< Pin34 */
    DLPC654X_GPT_PIN35 = 0x23,                                        /**< Pin35 */
    DLPC654X_GPT_PIN36 = 0x24,                                        /**< Pin36 */
    DLPC654X_GPT_PIN37 = 0x25,                                        /**< Pin37 */
    DLPC654X_GPT_PIN38 = 0x26,                                        /**< Pin38 */
    DLPC654X_GPT_PIN39 = 0x27,                                        /**< Pin39 */
    DLPC654X_GPT_PIN40 = 0x28,                                        /**< Pin40 */
    DLPC654X_GPT_PIN41 = 0x29,                                        /**< Pin41 */
    DLPC654X_GPT_PIN42 = 0x2A,                                        /**< Pin42 */
    DLPC654X_GPT_PIN43 = 0x2B,                                        /**< Pin43 */
    DLPC654X_GPT_PIN44 = 0x2C,                                        /**< Pin44 */
    DLPC654X_GPT_PIN45 = 0x2D,                                        /**< Pin45 */
    DLPC654X_GPT_PIN46 = 0x2E,                                        /**< Pin46 */
    DLPC654X_GPT_PIN47 = 0x2F,                                        /**< Pin47 */
    DLPC654X_GPT_PIN48 = 0x30,                                        /**< Pin48 */
    DLPC654X_GPT_PIN49 = 0x31,                                        /**< Pin49 */
    DLPC654X_GPT_PIN50 = 0x32,                                        /**< Pin50 */
    DLPC654X_GPT_PIN51 = 0x33,                                        /**< Pin51 */
    DLPC654X_GPT_PIN52 = 0x34,                                        /**< Pin52 */
    DLPC654X_GPT_PIN53 = 0x35,                                        /**< Pin53 */
    DLPC654X_GPT_PIN54 = 0x36,                                        /**< Pin54 */
    DLPC654X_GPT_PIN55 = 0x37,                                        /**< Pin55 */
    DLPC654X_GPT_PIN56 = 0x38,                                        /**< Pin56 */
    DLPC654X_GPT_PIN57 = 0x39,                                        /**< Pin57 */
    DLPC654X_GPT_PIN58 = 0x3A,                                        /**< Pin58 */
    DLPC654X_GPT_PIN59 = 0x3B,                                        /**< Pin59 */
    DLPC654X_GPT_PIN60 = 0x3C,                                        /**< Pin60 */
    DLPC654X_GPT_PIN61 = 0x3D,                                        /**< Pin61 */
    DLPC654X_GPT_PIN62 = 0x3E,                                        /**< Pin62 */
    DLPC654X_GPT_PIN63 = 0x3F,                                        /**< Pin63 */
    DLPC654X_GPT_PIN64 = 0x40,                                        /**< Pin64 */
    DLPC654X_GPT_PIN65 = 0x41,                                        /**< Pin65 */
    DLPC654X_GPT_PIN66 = 0x42,                                        /**< Pin66 */
    DLPC654X_GPT_PIN67 = 0x43,                                        /**< Pin67 */
    DLPC654X_GPT_PIN68 = 0x44,                                        /**< Pin68 */
    DLPC654X_GPT_PIN69 = 0x45,                                        /**< Pin69 */
    DLPC654X_GPT_PIN70 = 0x46,                                        /**< Pin70 */
    DLPC654X_GPT_PIN71 = 0x47,                                        /**< Pin71 */
    DLPC654X_GPT_PIN72 = 0x48,                                        /**< Pin72 */
    DLPC654X_GPT_PIN73 = 0x49,                                        /**< Pin73 */
    DLPC654X_GPT_PIN74 = 0x4A,                                        /**< Pin74 */
    DLPC654X_GPT_PIN75 = 0x4B,                                        /**< Pin75 */
    DLPC654X_GPT_PIN76 = 0x4C,                                        /**< Pin76 */
    DLPC654X_GPT_PIN77 = 0x4D,                                        /**< Pin77 */
    DLPC654X_GPT_PIN78 = 0x4E,                                        /**< Pin78 */
    DLPC654X_GPT_PIN79 = 0x4F,                                        /**< Pin79 */
    DLPC654X_GPT_PIN80 = 0x50,                                        /**< Pin80 */
    DLPC654X_GPT_PIN81 = 0x51,                                        /**< Pin81 */
    DLPC654X_GPT_PIN82 = 0x52,                                        /**< Pin82 */
    DLPC654X_GPT_PIN83 = 0x53,                                        /**< Pin83 */
    DLPC654X_GPT_PIN84 = 0x54,                                        /**< Pin84 */
    DLPC654X_GPT_PIN85 = 0x55,                                        /**< Pin85 */
    DLPC654X_GPT_PIN86 = 0x56,                                        /**< Pin86 */
    DLPC654X_GPT_PIN87 = 0x57,                                        /**< Pin87 */
    DLPC654X_GPT_MAX_PIN = 0x58,                                      /**< Max Pin */
} DLPC654X_GpioPinsT_e;

typedef enum
{
    DLPC654X_DMT_MOTION_ADAPTIVE = 0x0,                               /**< Motion Adaptive */
    DLPC654X_DMT_FIELD_JAM = 0x1,                                     /**< Field Jam */
    DLPC654X_DMT_SCALED_FIELD = 0x2,                                  /**< Scaled Field */
} DLPC654X_DeiModeT_e;

typedef enum
{
    DLPC654X_DMT_DDI_MACRO0 = 0x0,                                    /**< 0 */
    DLPC654X_DMT_DDI_MACRO1 = 0x1,                                    /**< 1 */
} DLPC654X_DdiMacroT_e;

typedef enum
{
    DLPC654X_DHM_AUTO = 0x0,                                          /**< Auto */
    DLPC654X_DHM_MANUAL = 0x1,                                        /**< Manual */
} DLPC654X_DpHdrMode_e;

typedef enum
{
    DLPC654X_URDST_RXD = 0x0,                                         /**< UART x.RXD is sourced by UART x RXD pin */
    DLPC654X_URDST_LAMPSTAT = 0x1,                                    /**< UART x.RXD is sourced by LAMPSTAT pin */
} DLPC654X_UrtRxDataSourceT_e;

typedef enum
{
    DLPC654X_ICPT_PORT0 = 0x0,                                        /**< I2C Port 0 */
    DLPC654X_ICPT_PORT1 = 0x1,                                        /**< I2C Port 1 */
    DLPC654X_ICPT_PORT2 = 0x2,                                        /**< I2C Port 2 */
    DLPC654X_ICPT_MAX_PORT = 0x3,                                     /**< Only three Ports are supported */
} DLPC654X_I2CPortT_e;

typedef enum
{
    DLPC654X_DBCT_BLACK = 0x0,                                        /**< Black color */
    DLPC654X_DBCT_DITHER_BLACK = 0x1,                                 /**< Reserved */
    DLPC654X_DBCT_WHITE = 0x2,                                        /**< White color */
    DLPC654X_DBCT_GREEN = 0x3,                                        /**< Green color */
    DLPC654X_DBCT_RED = 0x4,                                          /**< Red color */
    DLPC654X_DBCT_BLUE = 0x5,                                         /**< Blue color */
    DLPC654X_DBCT_YELLOW = 0x6,                                       /**< Yellow color */
    DLPC654X_DBCT_CYAN = 0x7,                                         /**< Cyan color */
    DLPC654X_DBCT_MAGENTA = 0x8,                                      /**< Magenta color */
    DLPC654X_DBCT_C1 = 0x9,                                           /**< Reserved */
    DLPC654X_DBCT_C2 = 0xA,                                           /**< Reserved */
} DLPC654X_DispBackgroundColorT_e;

typedef enum
{
    DLPC654X_TPT_SOLID_FIELD = 0x0,                                   /**< Solid Field */
    DLPC654X_TPT_HORZ_RAMP = 0x1,                                     /**< Horz Ramp */
    DLPC654X_TPT_VERT_RAMP = 0x2,                                     /**< Vert Ramp */
    DLPC654X_TPT_HORZ_LINES = 0x3,                                    /**< Horz Lines */
    DLPC654X_TPT_DIAG_LINES = 0x4,                                    /**< Diag Lines */
    DLPC654X_TPT_VERT_LINES = 0x5,                                    /**< Vert Lines */
    DLPC654X_TPT_GRID = 0x6,                                          /**< Grid */
    DLPC654X_TPT_CHECKERBOARD = 0x7,                                  /**< Checkerboard */
    DLPC654X_TPT_COLORBARS = 0x8,                                     /**< Colorbars */
    DLPC654X_TPT_MULT_COLOR_HORZ_RAMP = 0x9,                          /**< Mult Color Horz Ramp */
    DLPC654X_TPT_FIXED_STEP_HORZ_RAMP = 0xA,                          /**< Fixed Step Horz Ramp */
    DLPC654X_TPT_DIAMOND_DIAG_LINES = 0xB,                            /**< Diamond Diag Lines */
} DLPC654X_TpgPatternT_e;

typedef enum
{
    DLPC654X_CSTT_TO_BOOT = 0x0,                                      /**< To Boot */
    DLPC654X_CSTT_TO_APP_VIA_RESET = 0x1,                             /**< To App Via Reset */
    DLPC654X_CSTT_TO_APP_DIRECT = 0x2,                                /**< Switch to application regardless of the BOOT HOLD GPIO State. This option is provided for debug purposes only */
} DLPC654X_CmdSwitchTypeT_e;

typedef enum
{
    DLPC654X_XWC_FIXEDEN = 0x0,                                       /**< Fixed Output Enable */
    DLPC654X_XWC_DACGAIN = 0x1,                                       /**< DAC Gain */
    DLPC654X_XWC_SFDELAY = 0x2,                                       /**< Subframe delay */
    DLPC654X_XWC_ACTTYPE = 0x3,                                       /**< Actuator Type (READ ONLY) */
    DLPC654X_XWC_OUTPUTSEL = 0x4,                                     /**< Output Enable/Disable */
    DLPC654X_XWC_CLOCKWIDTH = 0x5,                                    /**< Clock Width */
    DLPC654X_XWC_DACOFFSET = 0x6,                                     /**< DAC Offset */
    DLPC654X_XWC_RAMPLEN = 0x7,                                       /**< Number of Segments */
    DLPC654X_XWC_SEGMENTLEN = 0x8,                                    /**< Segment Length */
    DLPC654X_XWC_INVPWMA = 0x9,                                       /**< Invert PWM A */
    DLPC654X_XWC_INVPWMB = 0xA,                                       /**< Invert PWM B */
    DLPC654X_XWC_SFFILTERVAL = 0xB,                                   /**< Subframe Filter Value */
    DLPC654X_XWC_SFWATCHDOG = 0xC,                                    /**< Subframe Watch Dog */
    DLPC654X_XWC_FIXEDOUTPUTVAL = 0xD,                                /**< Fixed Output Value */
} DLPC654X_Xpr4WayCommand_e;

typedef enum
{
    DLPC654X_UPT_PORT0 = 0x0,                                         /**< Port0 */
    DLPC654X_UPT_PORT1 = 0x1,                                         /**< Port1 */
    DLPC654X_UPT_PORT2 = 0x2,                                         /**< Port2 */
} DLPC654X_UrtPortT_e;

typedef enum
{
    DLPC654X_GFT_SSP1_CLK = 0x0,                                      /**< Pin 0 */
    DLPC654X_GFT_SSP1_DI_IN = 0x1,                                    /**< Pin 1 */
    DLPC654X_GFT_SSP1_DO_OUT = 0x2,                                   /**< Pin 2 */
    DLPC654X_GFT_SSP1_CSZ0 = 0x3,                                     /**< Pin 3 */
    DLPC654X_GFT_SSP1_CSZ1 = 0x4,                                     /**< Pin 4 */
    DLPC654X_GFT_SSP1_CSZ2 = 0x5,                                     /**< Pin 5 */
    DLPC654X_GFT_SAS_CLK_OUT = 0x6,                                   /**< Pin 6 */
    DLPC654X_GFT_SAS_DI_IN = 0x7,                                     /**< Pin 7 */
    DLPC654X_GFT_SAS_DO_OUT = 0x8,                                    /**< Pin 8 */
    DLPC654X_GFT_SAS_CSZ_OUT = 0x9,                                   /**< Pin 9 */
    DLPC654X_GFT_SAS_INTGTR_EN_OUT = 0xA,                             /**< Pin 10 */
    DLPC654X_GFT_I2_C1 = 0xB,                                         /**< Pin 11 SCL & Pin 12 SDA */
    DLPC654X_GFT_UART1_TXD_RXD = 0xC,                                 /**< Pin 13 TXD (O) Pin 14 RXD (I) */
    DLPC654X_GFT_UART1_CTSZ_RTSZ = 0xD,                               /**< Pin 15 CTSZ (I) and Pin 16 RTSZ (O) */
    DLPC654X_GFT_IR0_IN = 0xE,                                        /**< Pin 18 */
    DLPC654X_GFT_IR1_IN = 0xF,                                        /**< Pin 19 */
    DLPC654X_GFT_PROG_AUX0_OUT = 0x10,                                /**< Pin 22 */
    DLPC654X_GFT_PROG_AUX1_OUT = 0x11,                                /**< Pin 23 */
    DLPC654X_GFT_PROG_AUX2_OUT = 0x12,                                /**< Pin 24 */
    DLPC654X_GFT_PROG_AUX3_OUT = 0x13,                                /**< Pin 25 */
    DLPC654X_GFT_PROG_AUX4_OUT = 0x14,                                /**< Pin 26 */
    DLPC654X_GFT_PROG_AUX5_OUT = 0x15,                                /**< Pin 27 */
    DLPC654X_GFT_PROG_AUX6_OUT = 0x16,                                /**< Pin 28 */
    DLPC654X_GFT_PROG_AUX7_OUT = 0x17,                                /**< Pin 29 */
    DLPC654X_GFT_PROG_AUX8_OUT = 0x18,                                /**< Pin 30 */
    DLPC654X_GFT_PROG_AUX9_OUT = 0x19,                                /**< Pin 31 */
    DLPC654X_GFT_PROG_AUX10_OUT = 0x1A,                               /**< Pin 32 */
    DLPC654X_GFT_PROG_AUX11_OUT = 0x1B,                               /**< Pin 33 */
    DLPC654X_GFT_PROG_AUX_INT0_OUT = 0x1C,                            /**< Pin 22 */
    DLPC654X_GFT_PROG_AUX_INT1_OUT = 0x1D,                            /**< Pin 23 */
    DLPC654X_GFT_PROG_AUX_INT2_OUT = 0x1E,                            /**< Pin 30 */
    DLPC654X_GFT_PROG_AUX_INT3_OUT = 0x1F,                            /**< Pin 31 */
    DLPC654X_GFT_LED_SELECT0_OUT = 0x20,                              /**< Pin 24 */
    DLPC654X_GFT_LED_SELECT1_OUT = 0x21,                              /**< Pin 25 */
    DLPC654X_GFT_LED_SELECT2_OUT = 0x22,                              /**< Pin 26 */
    DLPC654X_GFT_LED_SELECT3_OUT = 0x23,                              /**< Pin 27 */
    DLPC654X_GFT_LED_SELECT4_OUT = 0x24,                              /**< Pin 28 */
    DLPC654X_GFT_I2_C2_OPT2 = 0x25,                                   /**< Pin 32 SCL & Pin 33 SDA */
    DLPC654X_GFT_WRP_CAMERA_TRIG_OUT = 0x26,                          /**< Pin 34 */
    DLPC654X_GFT_GPCLK2_OUT_OPT2 = 0x27,                              /**< Pin 35 */
    DLPC654X_GFT_DAO_DO0_OUT = 0x28,                                  /**< Pin 35 */
    DLPC654X_GFT_DAO_DO1_OUT = 0x29,                                  /**< Pin 36 */
    DLPC654X_GFT_DAO_CLKOUT_OUT = 0x2A,                               /**< Pin 37 */
    DLPC654X_GFT_CW_INDEX1_OPT2 = 0x2B,                               /**< Pin 36 */
    DLPC654X_GFT_CW_INDEX2_OPT2 = 0x2C,                               /**< Pin 37 */
    DLPC654X_GFT_UART2_TXD_RXD_OPT1 = 0x2D,                           /**< Pin 38 TXD (O) Pin 39 RXD (I) */
    DLPC654X_GFT_HBT_DO_OUT = 0x2E,                                   /**< Pin 38 */
    DLPC654X_GFT_HBT_CLKOUT_OUT = 0x2F,                               /**< Pin 39 */
    DLPC654X_GFT_I2_C2_OPT1 = 0x30,                                   /**< Pin 41 SCL & Pin 42 SDA */
    DLPC654X_GFT_SSP2_CLK = 0x31,                                     /**< Pin 40 */
    DLPC654X_GFT_SSP2_DI_IN = 0x32,                                   /**< Pin 41 */
    DLPC654X_GFT_SSP2_DO_OUT = 0x33,                                  /**< Pin 42 */
    DLPC654X_GFT_SSP2_CSZ0 = 0x34,                                    /**< Pin 43 */
    DLPC654X_GFT_SSP2_CSZ1 = 0x35,                                    /**< Pin 44 */
    DLPC654X_GFT_SSP2_CSZ2 = 0x36,                                    /**< Pin 45 */
    DLPC654X_GFT_SSP2_BC_CSZ = 0x37,                                  /**< Pin 46 */
    DLPC654X_GFT_GPCLK3_OUT_OPT1 = 0x38,                              /**< Pin 43 */
    DLPC654X_GFT_GPCLK2_OUT_OPT1 = 0x39,                              /**< Pin 44 */
    DLPC654X_GFT_CW_INDEX1_OPT1 = 0x3A,                               /**< Pin 45 */
    DLPC654X_GFT_CW_INDEX2_OPT1 = 0x3B,                               /**< Pin 46 */
    DLPC654X_GFT_PMADDR23_OUT = 0x3C,                                 /**< Pin 47 */
    DLPC654X_GFT_SSP0_CSZ5 = 0x3D,                                    /**< Pin 48 */
    DLPC654X_GFT_SSP0_CSZ4 = 0x3E,                                    /**< Pin 49 */
    DLPC654X_GFT_SSP0_CSZ3 = 0x3F,                                    /**< Pin 50 */
    DLPC654X_GFT_DMD_PWR_EN_OUT = 0x40,                               /**< Pin 51 */
    DLPC654X_GFT_LAMPSYNC_OUT_OPT2 = 0x41,                            /**< Pin 51 */
    DLPC654X_GFT_LAMPCTRL_OUT = 0x42,                                 /**< Pin 52 */
    DLPC654X_GFT_LAMPSTAT_IN = 0x43,                                  /**< Pin 53 */
    DLPC654X_GFT_LED_DRIVER_ON_OUT = 0x44,                            /**< Pin 53 */
    DLPC654X_GFT_PWM_IN0 = 0x45,                                      /**< Pin 20 */
    DLPC654X_GFT_PWM_IN1 = 0x46,                                      /**< Pin 21 */
    DLPC654X_GFT_PWM_OUT0 = 0x47,                                     /**< Pin 56 */
    DLPC654X_GFT_PWM_OUT1 = 0x48,                                     /**< Pin 57 */
    DLPC654X_GFT_PWM_OUT2 = 0x49,                                     /**< Pin 58 */
    DLPC654X_GFT_PWM_OUT3 = 0x4A,                                     /**< Pin 59 */
    DLPC654X_GFT_PWM_OUT4 = 0x4B,                                     /**< Pin 60 */
    DLPC654X_GFT_PWM_OUT5 = 0x4C,                                     /**< Pin 61 */
    DLPC654X_GFT_PWM_OUT6 = 0x4D,                                     /**< Pin 62 */
    DLPC654X_GFT_PWM_OUT7 = 0x4E,                                     /**< Pin 63 */
    DLPC654X_GFT_PWM_OUT8 = 0x4F,                                     /**< Pin 54 */
    DLPC654X_GFT_PWM_OUT9 = 0x50,                                     /**< Pin 55 */
    DLPC654X_GFT_ALF_CLAMP_OUT = 0x51,                                /**< Pin 55 */
    DLPC654X_GFT_ALF_COAST_OUT = 0x52,                                /**< Pin 58 */
    DLPC654X_GFT_UART2_TXD_RXD_OPT2 = 0x53,                           /**< Pin 59 TXD (O) Pin 60 RXD (I) */
    DLPC654X_GFT_GP_CLK3_OUT_OPT2 = 0x54,                             /**< Pin 63 */
    DLPC654X_GFT_GP_CLK1_OUT = 0x55,                                  /**< Pin 64 */
    DLPC654X_GFT_AWC0_ENZ_OUT = 0x56,                                 /**< Pin 65 */
    DLPC654X_GFT_AWC0_DACCLK01_OUT = 0x57,                            /**< Pin 66 */
    DLPC654X_GFT_AWC0_DACS_PWMA0_OUT = 0x58,                          /**< Pin 67 */
    DLPC654X_GFT_AWC0_DACD_PWMB0_OUT = 0x59,                          /**< Pin 68 */
    DLPC654X_GFT_AWC0_DACS_PWMA1_OUT = 0x5A,                          /**< Pin 69 */
    DLPC654X_GFT_AWC0_DACD_PWMB1_OUT = 0x5B,                          /**< Pin 70 */
    DLPC654X_GFT_AWC1_OUT_ENZ_OUT = 0x5C,                             /**< Pin 71 */
    DLPC654X_GFT_AWC1_DACCLK01_OUT = 0x5D,                            /**< Pin 72 */
    DLPC654X_GFT_AWC1_DACS_PWMA0_OUT = 0x5E,                          /**< Pin 73 */
    DLPC654X_GFT_AWC1_DACD_PWMB0_OUT = 0x5F,                          /**< Pin 74 */
    DLPC654X_GFT_AWC1_DACS_PWMA1_OUT = 0x60,                          /**< Pin 75 */
    DLPC654X_GFT_AWC1_DACD_PWMB1_OUT = 0x61,                          /**< Pin 76 */
    DLPC654X_GFT_I2_C2_OPT3 = 0x62,                                   /**< Pin 67 SCL & Pin 68 SDA */
    DLPC654X_GFT_MEMAUX1_OUT_OPT1 = 0x63,                             /**< Pin 40 */
    DLPC654X_GFT_MEMAUX1_OUT_OPT2 = 0x64,                             /**< Pin 69 */
    DLPC654X_GFT_BT656_DATA0_IN = 0x65,                               /**< Pin 77 */
    DLPC654X_GFT_BT656_DATA1_IN = 0x66,                               /**< Pin 78 */
    DLPC654X_GFT_BT656_DATA2_IN = 0x67,                               /**< Pin 79 */
    DLPC654X_GFT_BT656_DATA3_IN = 0x68,                               /**< Pin 80 */
    DLPC654X_GFT_BT656_DATA4_IN = 0x69,                               /**< Pin 81 */
    DLPC654X_GFT_BT656_DATA5_IN = 0x6A,                               /**< Pin 82 */
    DLPC654X_GFT_BT656_DATA6_IN = 0x6B,                               /**< Pin 83 */
    DLPC654X_GFT_BT656_DATA7_IN = 0x6C,                               /**< Pin 84 */
    DLPC654X_GFT_BT656_DATA8_IN = 0x6D,                               /**< Pin 85 */
    DLPC654X_GFT_BT656_DATA9_IN = 0x6E,                               /**< Pin 86 */
    DLPC654X_GFT_BT656_CLK_IN = 0x6F,                                 /**< Pin 87 */
    DLPC654X_GFT_EFSYNC_OUT_DASYNC_IN = 0x70,                         /**< Pin 77 */
    DLPC654X_GFT_SEQ_SYNC = 0x71,                                     /**< Pin 78 */
    DLPC654X_GFT_HBT_DI0_IN = 0x72,                                   /**< Pin 79 */
    DLPC654X_GFT_HBT_CLKIN0_IN = 0x73,                                /**< Pin 80 */
    DLPC654X_GFT_HBT_DI1_IN = 0x74,                                   /**< Pin 81 */
    DLPC654X_GFT_HBT_CLKIN1_IN = 0x75,                                /**< Pin 82 */
    DLPC654X_GFT_HBT_DI2_IN = 0x76,                                   /**< Pin 83 */
    DLPC654X_GFT_HBT_CLKIN2_IN = 0x77,                                /**< Pin 84 */
    DLPC654X_GFT_DAO_DI0_IN = 0x78,                                   /**< Pin 85 */
    DLPC654X_GFT_DAO_DI1_IN = 0x79,                                   /**< Pin 86 */
    DLPC654X_GFT_DAO_CLKIN_IN = 0x7A,                                 /**< Pin 87 */
    DLPC654X_GFT_FN_MAX = 0x7B,                                       /**< Fn Max */
    DLPC654X_GFT_LAMPSYNC_OUT_OPT1 = 0x10,                            /**< Pin 22 */
    DLPC654X_GFT_LED_SENSE_PULSE_OUT = 0x10,                          /**< Pin 22 */
    DLPC654X_GFT_SEQ_INDEX_OUT = 0x11,                                /**< Pin 23 */
    DLPC654X_GFT_LED1_EN_OUT = 0x12,                                  /**< Pin 24 */
    DLPC654X_GFT_LED2_EN_OUT = 0x13,                                  /**< Pin 25 */
    DLPC654X_GFT_LED3_EN_OUT = 0x14,                                  /**< Pin 26 */
    DLPC654X_GFT_LED4_EN_OUT = 0x15,                                  /**< Pin 27 */
    DLPC654X_GFT_LED5_EN_OUT = 0x16,                                  /**< Pin 28 */
    DLPC654X_GFT_RED_LED_EN_OUT = 0x12,                               /**< Pin 24 */
    DLPC654X_GFT_GRN_LED_EN_OUT = 0x13,                               /**< Pin 25 */
    DLPC654X_GFT_BLU_LED_EN_OUT = 0x14,                               /**< Pin 26 */
    DLPC654X_GFT_IR_LED_EN_OUT = 0x15,                                /**< Pin 27 */
    DLPC654X_GFT_UV_LED_EN_OUT = 0x16,                                /**< Pin 28 */
    DLPC654X_GFT_SSI_SUBFRAME_OUT = 0x17,                             /**< Pin 29 */
    DLPC654X_GFT_XPR_X_OUT = 0x18,                                    /**< Pin 30 */
    DLPC654X_GFT_XPR_Y_OUT = 0x19,                                    /**< Pin 31 */
    DLPC654X_GFT_CW_REV_OUT = 0x1A,                                   /**< Pin 32 */
    DLPC654X_GFT_CW_SPOKE_OUT = 0x1B,                                 /**< Pin 32 */
    DLPC654X_GFT_PWM_LED1 = 0x4A,                                     /**< Pin 59 */
    DLPC654X_GFT_PWM_LED2 = 0x4B,                                     /**< Pin 60 */
    DLPC654X_GFT_PWM_LED3 = 0x4C,                                     /**< Pin 61 */
    DLPC654X_GFT_PWM_LED4 = 0x4D,                                     /**< Pin 62 */
    DLPC654X_GFT_PWM_LED5 = 0x4E,                                     /**< Pin 63 */
    DLPC654X_GFT_PWM_STD0 = 0x47,                                     /**< Pin 56 */
    DLPC654X_GFT_PWM_STD1 = 0x48,                                     /**< Pin 57 */
    DLPC654X_GFT_PWM_STD2 = 0x49,                                     /**< Pin 58 */
    DLPC654X_GFT_PWM_RED_LED = 0x4A,                                  /**< Pin 59 */
    DLPC654X_GFT_PWM_GRN_LED = 0x4B,                                  /**< Pin 60 */
    DLPC654X_GFT_PWM_BLU_LED = 0x4C,                                  /**< Pin 61 */
    DLPC654X_GFT_PWM_IR_LED = 0x4D,                                   /**< Pin 62 */
    DLPC654X_GFT_PWM_UV_LED = 0x4E,                                   /**< Pin 63 */
    DLPC654X_GFT_PWM_CW1 = 0x4F,                                      /**< Pin 54 */
    DLPC654X_GFT_PWM_CW2 = 0x50,                                      /**< Pin 55 */
} DLPC654X_GpioFuncT_e;

typedef enum
{
    DLPC654X_DXEMT_AUTO = 0x0,                                        /**< Decision to enable XPR is based on source resolution */
    DLPC654X_DXEMT_ON = 0x1,                                          /**< XPR should always be turned on */
    DLPC654X_DXEMT_OFF = 0x2,                                         /**< XPR should always be turned off */
} DLPC654X_DispXprEnableModeT_e;

typedef enum
{
    DLPC654X_CWBTS_API = 0x0,                                         /**< Api */
    DLPC654X_CWBTS_EEPROM = 0x1,                                      /**< Eeprom */
} DLPC654X_CmdWpcBrightnessTableSource_e;

typedef enum
{
    DLPC654X_TLCT_RED = 0x0,                                          /**< Red */
    DLPC654X_TLCT_GREEN = 0x1,                                        /**< Green */
    DLPC654X_TLCT_BLUE = 0x2,                                         /**< Blue */
} DLPC654X_TpgLineColorT_e;

typedef enum
{
    DLPC654X_UPT_NONE = 0x0,                                          /**< Parity bit is neither transmitted or checked */
    DLPC654X_UPT_EVEN = 0x1,                                          /**< Even parity is transmitted and checked */
    DLPC654X_UPT_ODD = 0x2,                                           /**< Odd parity is transmitted and checked */
} DLPC654X_UrtParityT_e;

typedef enum
{
    DLPC654X_TCT_BLACK = 0x0,                                         /**< Black */
    DLPC654X_TCT_RED = 0x1,                                           /**< Red */
    DLPC654X_TCT_GREEN = 0x2,                                         /**< Green */
    DLPC654X_TCT_BLUE = 0x4,                                          /**< Blue */
    DLPC654X_TCT_YELLOW = 0x3,                                        /**< Yellow */
    DLPC654X_TCT_MAGENTA = 0x5,                                       /**< Magenta */
    DLPC654X_TCT_CYAN = 0x6,                                          /**< Cyan */
    DLPC654X_TCT_WHITE = 0x7,                                         /**< White */
} DLPC654X_TpgColorT_e;

typedef enum
{
    DLPC654X_TST_DELAYED_CW_INDEX = 0x0,                              /**< Delayed Color Wheel Index */
    DLPC654X_TST_CW_INDEX = 0x1,                                      /**< Color Wheel Index */
    DLPC654X_TST_VSYNC = 0x2,                                         /**< VSYNC. This VSync drives the Display SubSystem. */
    DLPC654X_TST_SEQUENCE_INDEX = 0x3,                                /**< Sequence Index */
    DLPC654X_TST_SPOKE_MARKER = 0x4,                                  /**< Spoke Marker */
    DLPC654X_TST_REVOLUTION_MARKER = 0x5,                             /**< Revolution Marker */
    DLPC654X_TST_DELAY0 = 0x6,                                        /**< Delay timer outputs */
    DLPC654X_TST_DELAY1 = 0x7,                                        /**< Delay1 */
    DLPC654X_TST_DELAY2 = 0x8,                                        /**< Delay2 */
    DLPC654X_TST_DELAY3 = 0x9,                                        /**< Delay3 */
    DLPC654X_TST_DELAY4 = 0xA,                                        /**< Delay4 */
    DLPC654X_TST_DELAY5 = 0xB,                                        /**< Delay5 */
    DLPC654X_TST_DELAY6 = 0xC,                                        /**< Delay6 */
    DLPC654X_TST_DELAY7 = 0xD,                                        /**< Delay7 */
    DLPC654X_TST_CAP0 = 0xE,                                          /**< Capture timer interrupts */
    DLPC654X_TST_CAP1 = 0xF,                                          /**< Cap1 */
    DLPC654X_TST_CAP2 = 0x10,                                         /**< Cap2 */
    DLPC654X_TST_CAP3 = 0x11,                                         /**< Cap3 */
    DLPC654X_TST_CAP4 = 0x12,                                         /**< Cap4 */
    DLPC654X_TST_CAP5 = 0x13,                                         /**< Cap5 */
    DLPC654X_TST_CAP6 = 0x14,                                         /**< Cap6 */
    DLPC654X_TST_CAP7 = 0x15,                                         /**< Cap7 */
    DLPC654X_TST_CAP8 = 0x16,                                         /**< Cap8 */
    DLPC654X_TST_SEQ_AUX0 = 0x17,                                     /**< Sequencer Auxiliary bits */
    DLPC654X_TST_SEQ_AUX1 = 0x18,                                     /**< Seq Aux1 */
    DLPC654X_TST_SEQ_AUX2 = 0x19,                                     /**< Seq Aux2 */
    DLPC654X_TST_SEQ_AUX3 = 0x1A,                                     /**< Seq Aux3 */
    DLPC654X_TST_SEQ_AUX4 = 0x1B,                                     /**< Seq Aux4 */
    DLPC654X_TST_SEQ_AUX5 = 0x1C,                                     /**< Seq Aux5 */
    DLPC654X_TST_SEQ_AUX6 = 0x1D,                                     /**< Seq Aux6 */
    DLPC654X_TST_SEQ_AUX7 = 0x1E,                                     /**< Seq Aux7 */
    DLPC654X_TST_DELAYED_CW2_INDEX = 0x1F,                            /**< Delayed Color Wheel Index */
    DLPC654X_TST_CW2_INDEX = 0x20,                                    /**< Color Wheel Index */
    DLPC654X_TST_SIGNAL_MAXIMUM = 0x21,                               /**< Signal Maximum */
} DLPC654X_TpmSignalType_e;

typedef enum
{
    DLPC654X_SPT_PORT0 = 0x0,                                         /**< Port0 */
    DLPC654X_SPT_PORT1 = 0x1,                                         /**< Port1 */
    DLPC654X_SPT_PORT2 = 0x2,                                         /**< Port2 */
} DLPC654X_SspPortT_e;

typedef enum
{
    DLPC654X_ACDME_STANDARD = 0x0,                                    /**< Standard */
    DLPC654X_ACDME_ENHANCED = 0x1,                                    /**< Enhanced */
} DLPC654X_AlcClockDetectionModeEnum_e;

typedef enum
{
    DLPC654X_SSCT_PORT_SYNC_NOINV = 0x0,                              /**< Input port sync is not modified(passed through). */
    DLPC654X_SSCT_PORT_SYNC_INV = 0x1,                                /**< Input port sync is inverted. */
    DLPC654X_SSCT_ALF_SYNC = 0x2,                                     /**< ALF Sync is selected as the Port sync source. Use when Autolock is used for source detection. */
    DLPC654X_SSCT_DECODED_SYNC = 0x3,                                 /**< Applicable for Topfield only. The TopField is decoded from HSync and VSync. */
} DLPC654X_SrcSyncConfigT_e;

typedef enum
{
    DLPC654X_EPT_BLANK = 0x0,                                         /**< Blank */
    DLPC654X_EPT_CHESS = 0x1,                                         /**< Chess */
    DLPC654X_EPT_GRID = 0x2,                                          /**< Grid */
    DLPC654X_EPT_HGRID = 0x3,                                         /**< Hgrid */
    DLPC654X_EPT_H_STRIPE = 0x4,                                      /**< H Stripe */
    DLPC654X_EPT_V_STRIPE = 0x5,                                      /**< V Stripe */
    DLPC654X_EPT_HLINES = 0x6,                                        /**< Hlines */
    DLPC654X_EPT_VLINES = 0x7,                                        /**< Vlines */
    DLPC654X_EPT_MOIRE = 0x8,                                         /**< Moire */
    DLPC654X_EPT_HRAMP_RED = 0x9,                                     /**< Hramp Red */
    DLPC654X_EPT_HRAMP_GREEN = 0xA,                                   /**< Hramp Green */
    DLPC654X_EPT_HRAMP_BLUE = 0xB,                                    /**< Hramp Blue */
    DLPC654X_EPT_HRAMP_WHITE = 0xC,                                   /**< Hramp White */
    DLPC654X_EPT_HRAMP_ALL = 0xD,                                     /**< Hramp All */
    DLPC654X_EPT_COLOR_BAR = 0xE,                                     /**< Color Bar */
    DLPC654X_EPT_XPR = 0xF,                                           /**< Xpr */
    DLPC654X_EPT_XPR_COLOR = 0x10,                                    /**< Xpr Color */
    DLPC654X_EPT_BLACK = 0x11,                                        /**< Black */
    DLPC654X_EPT_WHITE = 0x12,                                        /**< White */
    DLPC654X_EPT_CURRENT = 0x13,                                      /**< Current */
} DLPC654X_EmuPatternT_e;

typedef enum
{
    DLPC654X_PE_RESET = 0x0,                                          /**< Reset */
    DLPC654X_PE_STANDBY = 0x1,                                        /**< Standby */
    DLPC654X_PE_ACTIVE = 0x2,                                         /**< Active */
    DLPC654X_PE_COOLING = 0x3,                                        /**< Cooling */
    DLPC654X_PE_WARMING = 0x4,                                        /**< Warming */
} DLPC654X_PowerstateEnum_e;

typedef enum
{
    DLPC654X_GIST_SRC0 = 0x0,                                         /**< Src0 */
    DLPC654X_GIST_SRC1 = 0x1,                                         /**< Src1 */
    DLPC654X_GIST_SRC2 = 0x2,                                         /**< Src2 */
    DLPC654X_GIST_SRC3 = 0x3,                                         /**< Src3 */
} DLPC654X_GpioIntSourceT_e;

typedef enum
{
    DLPC654X_CCCT_SINGLE = 0x0,                                       /**< Single */
    DLPC654X_CCCT_MULTIPLE = 0x1,                                     /**< Multiple */
} DLPC654X_CmdControllerConfigT_e;

typedef enum
{
    DLPC654X_SFDIMT_ODDEVEN_ON_A = 0x0,                               /**< Single Port (A) FPD-link interface */
    DLPC654X_SFDIMT_ODDEVEN_ON_B = 0x1,                               /**< Single Port (B) FPD-link interface */
    DLPC654X_SFDIMT_PORT_A_EVEN_B_ODD = 0x2,                          /**< Dual-Port FPD (AB), if PortA carries Even and PortB carries Odd data */
    DLPC654X_SFDIMT_PORT_A_ODD_B_EVEN = 0x3,                          /**< Dual-Port FPD, if PortA carries Even and PortB carries Odd data */
    DLPC654X_SFDIMT_INVALID_DATA_INTERFACE = 0x4,                     /**< Invalid input type */
} DLPC654X_SrcFpdDataInterfaceModeT_e;

typedef enum
{
    DLPC654X_CAWT_UINT32 = 0x0,                                       /**< Uint32 */
    DLPC654X_CAWT_UINT16 = 0x1,                                       /**< Uint16 */
    DLPC654X_CAWT_UINT08 = 0x2,                                       /**< Uint08 */
} DLPC654X_CmdAccessWidthT_e;

typedef enum
{
    DLPC654X_SPWT_SRC30_BITS = 0x0,                                   /**< A total of 30 bits with 10-bits on each of R, G ,B or Y, Cb, Cr data on channels A, B and C. */
    DLPC654X_SPWT_SRC24_BITS = 0x1,                                   /**< A total of 24 bits with 8-bits on each of R, G, B or Y, Cb, Cr data on channels A, B and C. */
    DLPC654X_SPWT_SRC20_BITS = 0x2,                                   /**< A total of 20 bits with 10-bits each of Y, (CbCr) data on any A, B, or C channel with the unused input channel (Channel C) being a 'don't care'. */
    DLPC654X_SPWT_SRC16_BITS = 0x3,                                   /**< A total of 16 bits with 8-bits each of Y, (CbCr) data on any A, B, or C channel with the unused input channel (Channel C) being a 'don't care'. */
    DLPC654X_SPWT_EXPANDED_HDMI = 0x4,                                /**< A total of 20 bits with 10-bits each of Y, (CrCb) on channel A and B respectively, Aout = Ain(9:2) + Cin(5:4); Bout = Bin(9:2) + Cin(9:8). Cout = 0 */
    DLPC654X_SPWT_SRC10_BITS = 0x5,                                   /**< A total of 10 bits of formatted BT-656 type Y/CbCr data stream. */
    DLPC654X_SPWT_SRC08_BITS = 0x6,                                   /**< A total of 08 bits of formatted BT-656 type Y/CbCr data stream. */
    DLPC654X_SPWT_PORT_WIDTH_BASED_ON_MODE = 0x7,                     /**< Only used for VBO and FPD sources to select width based on mode */
} DLPC654X_SrcPortWidthT_e;

typedef enum
{
    DLPC654X_KE_NIL = 0x0,                                            /**< Nil */
    DLPC654X_KE_IR_HOME = 0xD,                                        /**< Ir Home */
    DLPC654X_KE_IR_VOLUME_INC = 0xE,                                  /**< Ir Volume Inc */
    DLPC654X_KE_IR_VOLUME_DEC = 0xF,                                  /**< Ir Volume Dec */
    DLPC654X_KE_IR_POWER = 0x10,                                      /**< Ir Power */
    DLPC654X_KE_IR_SOURCE = 0x11,                                     /**< Ir Source */
    DLPC654X_KE_IR_LEFT_ARROW = 0x12,                                 /**< Ir Left Arrow */
    DLPC654X_KE_IR_RIGHT_ARROW = 0x13,                                /**< Ir Right Arrow */
    DLPC654X_KE_IR_BLANK = 0x14,                                      /**< Ir Blank */
    DLPC654X_KE_IR_ZOOM_INC = 0x15,                                   /**< Ir Zoom Inc */
    DLPC654X_KE_IR_KEYSTONE_INC = 0x16,                               /**< Ir Keystone Inc */
    DLPC654X_KE_IR_BRIGHT_INC = 0x17,                                 /**< Ir Bright Inc */
    DLPC654X_KE_IR_ZOOM_DEC = 0x18,                                   /**< Ir Zoom Dec */
    DLPC654X_KE_IR_KEYSTONE_DEC = 0x19,                               /**< Ir Keystone Dec */
    DLPC654X_KE_IR_BRIGHT_DEC = 0x20,                                 /**< Ir Bright Dec */
    DLPC654X_KE_IR_EXIT = 0x21,                                       /**< Ir Exit */
    DLPC654X_KE_IR_MENU = 0x22,                                       /**< Ir Menu */
    DLPC654X_KE_IR_SELECT = 0x23,                                     /**< Ir Select */
    DLPC654X_KE_IR_LEFTCLICK = 0x24,                                  /**< Ir Leftclick */
    DLPC654X_KE_IR_RIGHTCLICK = 0x25,                                 /**< Ir Rightclick */
    DLPC654X_KE_IR_LEFTRIGHT = 0x26,                                  /**< Ir Leftright */
    DLPC654X_KE_IR_MM_UP = 0x27,                                      /**< Ir Mm Up */
    DLPC654X_KE_IR_MM_DOWN = 0x28,                                    /**< Ir Mm Down */
    DLPC654X_KE_IR_MM_LEFT = 0x29,                                    /**< Ir Mm Left */
    DLPC654X_KE_IR_MM_RIGHT = 0x2A,                                   /**< Ir Mm Right */
    DLPC654X_KE_IR_MM_LEFTCLICK = 0x2B,                               /**< Ir Mm Leftclick */
    DLPC654X_KE_IR_MM_RIGHTCLICK = 0x2C,                              /**< Ir Mm Rightclick */
    DLPC654X_KE_IR_MM_LEFTRIGHT = 0x2D,                               /**< Ir Mm Leftright */
    DLPC654X_KE_IR_NOBUTTON = 0x2E,                                   /**< Ir Nobutton */
    DLPC654X_KE_IR_KEYSTONE = 0x2F,                                   /**< Ir Keystone */
    DLPC654X_KE_IR_RESYNC = 0x30,                                     /**< Ir Resync */
    DLPC654X_KE_IR_LAST = 0x31,                                       /**< Ir Last */
    DLPC654X_KE_UART_U_LPM = 0x32,                                    /**< Uart U Lpm */
    DLPC654X_KE_PAD_POWER = 0x40,                                     /**< Pad Power */
    DLPC654X_KE_PAD_ZOOM = 0x41,                                      /**< Pad Zoom */
    DLPC654X_KE_PAD_BRIGHT = 0x42,                                    /**< Pad Bright */
    DLPC654X_KE_PAD_SOURCE = 0x43,                                    /**< Pad Source */
    DLPC654X_KE_PAD_KEYSTONE = 0x44,                                  /**< Pad Keystone */
    DLPC654X_KE_PAD_MENU = 0x45,                                      /**< Pad Menu */
    DLPC654X_KE_PAD_LEFTRIGHT = 0x46,                                 /**< Pad Leftright */
    DLPC654X_KE_PAD_UPDOWN = 0x47,                                    /**< Pad Updown */
    DLPC654X_KE_PAD_MM_LEFT = 0x48,                                   /**< Pad Mm Left */
    DLPC654X_KE_PAD_MM_RIGHT = 0x49,                                  /**< Pad Mm Right */
    DLPC654X_KE_PAD_MM_UP = 0x4A,                                     /**< Pad Mm Up */
    DLPC654X_KE_PAD_MM_DOWN = 0x4B,                                   /**< Pad Mm Down */
    DLPC654X_KE_PAD_MM_SELECT = 0x4C,                                 /**< Pad Mm Select */
    DLPC654X_KE_PAD_LAST = 0x4D,                                      /**< Pad Last */
} DLPC654X_KcodeEnum_e;

typedef enum
{
    DLPC654X_DART_FILL = 0x0,                                         /**< Fill (uses DMD aspect ratio) */
    DLPC654X_DART_NATIVE = 0x1,                                       /**< Native */
    DLPC654X_DART_ASPECT43 = 0x2,                                     /**< 4:3 */
    DLPC654X_DART_ASPECT169 = 0x3,                                    /**< 16:9 */
    DLPC654X_DART_ANAMORPHIC = 0x4,                                   /**< Anamorphic */
} DLPC654X_DispfmtAspectRatioT_e;

typedef enum
{
    DLPC654X_AASCE_SMT1_SMT2_ALF = 0x0,                               /**< Smt1 Smt2 Alf */
    DLPC654X_AASCE_SMT2_ALF = 0x1,                                    /**< Smt2 Alf */
    DLPC654X_AASCE_ALF = 0x2,                                         /**< Always run Autolock. Never restore from Saved Mode Table. */
    DLPC654X_AASCE_SMT1_SMT2 = 0x3,                                   /**< Always restore from Saved Mode Table. If no mathcing entries are found, then run Autolock. */
    DLPC654X_AASCE_ALF_SMT1_SMT2 = 0x4,                               /**< Run Autolock first. On subsequent resyncs, restore from Saved Mode Table. */
    DLPC654X_AASCE_SMT1_ALF_SMT2 = 0x5,                               /**< Smt1 Alf Smt2 */
} DLPC654X_AlcAutolockSmtCfgEnum_e;

typedef enum
{
    DLPC654X_P_OUT0 = 0x0,                                            /**< Output PWM 0 */
    DLPC654X_P_OUT1 = 0x1,                                            /**< Output PWM 1 */
    DLPC654X_P_OUT2 = 0x2,                                            /**< Output PWM 2 */
    DLPC654X_P_LEDR = 0x3,                                            /**< Output PWM 3 */
    DLPC654X_P_LEDG = 0x4,                                            /**< Output PWM 4 */
    DLPC654X_P_LEDB = 0x5,                                            /**< Output PWM 5 */
    DLPC654X_P_LEDIR = 0x6,                                           /**< Output PWM 6 */
    DLPC654X_P_LEDUV = 0x7,                                           /**< Output PWM 7 */
    DLPC654X_P_CW0 = 0x8,                                             /**< Color Wheel 0 PWM */
    DLPC654X_P_CW1 = 0x9,                                             /**< Color Wheel 1 PWM */
    DLPC654X_P_CW2 = 0xA,                                             /**< Color Wheel 2 PWM */
    DLPC654X_P_IN0 = 0xB,                                             /**< Input  PWM 0 */
    DLPC654X_P_IN1 = 0xC,                                             /**< Input  PWM 1 */
    DLPC654X_P_NOT_USED = 0xD,                                        /**< Used for setting the DynamicBlack PWM port when DB PWM is not used */
    DLPC654X_P_MAX_PORT = 0xE,                                        /**< Invalid PWM Port */
} DLPC654X_Pwmport_e;

typedef enum
{
    DLPC654X_SFDMMT_SRC_FPD30_BITMAP_MODE0 = 0x0,                     /**< 30-bit Mode 0 */
    DLPC654X_SFDMMT_SRC_FPD30_BITMAP_MODE1 = 0x1,                     /**< 30-bit Mode 1 */
    DLPC654X_SFDMMT_SRC_FPD30_BITMAP_MODE2 = 0x2,                     /**< 30-bit Mode 2 */
    DLPC654X_SFDMMT_SRC_FPD24_BITMAP_MODE0 = 0x3,                     /**< 24-bit Mode 0 */
    DLPC654X_SFDMMT_SRC_FPD24_BITMAP_MODE1 = 0x4,                     /**< 24-bit Mode 1 */
    DLPC654X_SFDMMT_INVALID_MODE = 0x5,                               /**< Not a valid FPD-Link data mode or mode is not used */
} DLPC654X_SrcFpdDataMapModeT_e;

typedef enum
{
    DLPC654X_SDORT_OE_ODDISLEFT = 0x0,                                /**< Odd field is Left Eye. */
    DLPC654X_SDORT_OE_ODDISRIGHT = 0x1,                               /**< Odd field is Right Eye. */
    DLPC654X_SDORT_OE_NO_REFERENCE = 0x2,                             /**< No Odd/Even reference is available. */
} DLPC654X_Src3DOeReferenceT_e;

typedef enum
{
    DLPC654X_HTFT_TRAD_GAM_SDR = 0x0,                                 /**< Trad Gam Sdr */
    DLPC654X_HTFT_TRAD_GAM_HDR = 0x1,                                 /**< Trad Gam Hdr */
    DLPC654X_HTFT_PQ = 0x2,                                           /**< Pq */
    DLPC654X_HTFT_HLG = 0x3,                                          /**< Hlg */
} DLPC654X_HdrTransferFnT_e;

typedef enum
{
    DLPC654X_CPM_DISP_EXTERNAL = 0x0,                                 /**< Disp External */
    DLPC654X_CPM_TEST_PATTERN = 0x1,                                  /**< Test Pattern */
    DLPC654X_CPM_SOLID_FIELD = 0x2,                                   /**< Solid Field */
    DLPC654X_CPM_SPLASH = 0x3,                                        /**< Splash */
    DLPC654X_CPM_CURTAIN = 0x4,                                       /**< Curtain */
} DLPC654X_CmdProjectionModes_e;

typedef enum
{
    DLPC654X_AACE_STOP = 0x0,                                         /**< Stop */
    DLPC654X_AACE_START = 0x1,                                        /**< Start */
} DLPC654X_AlcAlgorithmCtlEnum_e;

typedef enum
{
    DLPC654X_DPWT_DB_WEIGHT0 = 0x0,                                   /**< Weighted 0% */
    DLPC654X_DPWT_DB_WEIGHT25 = 0x1,                                  /**< Weighted 25% */
    DLPC654X_DPWT_DB_WEIGHT50 = 0x2,                                  /**< Weighted 50% */
    DLPC654X_DPWT_DB_WEIGHT75 = 0x3,                                  /**< Weighted 75% */
} DLPC654X_DbPixelWeightT_e;

typedef enum
{
    DLPC654X_CFUT_LOCK = 0x0,                                         /**< Lock */
    DLPC654X_CFUT_UNLOCK = 0xF7A54027,                                /**< Unlock */
} DLPC654X_CmdFlashUpdateT_e;

typedef enum
{
    DLPC654X_UFCT_OFF = 0x0,                                          /**< Off */
    DLPC654X_UFCT_HW = 0x1,                                           /**< Hardware flow control */
} DLPC654X_UrtFlowControlT_e;

typedef enum
{
    DLPC654X_SAMT_STRAIGHT_THRU = 0x0,                                /**< ABC -> ABC (1-1 mapping) <BR> */
    DLPC654X_SAMT_ROTATE_RIGHT = 0x1,                                 /**< ABC -> CAB               <BR> */
    DLPC654X_SAMT_ROTATE_LEFT = 0x2,                                  /**< ABC -> BCA               <BR> */
    DLPC654X_SAMT_SWAP_BC = 0x3,                                      /**< ABC -> ACB               <BR> */
    DLPC654X_SAMT_SWAP_AB = 0x4,                                      /**< ABC -> BAC               <BR> */
    DLPC654X_SAMT_SWAP_AC = 0x5,                                      /**< ABC -> CBA               <BR> */
} DLPC654X_SrcAbcMuxT_e;

typedef enum
{
    DLPC654X_ASPE_NEGATIVE_POLARITY = 0x0,                            /**< Negative Polarity */
    DLPC654X_ASPE_POSITIVE_POLARITY = 0x1,                            /**< Positive Polarity */
} DLPC654X_AlcSignalPolarityEnum_e;

typedef enum
{
    DLPC654X_DSS_DETECT_STABLE_VIDEO = 0x0,                           /**< Detect Stable Video */
    DLPC654X_DSS_SEARCHING = 0x1,                                     /**< Searching */
    DLPC654X_DSS_SYNCDETECTED = 0x2,                                  /**< Syncdetected */
    DLPC654X_DSS_LOCKED = 0x3,                                        /**< Locked */
    DLPC654X_DSS_SUSPENDED = 0x4,                                     /**< Suspended */
} DLPC654X_DpScanStatus_e;

typedef enum
{
    DLPC654X_SDSCT_NO_DOWNSAMPLE = 0x0,                               /**< Down Sample Operation disabled (data pass through unmodified). */
    DLPC654X_SDSCT_DOWNSAMPLE_FIRST_PXL = 0x1,                        /**< Down Sample Operation enabled. Select First Data Sample Positions from Sample Position Reference. */
    DLPC654X_SDSCT_DOWNSAMPLE_SECOND_PXL = 0x2,                       /**< Down Sample Operation enabled. Select Second Data Sample Positions from Sample Position Reference. */
} DLPC654X_SrcDownSampleConfigT_e;

typedef enum
{
    DLPC654X_UDBT_URT_DATBITS5 = 0x0,                                 /**< 5 */
    DLPC654X_UDBT_URT_DATBITS6 = 0x1,                                 /**< 6 */
    DLPC654X_UDBT_URT_DATBITS7 = 0x2,                                 /**< 7 */
    DLPC654X_UDBT_URT_DATBITS8 = 0x3,                                 /**< 8 */
} DLPC654X_UrtDataBitsT_e;

typedef enum
{
    DLPC654X_USBT_URT_STPBITS1 = 0x0,                                 /**< 1 */
    DLPC654X_USBT_URT_STPBITS2 = 0x1,                                 /**< 2 */
} DLPC654X_UrtStopBitsT_e;

typedef enum
{
    DLPC654X_UBRT_URT_BAUD1200 = 0x0,                                 /**< 1200 */
    DLPC654X_UBRT_URT_BAUD2400 = 0x1,                                 /**< 2400 */
    DLPC654X_UBRT_URT_BAUD4800 = 0x2,                                 /**< 4800 */
    DLPC654X_UBRT_URT_BAUD9600 = 0x3,                                 /**< 9600 */
    DLPC654X_UBRT_URT_BAUD14400 = 0x4,                                /**< 14400 */
    DLPC654X_UBRT_URT_BAUD19200 = 0x5,                                /**< 19200 */
    DLPC654X_UBRT_URT_BAUD38400 = 0x6,                                /**< 38400 */
    DLPC654X_UBRT_URT_BAUD57600 = 0x7,                                /**< 57600 */
    DLPC654X_UBRT_URT_BAUD115200 = 0x8,                               /**< 115200 */
    DLPC654X_UBRT_URT_BAUD230400 = 0x9,                               /**< 230400 */
    DLPC654X_UBRT_URT_BAUD460800 = 0xA,                               /**< 460800 */
    DLPC654X_UBRT_URT_BAUD921600 = 0xB,                               /**< 921600 */
} DLPC654X_UrtBaudRateT_e;

typedef enum
{
    DLPC654X_EPST_NO_SHIFT = 0x0,                                     /**< No Shift */
    DLPC654X_EPST_H_SHIFT = 0x1,                                      /**< H Shift */
    DLPC654X_EPST_V_SHIFT = 0x2,                                      /**< V Shift */
    DLPC654X_EPST_HV_SWITCH = 0x3,                                    /**< Hv Switch */
} DLPC654X_EmuPatternShiftT_e;

typedef enum
{
    DLPC654X_TST_APP0 = 0x0,                                          /**< Generic application software code block */
    DLPC654X_TST_APP1 = 0x1,                                          /**< Generic application software code block */
    DLPC654X_TST_APP2 = 0x2,                                          /**< Generic application software code block */
    DLPC654X_TST_APP3 = 0x3,                                          /**< Generic application software code block */
    DLPC654X_TST_APP4 = 0x4,                                          /**< Generic application software code block */
    DLPC654X_TST_APP5 = 0x5,                                          /**< Generic application software code block */
    DLPC654X_TST_APP6 = 0x6,                                          /**< Generic application software code block */
    DLPC654X_TST_APP7 = 0x7,                                          /**< Generic application software code block */
    DLPC654X_TST_FRAME_TASK = 0x8,                                    /**< Frame Task */
    DLPC654X_TST_CW_INDEX_TASK = 0x9,                                 /**< Color wheel index task */
    DLPC654X_TST_SEQ_TASK = 0xA,                                      /**< Sequence Interrupt Handler */
    DLPC654X_TST_PMD_TASK = 0xB,                                      /**< PMD Interrupt Handler */
    DLPC654X_TST_APP12 = 0xC,                                         /**< Generic application software code block */
    DLPC654X_TST_APP13 = 0xD,                                         /**< Generic application software code block */
    DLPC654X_TST_APP14 = 0xE,                                         /**< Generic application software code block */
    DLPC654X_TST_APP15 = 0xF,                                         /**< Generic application software code block */
    DLPC654X_TST_MTR_SPIN_IRQ = 0x10,                                 /**< Motor spin interrupt handler */
    DLPC654X_TST_CW_PHASE_LOCK = 0x11,                                /**< Color wheel phase lock interrupt handler */
    DLPC654X_TST_CW_FREQ_LOCK = 0x12,                                 /**< Color wheel frequency lock */
    DLPC654X_TST_SEQ_FREQ_LOCK = 0x13,                                /**< Sequence frequency lock */
    DLPC654X_TST_SEQ_PHASE_LOCK = 0x14,                               /**< Sequence phase lock */
    DLPC654X_TST_APP21 = 0x15,                                        /**< Generic application software code block */
    DLPC654X_TST_APP22 = 0x16,                                        /**< Generic application software code block */
    DLPC654X_TST_APP23 = 0x17,                                        /**< Generic application software code block */
    DLPC654X_TST_OSD_IRQ = 0x18,                                      /**< OSD menu interrupt handler */
    DLPC654X_TST_USB_IRQ = 0x19,                                      /**< USB interrupt handler */
    DLPC654X_TST_APP26 = 0x1A,                                        /**< Generic application software code block */
    DLPC654X_TST_APP27 = 0x1B,                                        /**< Generic application software code block */
    DLPC654X_TST_APP28 = 0x1C,                                        /**< Generic application software code block */
    DLPC654X_TST_APP29 = 0x1D,                                        /**< Generic application software code block */
    DLPC654X_TST_APP30 = 0x1E,                                        /**< Generic application software code block */
    DLPC654X_TST_APP31 = 0x1F,                                        /**< Generic application software code block */
    DLPC654X_TST_FRONT_END_BLANKING_TASK = 0x20,                      /**< Front End Blanking Task. */
    DLPC654X_TST_ALF_TASK = 0x21,                                     /**< ALF Task */
    DLPC654X_TST_TPM3_D_DETECT_TASK = 0x22,                           /**< 3D Detection Task */
    DLPC654X_TST_BRS_TASK = 0x23,                                     /**< Generic application software code block */
    DLPC654X_TST_APP36 = 0x24,                                        /**< Generic application software code block */
    DLPC654X_TST_APP37 = 0x25,                                        /**< Generic application software code block */
    DLPC654X_TST_APP38 = 0x26,                                        /**< Generic application software code block */
    DLPC654X_TST_APP39 = 0x27,                                        /**< Generic application software code block */
    DLPC654X_TST_APP40 = 0x28,                                        /**< Generic application software code block */
    DLPC654X_TST_APP41 = 0x29,                                        /**< Generic application software code block */
    DLPC654X_TST_APP42 = 0x2A,                                        /**< Generic application software code block */
    DLPC654X_TST_APP43 = 0x2B,                                        /**< Generic application software code block */
    DLPC654X_TST_APP44 = 0x2C,                                        /**< Generic application software code block */
    DLPC654X_TST_APP45 = 0x2D,                                        /**< Generic application software code block */
    DLPC654X_TST_APP46 = 0x2E,                                        /**< Generic application software code block */
    DLPC654X_TST_APP47 = 0x2F,                                        /**< Generic application software code block */
    DLPC654X_TST_APP48 = 0x30,                                        /**< Generic application software code block */
    DLPC654X_TST_APP49 = 0x31,                                        /**< Generic application software code block */
    DLPC654X_TST_APP50 = 0x32,                                        /**< Generic application software code block */
    DLPC654X_TST_APP51 = 0x33,                                        /**< Generic application software code block */
    DLPC654X_TST_APP52 = 0x34,                                        /**< Generic application software code block */
    DLPC654X_TST_APP53 = 0x35,                                        /**< Generic application software code block */
    DLPC654X_TST_APP54 = 0x36,                                        /**< Generic application software code block */
    DLPC654X_TST_APP55 = 0x37,                                        /**< Generic application software code block */
    DLPC654X_TST_APP56 = 0x38,                                        /**< Generic application software code block */
    DLPC654X_TST_APP57 = 0x39,                                        /**< Generic application software code block */
    DLPC654X_TST_APP58 = 0x3A,                                        /**< Generic application software code block */
    DLPC654X_TST_APP59 = 0x3B,                                        /**< Generic application software code block */
    DLPC654X_TST_PB_TASK = 0x3C,                                      /**< Generic application software code block */
    DLPC654X_TST_PB_IRQ = 0x3D,                                       /**< Generic application software code block */
    DLPC654X_TST_PB_DMD_COORD_CONV = 0x3E,                            /**< Generic application software code block */
    DLPC654X_TST_SSP_IRQ = 0x3F,                                      /**< SSP IRQ for point blank */
    DLPC654X_TST_RTP_DBGM_GPSWFLG0 = 0x40,                            /**< Non-reset Software General Purpose Flag0 */
    DLPC654X_TST_RTP_DBGM_GPSWFLG1 = 0x41,                            /**< Non-reset Software General Purpose Flag1 */
    DLPC654X_TST_RTP_DBGM_GPSWFLG2 = 0x42,                            /**< Non-reset Software General Purpose Flag2 */
    DLPC654X_TST_RTP_DBGM_GPSWFLG3 = 0x43,                            /**< Non-reset Software General Purpose Flag3 */
    DLPC654X_TST_RTP_DBGM_GPSWFLG4 = 0x44,                            /**< Non-reset Software General Purpose Flag4 */
    DLPC654X_TST_RTP_DBGM_GPSWFLG5 = 0x45,                            /**< Non-reset Software General Purpose Flag5 */
    DLPC654X_TST_RTP_DBGM_GPSWFLG6 = 0x46,                            /**< Non-reset Software General Purpose Flag6 */
    DLPC654X_TST_RTP_DBGM_GPSWFLG7 = 0x47,                            /**< Non-reset Software General Purpose Flag7 */
    DLPC654X_TST_APPCB_MAXIMUM = 0x48,                                /**< Appcb Maximum */
} DLPC654X_TpmSbType_e;

typedef enum
{
    DLPC654X_DSFT_VIDEO = 0x0,                                        /**< Video */
    DLPC654X_DSFT_GRAPHICS = 0x1,                                     /**< Graphics */
} DLPC654X_DispScalingFilterT_e;

typedef enum
{
    DLPC654X_VSTT_VFM_SRC128072060 = 0x0,                             /**< 1280 720 60 */
    DLPC654X_VSTT_VFM_SRC128072050 = 0x1,                             /**< 1280 720 50 */
    DLPC654X_VSTT_VFM_SRC128072030 = 0x2,                             /**< 1280 720 30 */
    DLPC654X_VSTT_VFM_SRC128051260 = 0x3,                             /**< 1280 512 60 */
    DLPC654X_VSTT_NONE = 0xFF,                                        /**< None */
} DLPC654X_VfmSourceTypeT_e;

typedef enum
{
    DLPC654X_SDCT_CLR_RED = 0x0,                                      /**< ChannelA=0, ChannelB=1023, ChannelC=0 for RGB sources. YUV sources will be converted. */
    DLPC654X_SDCT_CLR_GREEN = 0x1,                                    /**< ChannelA=1023, ChannelB=0, ChannelC=0 for RGB sources.  YUV sources will be converted. */
    DLPC654X_SDCT_CLR_BLUE = 0x2,                                     /**< ChannelA=0, ChannelB=0, ChannelC=1023 for RGB sources. YUV sources will be converted. */
    DLPC654X_SDCT_CLR_CYAN = 0x3,                                     /**< ChannelA=1023, ChannelB=0, ChannelC=1023 for RGB sources. YUV sources will be converted. */
    DLPC654X_SDCT_CLR_MAGENTA = 0x4,                                  /**< ChannelA=0, ChannelB=1023, ChannelC=1023 for RGB sources YUV sources will be converted. */
    DLPC654X_SDCT_CLR_YELLOW = 0x5,                                   /**< ChannelA=1023, ChannelB=1023, ChannelC=0 for RGB sources. YUV sources will be converted. */
    DLPC654X_SDCT_CLR_WHITE = 0x6,                                    /**< ChannelA=1023, ChannelB=1023, ChannelC=1023 for RGB sources. YUV sources will be converted. */
    DLPC654X_SDCT_CLR_BLACK = 0x7,                                    /**< ChannelA=0, ChannelB=0, ChannelC=0 for RGB sources. YUV sources will be converted. */
    DLPC654X_SDCT_CLR75_BLUE25_BLACK = 0x8,                           /**< 75% of the line is Blue, 25% is black */
    DLPC654X_SDCT_CLR25_BLUE75_BLACK = 0x9,                           /**< 25% of the line is Blue, 75% is black */
} DLPC654X_Src3DColorT_e;

typedef enum
{
    DLPC654X_SPFT_RGB = 0x0,                                          /**< Rgb */
    DLPC654X_SPFT_YUV444 = 0x1,                                       /**< Yuv444 */
    DLPC654X_SPFT_YUV422 = 0x2,                                       /**< Yuv422 */
    DLPC654X_SPFT_YUV420 = 0x3,                                       /**< Yuv420 */
} DLPC654X_SrcPixelFormatT_e;

typedef enum
{
    DLPC654X_AOME_MODE = 0x0,                                         /**< Mode */
    DLPC654X_AOME_NO_GAIN_IN_MONITOR = 0x1,                           /**< No Gain In Monitor */
    DLPC654X_AOME_CALIBRATED_GAIN = 0x2,                              /**< Calibrated Gain */
    DLPC654X_AOME_CALIBRATED_GAIN_ADJ = 0x3,                          /**< Calibrated Gain Adj */
    DLPC654X_AOME_OFFSET = 0x4,                                       /**< Offset */
} DLPC654X_AlcOperatingModeEnum_e;

typedef enum
{
    DLPC654X_DTT_CW_INDEX = 0x0,                                      /**< Cw Index */
    DLPC654X_DTT_I2_CSLAVE = 0x1,                                     /**< I2cslave */
    DLPC654X_DTT_OSD = 0x2,                                           /**< Osd */
    DLPC654X_DTT_SEQ = 0x3,                                           /**< Seq */
    DLPC654X_DTT_SPLASH = 0x4,                                        /**< Splash */
    DLPC654X_DTT_ALC = 0x5,                                           /**< Alc */
    DLPC654X_DTT_SEQ_LUT = 0x6,                                       /**< Seq Lut */
    DLPC654X_DTT_IVD_COMM = 0x7,                                      /**< Ivd Comm */
    DLPC654X_DTT_SRC3_D_DETECTION = 0x8,                              /**< Src 3d Detection */
    DLPC654X_DTT_WPC = 0x9,                                           /**< Wpc */
    DLPC654X_DTT_INTERACTIVEIO = 0xA,                                 /**< Interactiveio */
    DLPC654X_DTT_FRAME = 0xB,                                         /**< Frame */
    DLPC654X_DTT_FRAME_FRONT_END_BLANKING = 0xC,                      /**< Frame Front End Blanking */
    DLPC654X_DTT_FRAME_BACK_END_VSYNC = 0xD,                          /**< Frame Back End Vsync */
} DLPC654X_DdpTaskT_e;

typedef enum
{
    DLPC654X_THRST_TPG_HORZRAMP_STEP0_P5 = 0x1,                       /**< Gray value Increment by 2 every 4 pixel */
    DLPC654X_THRST_TPG_HORZRAMP_STEP1_P0 = 0x2,                       /**< Gray value Increment by 1 every pixel */
    DLPC654X_THRST_TPG_HORZRAMP_STEP2_P0 = 0x3,                       /**< Gray value Increment by 2 every pixel */
    DLPC654X_THRST_TPG_HORZRAMP_STEP0_P25 = 0x5,                      /**< Gray value Increment by 1 every 4 pixel */
} DLPC654X_TpgHorzRampStepT_e;

typedef enum
{
    DLPC654X_PI_INCOUNT0 = 0x0,                                       /**< PWM Input Counter 0 */
    DLPC654X_PI_INCOUNT1 = 0x1,                                       /**< PWM Input Counter 1 */
} DLPC654X_PwmIncounter_e;

typedef enum
{
    DLPC654X_AALPCE_ADC_OUTPUT = 0x0,                                 /**< Adc Output */
    DLPC654X_AALPCE_DIG_OUTPUT = 0x1,                                 /**< Dig Output */
    DLPC654X_AALPCE_MONITOR = 0x2,                                    /**< Monitor */
    DLPC654X_AALPCE_DISABLE = 0x3,                                    /**< Disable */
} DLPC654X_AlcAutoLockPortCmdEnum_e;

typedef enum
{
    DLPC654X_TVRST_TPG_VERTRAMP_STEP1_P0 = 0x1,                       /**< Gray value Increment by 1 every line */
    DLPC654X_TVRST_TPG_VERTRAMP_STEP2_P0 = 0x2,                       /**< Gray value Increment by 2 every line */
    DLPC654X_TVRST_TPG_VERTRAMP_STEP4_P0 = 0x3,                       /**< Gray value Increment by 4 every line */
} DLPC654X_TpgVertRampStepT_e;

typedef enum
{
    DLPC654X_ASTE_VERT_AND_HORIZ_SYNC = 0x0,                          /**< Vert And Horiz Sync */
    DLPC654X_ASTE_SYNC_ON_GREEN = 0x1,                                /**< Sync On Green */
    DLPC654X_ASTE_ILLEGAL_SYNC = 0x2,                                 /**< Illegal Sync */
    DLPC654X_ASTE_COMPOSITE_SYNC = 0x3,                               /**< Composite Sync */
    DLPC654X_ASTE_NO_SYNC_SIGNALS = 0x4,                              /**< No Sync Signals */
    DLPC654X_ASTE_NO_VSYNC_SIGNALS = 0x5,                             /**< No Vsync Signals */
    DLPC654X_ASTE_NO_HSYNC_SIGNALS = 0x6,                             /**< No Hsync Signals */
} DLPC654X_AlcSyncTypeEnum_e;

typedef enum
{
    DLPC654X_SDTRT_TB_TOPISLEFT = 0x0,                                /**< Top is Left Eye. */
    DLPC654X_SDTRT_TB_TOPISRIGHT = 0x1,                               /**< Top is Right Eye. */
    DLPC654X_SDTRT_TB_NO_REFERENCE = 0x2,                             /**< No Top/Bottom reference is available. */
} DLPC654X_Src3DTbReferenceT_e;

typedef enum
{
    DLPC654X_AALPE_ALC_PORT1 = 0x0,                                   /**< 1 */
    DLPC654X_AALPE_ALC_PORT2 = 0x1,                                   /**< 2 */
    DLPC654X_AALPE_ALC_PORT3 = 0x2,                                   /**< 3 */
} DLPC654X_AlcAutoLockPortEnum_e;

typedef enum
{
    DLPC654X_WLCT_RED = 0x0,                                          /**< Red */
    DLPC654X_WLCT_GREEN = 0x1,                                        /**< Green */
    DLPC654X_WLCT_BLUE = 0x2,                                         /**< Blue */
} DLPC654X_WpcLedColorT_e;

typedef enum
{
    DLPC654X_AASE_STOPPED = 0x0,                                      /**< AutoLock function has never been started or has been commanded to stop. */
    DLPC654X_AASE_PROCESSING = 0x1,                                   /**< AutoLock function is currently processing. */
    DLPC654X_AASE_AUTOLOCKED = 0x2,                                   /**< AutoLock function has locked to the source. */
    DLPC654X_AASE_LOCK_FAILED = 0x3,                                  /**< AutoLock function is unable to lock. */
} DLPC654X_AlcAlgorithmStatusEnum_e;

typedef enum
{
    DLPC654X_DIST_OPTIMIZED0 = 0x0,                                   /**< Full intra-field interpolation */
    DLPC654X_DIST_OPTIMIZED1 = 0x1,                                   /**< Optimized 1 */
    DLPC654X_DIST_OPTIMIZED2 = 0x2,                                   /**< Optimized 2 */
    DLPC654X_DIST_OPTIMIZED3 = 0x3,                                   /**< Default setting */
    DLPC654X_DIST_OPTIMIZED4 = 0x4,                                   /**< Optimized 4 */
    DLPC654X_DIST_OPTIMIZED5 = 0x5,                                   /**< Optimized 5 */
    DLPC654X_DIST_OPTIMIZED6 = 0x6,                                   /**< Optimized 6 */
    DLPC654X_DIST_OPTIMIZED7 = 0x7,                                   /**< Full field jam */
    DLPC654X_DIST_USER_DEFINED = 0x8,                                 /**< User Defined */
} DLPC654X_DeiInterpolationStrengthT_e;

typedef enum
{
    DLPC654X_SPT_VBO = 0x0,                                           /**< V-By-One Port */
    DLPC654X_SPT_FPD = 0x1,                                           /**< FPD Link on Port */
    DLPC654X_SPT_PARALLEL = 0x2,                                      /**< Parallel Video on Port A, B and C */
    DLPC654X_SPT_BT656 = 0x3,                                         /**< BT656 Port using GPIO 77 through GPIO 87 */
    DLPC654X_SPT_TPG = 0x4,                                           /**< Port for TPG */
} DLPC654X_SrcPortT_e;

typedef enum
{
    DLPC654X_SVDMMT_MODE0 = 0x0,                                      /**< 36bpp/30bpp RGB/YCbCr444 */
    DLPC654X_SVDMMT_MODE1 = 0x1,                                      /**< 27bpp RGB/YCbCr444 */
    DLPC654X_SVDMMT_MODE2 = 0x2,                                      /**< 24bpp RGB/YCbCr444 */
    DLPC654X_SVDMMT_MODE3 = 0x3,                                      /**< 32bpp/24bpp/20bpp YCbCr422 */
    DLPC654X_SVDMMT_MODE4 = 0x4,                                      /**< 18bpp YCbCr422 */
    DLPC654X_SVDMMT_MODE5 = 0x5,                                      /**< 16bpp YCbCr422 */
    DLPC654X_SVDMMT_MODE6 = 0x6,                                      /**< 12bpp/10bpp YCbCr420 Config 1 */
    DLPC654X_SVDMMT_MODE7 = 0x7,                                      /**< 8bpp YCbCr420 Config 1 */
    DLPC654X_SVDMMT_MODE8 = 0x8,                                      /**< 10bpp YCbCr420 Config 2 */
    DLPC654X_SVDMMT_MODE9 = 0x9,                                      /**< 8bpp YCbCr420 Config 2 */
    DLPC654X_SVDMMT_MODE_NONE = 0xA,                                  /**< Not a valid V-by-one data mode or mode is not used */
} DLPC654X_SrcVboDataMapModeT_e;

typedef enum
{
    DLPC654X_GTT_LEVEL = 0x0,                                         /**< Level Trigger */
    DLPC654X_GTT_EDGE = 0x1,                                          /**< Edge Trigger */
} DLPC654X_GpioTriggerT_e;

typedef enum
{
    DLPC654X_ICGT_IMG_CTI_GAIN1 = 0x0,                                /**< Gain = 1 */
    DLPC654X_ICGT_IMG_CTI_GAIN2 = 0x1,                                /**< Gain = 2 */
    DLPC654X_ICGT_IMG_CTI_GAIN4 = 0x2,                                /**< Gain = 4 */
    DLPC654X_ICGT_IMG_CTI_GAIN8 = 0x3,                                /**< Gain = 8 */
} DLPC654X_ImgCtiGainT_e;

typedef enum
{
    DLPC654X_AALEE_SOURCE_CHANGE_DETECTED = 0x0,                      /**< Generated when new sync signals are detected. */
    DLPC654X_AALEE_SOURCE_DETECTED_STABLE = 0x1,                      /**< Generated when sync signals are considered stable. */
    DLPC654X_AALEE_SOURCE_NO_SYNCS = 0x2,                             /**< Generated when no syncs are detected. */
    DLPC654X_AALEE_SOURCE_NO_VSYNCS = 0x3,                            /**< Generated when no VSYNC is detected. */
    DLPC654X_AALEE_SOURCE_NO_HSYNCS = 0x4,                            /**< Generated when no HSYNC is detected. */
    DLPC654X_AALEE_ASM_LOCKED = 0x5,                                  /**< Generated when the algorithm declares the source locked. */
    DLPC654X_AALEE_ASM_SYNCS_LOST = 0x6,                              /**< Asm Syncs Lost */
    DLPC654X_AALEE_ASM_REACQUIRE = 0x7,                               /**< Asm Reacquire */
    DLPC654X_AALEE_ASM_FAILED = 0x8,                                  /**< Asm Failed */
    DLPC654X_AALEE_DSM_LOCKED = 0x9,                                  /**< Generated when the algorithm declares the source locked. */
    DLPC654X_AALEE_DSM_SYNCS_LOST = 0xA,                              /**< Dsm Syncs Lost */
    DLPC654X_AALEE_DSM_REACQUIRE = 0xB,                               /**< Dsm Reacquire */
    DLPC654X_AALEE_DSM_FAILED = 0xC,                                  /**< Dsm Failed */
    DLPC654X_AALEE_I2_C_DRIVER_TERMINAL_ERROR = 0xD,                  /**< I2c Driver Terminal Error */
} DLPC654X_AlcAutoLockEventEnum_e;

typedef enum
{
    DLPC654X_SDFT_VSYNC_SEPARATED_HALF = 0x0,                         /**< VSync separated (field sequential) format. */
    DLPC654X_SDFT_VSYNC_SEPARATED_FULL = 0x1,                         /**< VSync separated (frame sequential progressive) format. */
    DLPC654X_SDFT_VERT_PACKED_HALF = 0x2,                             /**< Over Under (vertically packed) half resolution format. */
    DLPC654X_SDFT_VERT_PACKED_FULL = 0x3,                             /**< Over Under (vertically packed) full resolution format. */
    DLPC654X_SDFT_HORIZ_PACKED_HALF = 0x4,                            /**< Side by Side (horizontally packed) half resolution format. */
    DLPC654X_SDFT_HORIZ_PACKED_FULL = 0x5,                            /**< Side by Side (horizontally packed) full resolution format. */
} DLPC654X_Src3DFormatT_e;

typedef enum
{
    DLPC654X_SVBMT_SRC_VBO3_BYTES_MODE = 0x1,                         /**< 8bit mode (=3Byte mode) */
    DLPC654X_SVBMT_SRC_VBO4_BYTES_MODE = 0x2,                         /**< 10bit mode (=4Byte mode) */
    DLPC654X_SVBMT_SRC_VBO5_BYTES_MODE = 0x3,                         /**< 12bit mode (=5Byte mode) */
} DLPC654X_SrcVboByteModeT_e;

typedef enum
{
    DLPC654X_SDFDT_LEFT_EYE = 0x0,                                    /**< VSync separated sources only, Left Eye is 1st frame in 3D image pair. */
    DLPC654X_SDFDT_RIGHT_EYE = 0x1,                                   /**< VSync separated sources only, Right Eye is 1st frame in 3D image pair. */
} DLPC654X_Src3DFrameDominanceT_e;

typedef enum
{
    DLPC654X_SDCT_RED = 0x0,                                          /**< SSI SPI/PWM designator for RED */
    DLPC654X_SDCT_GREEN = 0x1,                                        /**< SSI SPI/PWM designator for GREEN */
    DLPC654X_SDCT_BLUE = 0x2,                                         /**< SSI SPI/PWM designator for BLUE */
    DLPC654X_SDCT_SAMPLE1_C1 = 0x3,                                   /**< Sample1 for SPI systems. */
    DLPC654X_SDCT_SAMPLE2_C2 = 0x4,                                   /**< Sample2 C2 */
    DLPC654X_SDCT_YELLOW_IR = 0x5,                                    /**< SSI SPI Yellow/ IR for PWM systems. */
    DLPC654X_SDCT_CYAN_SENSE = 0x6,                                   /**< SSI  CYAN for SPI SENSE for PWM systems */
    DLPC654X_SDCT_MAGENTA_FREQUENCY = 0x7,                            /**< SSI MAGENTA for SPI / Frequency for PWM systems */
    DLPC654X_SDCT_WHITE = 0x8,                                        /**< SSI SPI designator for WHITE */
    DLPC654X_SDCT_BLACK = 0x9,                                        /**< SSI SPI designator for Black */
    DLPC654X_SDCT_IR = 0xA,                                           /**< SSI SPI designator for IR */
} DLPC654X_SsiDrvColorT_e;

typedef enum
{
    DLPC654X_GPT_ACTIVE_HI = 0x0,                                     /**< Active High */
    DLPC654X_GPT_ACTIVE_LO = 0x1,                                     /**< Active low */
} DLPC654X_GpioPolarityT_e;

typedef enum
{
    DLPC654X_FFMT_FIXED = 0x0,                                        /**< Fixed output frame rate range of 47-63Hz. */
    DLPC654X_FFMT_SYNC1_X = 0x1,                                      /**< FRC in sync with the incoming frame rate. */
    DLPC654X_FFMT_SYNC2_X = 0x2,                                      /**< FRC doubles the incoming frame rate. */
    DLPC654X_FFMT_SYNC3_X = 0x3,                                      /**< FRC triples the incoming frame rate. */
    DLPC654X_FFMT_SYNC4_X = 0x4,                                      /**< FRC 4 X incoming frame rate. */
    DLPC654X_FFMT_SYNC6_X = 0x5,                                      /**< FRC 6 X incoming frame rate. */
} DLPC654X_FrameFrcModeT_e;

typedef enum
{
    DLPC654X_URDPT_NONINV = 0x0,                                      /**< Supply non-inverted version of UART RXD input */
    DLPC654X_URDPT_INV = 0x1,                                         /**< Supply inverted version of UART RXD input */
} DLPC654X_UrtRxDataPolarityT_e;

typedef enum
{
    DLPC654X_CMT_BOOTLOADER = 0x0,                                    /**< Bootloader */
    DLPC654X_CMT_MAIN_APPLICATION = 0x1,                              /**< Main Application */
} DLPC654X_CmdModeT_e;

typedef enum
{
    DLPC654X_SDLRT_LR3_DREF_IN = 0x0,                                 /**< 3D LR from frame determines L/R (High=Left). */
    DLPC654X_SDLRT_LR_GPIO_IN = 0x1,                                  /**< GPIO determines L/R (High=Left). */
    DLPC654X_SDLRT_LR_TOPFIELD = 0x2,                                 /**< Vsync/Hsync alignment determines L/R. */
    DLPC654X_SDLRT_LR1_ST_FRAME = 0x3,                                /**< Lr 1st Frame */
    DLPC654X_SDLRT_LR_EMBEDDED_ENCODING = 0x4,                        /**< L/R reference is embedded in video data. (see @ref SRC 3D LR Encoding t) */
} DLPC654X_Src3DLrReferenceT_e;

typedef enum
{
    DLPC654X_XSMT_NORMAL = 0x0,                                       /**< Normal */
    DLPC654X_XSMT_ALL1 = 0x1,                                         /**< All1 */
    DLPC654X_XSMT_ALL0 = 0x2,                                         /**< All0 */
    DLPC654X_XSMT_PLEASINGCOLOR = 0x3,                                /**< Pleasingcolor */
} DLPC654X_XprSubframeMaskT_e;

typedef enum
{
    DLPC654X_DFMT_MODE_DISABLE = 0x0,                                 /**< Disable film mode */
    DLPC654X_DFMT_MODE_AUTO32 = 0x1,                                  /**< Mode Auto 3 2 */
    DLPC654X_DFMT_MODE_AUTO22 = 0x2,                                  /**< Mode Auto 2 2 */
    DLPC654X_DFMT_MODE_SW32 = 0x3,                                    /**< Software controlled mode for 60Hz source */
    DLPC654X_DFMT_MODE_SW22 = 0x4,                                    /**< Software controlled mode for 50Hz source */
} DLPC654X_DeiFilmModesT_e;

typedef struct
{
    uint8_t AppMajor;
    uint8_t AppMinor;
    uint16_t AppPatch;
    uint8_t ApiMajor;
    uint8_t ApiMinor;
    uint16_t ApiPatch;
} DLPC654X_Version_s;

typedef struct
{
    uint32_t SectorSize;
    uint16_t NumSectors;
} DLPC654X_SectorInfo_s;

typedef struct
{
    uint32_t StartAddress;
    uint8_t AddressIncrement;
    DLPC654X_CmdAccessWidthT_e AccessWidth;
    uint16_t NumberOfWords;
    uint8_t NumberOfBytesPerWord;
    uint8_t* Data;
} DLPC654X_MemoryArray_s;

typedef struct
{
    DLPC654X_Src3DFormatT_e Format;
    DLPC654X_Src3DLrReferenceT_e LrReference;
    DLPC654X_Src3DFrameDominanceT_e FrameDominance;
    DLPC654X_Src3DLrEncodingT_e LrEncoding;
    DLPC654X_Src3DTbReferenceT_e TbReference;
    DLPC654X_Src3DOeReferenceT_e OeReference;
    uint8_t NumActiveBlankLines;
    uint8_t NumberOfEncodedLines;
    uint16_t LeftEncodedLineLocation;
    uint16_t RightEncodedLineLocation;
    DLPC654X_Src3DColorT_e BlankingColor;
} DLPC654X_ThreeDSourceConfiguration_s;

typedef struct
{
    DLPC654X_AlcAlgorithmCtlEnum_e AlgorithmControl;
    DLPC654X_AlcAutoLockPortCmdEnum_e Port1Command;
    DLPC654X_AlcAutoLockPortCmdEnum_e Port2Command;
    DLPC654X_AlcAutoLockPortCmdEnum_e Port3Command;
    DLPC654X_AlcAutoLockPortCmdEnum_e ActivePort;
    DLPC654X_AlcSignalPolarityEnum_e PortVsyncPolarity;
    DLPC654X_AlcSignalPolarityEnum_e PortHsyncPolarity;
    bool PortIsInterlaced;
    uint16_t PortLinesPerFrame;
    uint16_t PortSystemClocksPerLine;
    uint16_t PortVertFreq;
    uint16_t PortHorzFreq;
    uint16_t PortActiveLines;
    uint16_t PortActivePixels;
    uint16_t PortPixelsPerLine;
    uint32_t PortPixelFreqInkHz;
    DLPC654X_AlcSyncTypeEnum_e RawSyncType;
    DLPC654X_AlcSignalPolarityEnum_e RawVsyncPolarity;
    DLPC654X_AlcSignalPolarityEnum_e RawHsyncPolarity;
    bool RawIsInterlaced;
    uint16_t RawLinesPerFrame;
    uint16_t RawSystemClocksPerLine;
    uint16_t RawVertFreq;
    uint16_t RawHorzFreq;
    DLPC654X_AlcAutoLockPortEnum_e AsmSourcePort;
    DLPC654X_AlcOperatingModeEnum_e AsmOperatingMode;
    DLPC654X_AlcAlgorithmStatusEnum_e AsmStatus;
    bool AsmIsPortSyncs;
    uint16_t AsmvRes;
    uint16_t AsmhRes;
    uint16_t AsmActiveTopLine;
    uint16_t AsmActiveBottomLine;
    uint16_t AsmActiveLeftPixel;
    uint16_t AsmActiveRightPixel;
    uint8_t AsmPhaseSetting;
    uint16_t AsmSampleClock;
    uint32_t AsmSampleClockFreqInkHz;
    bool AsmIsComponentVideo;
    bool AsmIsHdtv;
    bool AsmIsYuv;
    bool AsmIsSubSampled;
    bool AsmIsTopfieldInverted;
    bool AsmIsSavedMode;
    uint8_t AsmSavedModeGroup;
    uint8_t AsmSavedModeIndex;
    uint16_t AsmVsyncsUntilResolution;
    DLPC654X_AlcAutoLockPortEnum_e DsmSourcePort;
    DLPC654X_AlcAlgorithmStatusEnum_e DsmStatus;
    uint16_t DsmvRes;
    uint16_t DsmhRes;
    uint16_t DsmPixelsPerLine;
    uint32_t DsmPixelFreqInkHz;
    uint16_t DsmVsyncsUntilResolution;
    DLPC654X_AlcAutoLockEventEnum_e PortPreviousEvent;
    DLPC654X_AlcAutoLockEventEnum_e AsmPreviousEvent;
    DLPC654X_AlcAutoLockEventEnum_e DsmPreviousEvent;
    DLPC654X_AlcAutoLockEventEnum_e MiscPreviousEvent;
} DLPC654X_AutolockStatus_s;

typedef struct
{
    DLPC654X_AlcOperatingModeEnum_e OperatingMode;
    DLPC654X_AlcClockDetectionModeEnum_e ClockDetectionMode;
    uint8_t WideMode;
    uint8_t NumPhases;
    uint16_t VsyncsUntilManual;
    uint16_t RedCalibratedGain;
    uint16_t GreenCalibratedGain;
    uint16_t BlueCalibratedGain;
    uint16_t RedCalibratedOffset;
    uint16_t GreenCalibratedOffset;
    uint16_t BlueCalibratedOffset;
    uint16_t RedChMidLevelCalibratedOffset;
    uint16_t BlueChMidLevelCalibratedOffset;
    DLPC654X_AlcAutolockSmtCfgEnum_e SmtCfg;
} DLPC654X_AutoLockAsmCfgWrite_s;

typedef struct
{
    uint8_t SourceType;
    DLPC654X_AlcOperatingModeEnum_e OperatingMode;
    DLPC654X_AlcClockDetectionModeEnum_e ClockDetectionMode;
    uint8_t WideMode;
    uint8_t NumPhases;
    uint16_t VsyncsUntilManual;
    uint16_t RedCalibratedGain;
    uint16_t GreenCalibratedGain;
    uint16_t BlueCalibratedGain;
    uint16_t RedCalibratedOffset;
    uint16_t GreenCalibratedOffset;
    uint16_t BlueCalibratedOffset;
    uint16_t RedChMidLevelCalibratedOffset;
    uint16_t BlueChMidLevelCalibratedOffset;
    uint8_t Group1NumSavedModes;
    uint8_t Group2NumSavedModes;
    DLPC654X_AlcAutolockSmtCfgEnum_e SmtCfg;
    uint32_t StateMachineDisable;
} DLPC654X_AutoLockAsmCfgRead_s;

typedef struct
{
    uint8_t Strength0;
    uint8_t Strength1;
    uint8_t Strength2;
    uint8_t Strength3;
    uint8_t Strength4;
    uint8_t Strength5;
    uint8_t Strength6;
    uint8_t Strength7;
    uint8_t Strength8;
    uint8_t Strength9;
    uint8_t Strength10;
    uint8_t Strength11;
    uint8_t Strength12;
    uint8_t Strength13;
    uint8_t Strength14;
    uint8_t Strength15;
} DLPC654X_UserInterpolationStrengthEnable_s;

typedef struct
{
    uint32_t MaskBits1;
    uint32_t MaskBits2;
    uint32_t MaskBits3;
    uint32_t LogicValue1;
    uint32_t LogicValue2;
    uint32_t LogicValue3;
} DLPC654X_GpioPins_s;

typedef struct
{
    uint16_t TopLeftX;
    uint16_t TopLeftY;
    uint16_t TopRightX;
    uint16_t TopRightY;
    uint16_t BottomLeftX;
    uint16_t BottomLeftY;
    uint16_t BottomRightX;
    uint16_t BottomRightY;
} DLPC654X_KeystoneCorners_s;

typedef struct
{
    uint16_t Appl;
    uint16_t Alpf;
    uint16_t Hfp;
    uint16_t Hbp;
    uint16_t Vfp;
    uint16_t Vbp;
} DLPC654X_BackendSource_s;

typedef struct
{
    uint8_t Gamma;
    uint8_t WhitePeaking;
    uint16_t RedHue;
    uint16_t RedSaturation;
    uint16_t RedGain;
    uint16_t GreenHue;
    uint16_t GreenSaturation;
    uint16_t GreenGain;
    uint16_t BlueHue;
    uint16_t BlueSaturation;
    uint16_t BlueGain;
    uint16_t CyanHue;
    uint16_t CyanSaturation;
    uint16_t CyanGain;
    uint16_t MagentaHue;
    uint16_t MagentaSaturation;
    uint16_t MagentaGain;
    uint16_t YellowHue;
    uint16_t YellowSaturation;
    uint16_t YellowGain;
    uint16_t WhiteRedGain;
    uint16_t WhiteGreenGain;
    uint16_t WhiteBlueGain;
} DLPC654X_ImageCphw_s;

typedef struct
{
    uint8_t Gamma;
    uint8_t WhitePeaking;
    uint16_t RedHue;
    uint16_t RedSaturation;
    uint16_t RedGain;
    uint16_t GreenHue;
    uint16_t GreenSaturation;
    uint16_t GreenGain;
    uint16_t BlueHue;
    uint16_t BlueSaturation;
    uint16_t BlueGain;
    uint16_t CyanHue;
    uint16_t CyanSaturation;
    uint16_t CyanGain;
    uint16_t MagentaHue;
    uint16_t MagentaSaturation;
    uint16_t MagentaGain;
    uint16_t YellowHue;
    uint16_t YellowSaturation;
    uint16_t YellowGain;
    uint16_t WhiteRedGain;
    uint16_t WhiteGreenGain;
    uint16_t WhiteBlueGain;
} DLPC654X_ImageCpeep_s;

typedef struct
{
    uint16_t HsgRedGain;
    uint16_t HsgRedSaturation;
    uint16_t HsgRedHue;
    uint16_t HsgGreenGain;
    uint16_t HsgGreenSaturation;
    uint16_t HsgGreenHue;
    uint16_t HsgBlueGain;
    uint16_t HsgBlueSaturation;
    uint16_t HsgBlueHue;
    uint16_t HsgCyanGain;
    uint16_t HsgCyanSaturation;
    uint16_t HsgCyanHue;
    uint16_t HsgMagentaGain;
    uint16_t HsgMagentaSaturation;
    uint16_t HsgMagentaHue;
    uint16_t HsgYellowGain;
    uint16_t HsgYellowSaturation;
    uint16_t HsgYellowHue;
    uint16_t HsgWhiteRedGain;
    uint16_t HsgWhiteGreenGain;
    uint16_t HsgWhiteBlueGain;
} DLPC654X_ImageInverseHsg_s;

typedef struct
{
    uint8_t WindowNumber;
    uint16_t Left;
    uint16_t Top;
    uint16_t Height;
    uint16_t Width;
} DLPC654X_GenerateWindow_s;

typedef struct
{
    DLPC654X_SspPortT_e Port;
    DLPC654X_SspChipSelectT_e ChipSelect;
    DLPC654X_SspDeviceT_e DeviceType;
    uint32_t ClockRate;
    uint8_t Command;
    uint8_t Data;
} DLPC654X_SspPassthru_s;

typedef struct
{
    uint16_t Width;
    uint16_t Height;
    uint32_t ByteCount;
    uint8_t PixFormat;
    uint8_t ChromaOrder;
    uint16_t CscTable[];
} DLPC654X_CaptureExternalFrame_s;

typedef struct
{
    uint8_t StartBitPlane;
    uint8_t EndBitPlane;
    uint16_t WhiteStripeWidth;
    uint16_t BlackStripeWidth;
    bool IsVerticalPattern;
    bool StartWithWhiteStripe;
} DLPC654X_SetStructuredPatternStripes_s;

typedef struct
{
    uint16_t Width;
    uint16_t Space;
    uint16_t LineColorRed;
    uint16_t LineColorGreen;
    uint16_t LineColorBlue;
    uint16_t BackgroundColorRed;
    uint16_t BackgroundColorGreen;
    uint16_t BackgroundColorBlue;
} DLPC654X_TpgHorizontalLines_s;

typedef struct
{
    uint16_t Width;
    uint16_t Space;
    uint16_t LineColorRed;
    uint16_t LineColorGreen;
    uint16_t LineColorBlue;
    uint16_t BackgroundColorRed;
    uint16_t BackgroundColorGreen;
    uint16_t BackgroundColorBlue;
} DLPC654X_TpgVerticalLines_s;

typedef struct
{
    uint16_t Period;
    uint16_t LineColorRed;
    uint16_t LineColorGreen;
    uint16_t LineColorBlue;
    uint16_t BackgroundColorRed;
    uint16_t BackgroundColorGreen;
    uint16_t BackgroundColorBlue;
} DLPC654X_TpgDiagonalLines_s;

typedef struct
{
    uint16_t HorizontalWidth;
    uint16_t HorizontalSpace;
    uint16_t VerticalWidth;
    uint16_t VerticalSpace;
    uint16_t LineColorRed;
    uint16_t LineColorGreen;
    uint16_t LineColorBlue;
    uint16_t BackgroundColorRed;
    uint16_t BackgroundColorGreen;
    uint16_t BackgroundColorBlue;
} DLPC654X_TpgGridLines_s;

typedef struct
{
    DLPC654X_TpgHorzRampStepT_e RampStep;
    uint16_t Height;
    bool ColorEnableRed;
    bool ColorEnableGreen;
    bool ColorEnableBlue;
} DLPC654X_TpgColorHorzRamp_s;

typedef struct
{
    uint16_t StartIntensity;
    uint16_t RampStepWidth;
    uint16_t RampStepInc;
    uint16_t BackgroundColorRed;
    uint16_t BackgroundColorGreen;
    uint16_t BackgroundColorBlue;
    DLPC654X_TpgColorT_e RampColor;
} DLPC654X_TpgFixedRamp_s;

typedef struct
{
    DLPC654X_TpgLineColorT_e ForwardLineStartColor;
    DLPC654X_TpgLineColorT_e BackwardLineColor;
    uint16_t BackgroundColorRed;
    uint16_t BackgroundColorGreen;
    uint16_t BackgroundColorBlue;
    bool DoubleLineMode;
    uint16_t Distance;
} DLPC654X_TpgDiamondDiagLines_s;

typedef struct
{
    uint16_t Width;
    uint16_t Height;
    uint16_t TopLeftCheckerColorRed;
    uint16_t TopLeftCheckerColorGreen;
    uint16_t TopLeftCheckerColorBlue;
    uint16_t NextCheckerColorRed;
    uint16_t NextCheckerColorGreen;
    uint16_t NextCheckerColorBlue;
} DLPC654X_TpgCheckerboard_s;

typedef struct
{
    bool CwSpinning;
    bool CwPhaselock;
    bool CwFreqlock;
    bool Lamplit;
    bool MemTstPassed;
    bool FrameRateConvEn;
    bool SeqPhaselock;
    bool SeqFreqlock;
    bool SeqSearch;
    bool ScpcalEnable;
    bool VicalEnable;
    bool BccalEnable;
    bool SequenceErr;
    bool PixclkOor;
    bool SyncvalStat;
    bool UartPort0CommErr;
    bool UartPort1CommErr;
    bool UartPort2CommErr;
    bool SspPort0CommErr;
    bool SspPort1CommErr;
    bool SspPort2CommErr;
    bool I2CPort0CommErr;
    bool I2CPort1CommErr;
    bool I2CPort2CommErr;
    bool DlpcInitErr;
    bool LampHwErr;
    bool LampPprftout;
    bool NoFreqBinErr;
    bool Dlpa3000CommErr;
    bool UmcRefreshBwUnderflowErr;
    bool DmdInitErr;
    bool DmdPwrDownErr;
    bool SrcdefNotpresent;
    bool SeqbinNotpresent;
    bool EepromInitFail;
} DLPC654X_SystemStatus_s;

typedef struct
{
    bool InvalidateSettings;
    bool InvalidateColorwheelLampData;
    bool InvalidateSsiCalibrationData;
    bool InvalidateAdcCalibrationData;
    bool InvalidateWpcSensorCalibrationData;
    bool InvalidateWpcBrightnessTableData;
    bool InvalidateXprCalibrationData;
    bool InvalidateXprWaveformCalibrationData;
    bool InvalidateSurfaceCorrectionData;
} DLPC654X_EepromInvalidate_s;

typedef struct
{
    uint8_t Major;
    uint8_t Minor1;
    uint8_t Minor2;
    uint16_t Patch;
    uint8_t* BuildHash;
} DLPC654X_ComposerVersion_s;

typedef struct
{
    double RedIdealDutyCycle;
    double GreenIdealDutyCycle;
    double BlueIdealDutyCycle;
    double RedOptimalDutyCycle;
    double GreenOptimalDutyCycle;
    double BlueOptimalDutyCycle;
} DLPC654X_WpcOptimalDutyCycle_s;

typedef struct
{
    uint16_t ChromaticX;
    uint16_t ChromaticY;
    uint32_t LuminenceY;
    uint32_t RedSensorOutput;
    uint32_t GreenSensorOutput;
    uint32_t BlueSensorOutput;
    double DutyCycle;
} DLPC654X_WpcCalibrationData_s;

typedef struct
{
    uint8_t Index;
    uint16_t AperturePosition;
    uint16_t RedDriveLevel;
    uint16_t GreenDriveLevel;
    uint16_t BlueDriveLevel;
} DLPC654X_WpcBrightnessTableEntry_s;

typedef struct
{
    bool Comm;
    bool ThreeD;
    bool MessageService;
    bool I2C;
    bool ClosedCaptioning;
    bool DdcCi;
    bool Gui;
    bool Environment;
    bool Illumination;
    bool System;
    bool Eeprom;
    bool Datapath;
    bool Autolock;
    bool ProjectorCtl;
    bool Peripheral;
    bool Ir;
    bool Usb;
    bool Mailbox;
} DLPC654X_DebugMessageMask_s;

typedef struct
{
    uint8_t HighCountTasks;
    uint8_t HighCountEvts;
    uint8_t HighCountGrpEvts;
    uint8_t HighCountMbxs;
    uint8_t HighCountMemPools;
    uint8_t HighCountSems;
    uint8_t CurrCountTasks;
    uint8_t CurrCountEvts;
    uint8_t CurrCountGrpEvts;
    uint8_t CurrCountMbxs;
    uint8_t CurrCountMemPools;
    uint8_t CurrCountSems;
} DLPC654X_Resource_s;

typedef struct
{
    uint16_t Appl;
    uint16_t Alpf;
    uint16_t TpplLargest;
    uint16_t Tlpf;
    uint16_t TpplSmallest;
    uint16_t Vfp;
    uint16_t Vbp;
    uint16_t Vsw;
    uint16_t Hfp;
    uint16_t Hbp;
    uint16_t Hsw;
    uint16_t Hs2Vs;
    uint16_t Vs2Hs;
    bool HSyncPolarityIsPositive;
    bool VSyncPolarityIsPositive;
    uint16_t FreqCaptured;
} DLPC654X_Vx1HwStatus_s;

typedef struct
{
    uint16_t CroppedAreaFirstPixel;
    uint16_t CroppedAreaFirstLine;
    uint16_t CroppedAreaPixelsPerLine;
    uint16_t CroppedAreaLinesPerFrame;
    uint16_t DisplayAreaFirstPixel;
    uint16_t DisplayAreaFirstLine;
    uint16_t DisplayAreaPixelsPerLine;
    uint16_t DisplayAreaLinesPerFrame;
} DLPC654X_DisplayImageSize_s;

typedef struct
{
    DLPC654X_SrcSyncConfigT_e VSyncConfiguration;
    DLPC654X_SrcSyncConfigT_e HSyncConfiguration;
    DLPC654X_SrcSyncConfigT_e TopFieldConfiguration;
    DLPC654X_SrcDownSampleConfigT_e DownSampleConfiguration;
    bool IsThreeD;
    bool IsClockPolarityPositive;
    DLPC654X_SrcPixelFormatT_e PixelFormat;
    bool IsExternalDataEnable;
    bool IsInterlaced;
    bool IsOffsetBinary;
    bool IsTopFieldInvertedAtScaler;
    uint16_t TotalAreaPixelsPerLine;
    uint16_t TotalAreaLinesPerFrame;
    uint16_t ActiveAreaFirstPixel;
    uint16_t ActiveAreaFirstLine;
    uint16_t ActiveAreaPixelsPerLine;
    uint16_t ActiveAreaLinesPerFrame;
    uint16_t BottomFieldFirstLine;
    uint32_t PixelClockFreqInKiloHertz;
    uint16_t ColorSpaceConvCoeffs0;
    uint16_t ColorSpaceConvCoeffs1;
    uint16_t ColorSpaceConvCoeffs2;
    uint16_t ColorSpaceConvCoeffs3;
    uint16_t ColorSpaceConvCoeffs4;
    uint16_t ColorSpaceConvCoeffs5;
    uint16_t ColorSpaceConvCoeffs6;
    uint16_t ColorSpaceConvCoeffs7;
    uint16_t ColorSpaceConvCoeffs8;
    uint16_t OffsetRed;
    uint16_t OffsetGreen;
    uint16_t OffsetBlue;
    uint8_t IsVideo;
    uint8_t IsHighDefinitionVideo;
    double FrameRate;
} DLPC654X_SourceConfiguration_s;

typedef struct
{
    DLPC654X_SrcFpdDataMapModeT_e FpdMode;
    DLPC654X_SrcFpdDataInterfaceModeT_e DataInterfaceMode;
    bool Enable3DRef;
    bool EnableField;
    bool EnablePixelRepeat;
} DLPC654X_FpdConfiguration_s;

typedef struct
{
    uint8_t ControlPointsDefinedByArray;
    uint16_t InputWidth;
    uint16_t InputHeight;
    uint16_t WarpColumns;
    uint16_t WarpRows;
    uint16_t HorizontalCtrlPoints[10];
    uint16_t VerticalCtrlPoints[];
} DLPC654X_ManualWarpControlPoints_s;

typedef struct
{
    uint16_t DriveLevelRed;
    uint16_t DriveLevelGreen;
    uint16_t DriveLevelBlue;
    uint16_t DriveLevelC1;
    uint16_t DriveLevelC2;
    uint16_t DriveLevelSense;
} DLPC654X_SsiDriveLevels_s;

typedef struct
{
    bool ChromaTransientImprovementEnableBit;
    bool GammaCorrectionEnableBit;
    bool ColorCoordinateAdjustmentEnableBit;
    bool BrilliantColorEnableBit;
    bool WhitePointCorrectionEnableBit;
    bool DynamicBlackEnableBit;
    bool HdrEnableBit;
} DLPC654X_ImageAlgorithmEnable_s;

typedef struct
{
    double OrigCoordsRedX;
    double OrigCoordsRedY;
    double OrigCoordsRedLum;
    double OrigCoordsGreenX;
    double OrigCoordsGreenY;
    double OrigCoordsGreenLum;
    double OrigCoordsBlueX;
    double OrigCoordsBlueY;
    double OrigCoordsBlueLum;
    double OrigCoordsWhiteX;
    double OrigCoordsWhiteY;
    double OrigCoordsWhiteLum;
    double OrigCoordsC1X;
    double OrigCoordsC1Y;
    double OrigCoordsC1Lum;
    double OrigCoordsC2X;
    double OrigCoordsC2Y;
    double OrigCoordsC2Lum;
    double OrigCoordsDraAX;
    double OrigCoordsDraAY;
    double OrigCoordsDraALum;
    double OrigCoordsDraBX;
    double OrigCoordsDraBY;
    double OrigCoordsDraBLum;
    double OrigCoordsDraCX;
    double OrigCoordsDraCY;
    double OrigCoordsDraCLum;
    double TargetCoordsRedX;
    double TargetCoordsRedY;
    double TargetCoordsRedGain;
    double TargetCoordsGreenX;
    double TargetCoordsGreenY;
    double TargetCoordsGreenGain;
    double TargetCoordsBlueX;
    double TargetCoordsBlueY;
    double TargetCoordsBlueGain;
    double TargetCoordsCyanX;
    double TargetCoordsCyanY;
    double TargetCoordsCyanGain;
    double TargetCoordsMagentaX;
    double TargetCoordsMagentaY;
    double TargetCoordsMagentaGain;
    double TargetCoordsYellowX;
    double TargetCoordsYellowY;
    double TargetCoordsYellowGain;
    double TargetCoordsWhiteX;
    double TargetCoordsWhiteY;
    double TargetCoordsWhiteGain;
} DLPC654X_ImageCcaCoordinates_s;

typedef struct
{
    double HsgRedGain;
    double HsgRedSaturation;
    double HsgRedHue;
    double HsgGreenGain;
    double HsgGreenSaturation;
    double HsgGreenHue;
    double HsgBlueGain;
    double HsgBlueSaturation;
    double HsgBlueHue;
    double HsgCyanGain;
    double HsgCyanSaturation;
    double HsgCyanHue;
    double HsgMagentaGain;
    double HsgMagentaSaturation;
    double HsgMagentaHue;
    double HsgYellowGain;
    double HsgYellowSaturation;
    double HsgYellowHue;
    double HsgWhiteRedGain;
    double HsgWhiteGreenGain;
    double HsgWhiteBlueGain;
} DLPC654X_ImageHsg_s;

typedef struct
{
    DLPC654X_HdrTransferFnT_e TransferFunction;
    double MasterDisplayBlackLevel;
    double MasterDisplayWhiteLevel;
    double MasterDisplayColorGamutRedX;
    double MasterDisplayColorGamutRedY;
    double MasterDisplayColorGamutGreenX;
    double MasterDisplayColorGamutGreenY;
    double MasterDisplayColorGamutBlueX;
    double MasterDisplayColorGamutBlueY;
    double MasterDisplayColorGamutWhiteX;
    double MasterDisplayColorGamutWhiteY;
} DLPC654X_HdrSourceConfiguration_s;

typedef struct
{
    bool MaskCwSpinning;
    bool MaskCwPhaselock;
    bool MaskCwFreqlock;
    bool MaskLamplit;
    bool MaskMemTstPassed;
    bool MaskFrameRateConvEn;
    bool MaskSeqPhaselock;
    bool MaskSeqFreqlock;
    bool MaskSeqSearch;
    bool MaskScpcalEnable;
    bool MaskVicalEnable;
    bool MaskBccalEnable;
    bool MaskSequenceErr;
    bool MaskPixclkOor;
    bool MaskSyncvalStat;
    bool MaskDadThermFlt;
    bool MaskDadVoltFlt;
    bool MaskDadCurrentFlt;
    bool MaskUartPort0CommErr;
    bool MaskUartPort1CommErr;
    bool MaskUartPort2CommErr;
    bool MaskSspPort0CommErr;
    bool MaskSspPort1CommErr;
    bool MaskSspPort2CommErr;
    bool MaskI2CPort0CommErr;
    bool MaskI2CPort1CommErr;
    bool MaskI2CPort2CommErr;
    bool MaskDlpcInitErr;
    bool MaskLampHwErr;
    bool MaskLampPprftout;
    bool MaskNoFreqBinErr;
    bool MaskDlpa3000CommErr;
    bool MaskUmcRefreshBwUnderflowErr;
    bool MaskEepromInitFail;
} DLPC654X_StatusIntMask_s;

typedef struct
{
    uint32_t SampleRate;
    uint8_t InCounterEnabled;
    uint16_t HighPulseWidth;
    uint16_t LowPulseWidth;
    uint8_t DutyCycle;
} DLPC654X_PwmInputConfiguration_s;

typedef struct
{
    DLPC654X_I2CPortT_e Port;
    uint8_t Is7Bit;
    uint8_t HasSubaddr;
    uint32_t ClockRate;
    uint16_t DeviceAddress;
    uint8_t* SubAddr;
    uint8_t* DataBytes;
} DLPC654X_I2CPassthrough_s;

typedef struct
{
    bool Enable;
    DLPC654X_UrtBaudRateT_e BaudRate;
    DLPC654X_UrtDataBitsT_e DataBits;
    DLPC654X_UrtStopBitsT_e StopBits;
    DLPC654X_UrtParityT_e Parity;
    DLPC654X_UrtFlowControlT_e FlowControl;
    DLPC654X_UrtFifoTriggerLevelT_e RxTrigLevel;
    DLPC654X_UrtFifoTriggerLevelT_e TxTrigLevel;
    DLPC654X_UrtRxDataPolarityT_e RxDataPolarity;
    DLPC654X_UrtRxDataSourceT_e RxDataSource;
} DLPC654X_UartConfiguration_s;

typedef struct
{
    bool DmdEnable;
    bool IllumEnable;
    bool BuckGp1Enable;
    bool BuckGp2Enable;
    bool BuckGp3Enable;
    bool CwEnable;
    bool FastShutdownEnable;
} DLPC654X_PadEnableRegister_s;

typedef struct
{
    bool TemperatureWarning;
    bool TemperatureShutdown;
    bool BatteryLowWarning;
    bool BatteryLowShutdown;
    bool DmdFault;
    bool ProjOn;
    bool IllumFault;
    bool SupplyFault;
} DLPC654X_PadMainStatusRegister_s;

typedef struct
{
    bool TemperatureWarningMask;
    bool TemperatureShutdownMask;
    bool BatteryLowWarningMask;
    bool BatteryLowShutdownMask;
    bool DmdFaultMask;
    bool ProjOnMask;
    bool IllumFaultMask;
    bool SupplyFaultMask;
} DLPC654X_PadInterruptMaskRegister_s;

typedef struct
{
    bool IllumExtSwitchSel;
    bool IllumIntSwitchSel;
    bool IllumDualOutputSel;
    bool Illum3AIntSwitchSel;
    bool IllumExtLsdCurrLimitEnable;
    bool DigSpiFastSel;
} DLPC654X_PadUserConfigurationSelection_s;

typedef struct
{
    bool DmdBuck2Use;
    bool DmdBuck1Use;
    bool DmdLdo2Use;
    bool DmdLdo1Use;
    bool CwCapability;
    bool IllumExtSwitchCapability;
} DLPC654X_PadCapabilityRegister_s;

typedef struct
{
    bool IllumBc2PgFault;
    bool IllumBc1PgFault;
    bool BuckGp2PgFault;
    bool BuckGp1PgFault;
    bool BuckGp3PgFault;
} DLPC654X_PadDetailedStatusRegister1_s;

typedef struct
{
    bool IllumBc2OvFault;
    bool IllumBc1OvFault;
    bool BuckGp2OvFault;
    bool BuckGp1OvFault;
    bool BuckGp3OvFault;
} DLPC654X_PadDetailedStatusRegister2_s;

typedef struct
{
    bool Gp2OrDmd2PgFault;
    bool Gp1OrDmd1PgFault;
    bool BuckDmd2PgFault;
    bool BuckDmd1PgFault;
    bool DmdPgFault;
} DLPC654X_PadDetailedStatusRegister3_s;

typedef struct
{
    uint8_t DmdMode;
    bool LegacyMode;
    bool DccOverride;
    bool CdrOverride;
    bool MpcEnable;
    bool NsflipEnable;
    bool SingleMacroMode;
    uint8_t ShadowResetMode;
    uint8_t SuperblkMode;
    bool RowdblEnable;
    bool PreconditionEnable;
    bool ExpFuseEnable;
    bool DisabledPhaseResets;
} DLPC654X_DmdModeRegister_s;

typedef struct
{
    bool SramReadEnable;
    uint8_t SramReadControl;
    bool SlrScanEnable;
    bool FuseReadEnable;
    bool ProgFuseEnable;
    bool CombinedBistMode;
    bool BurnInEnable;
    bool BurnInBistClear;
    uint8_t BurnInData;
    bool BistEnable;
    bool BsaRstTestMode;
    uint8_t TestResetVoltage;
    bool BurnInBypassMode;
    bool RstIccqMode;
    bool BsaBypass;
    bool IgnoreParity;
} DLPC654X_DmdTestRegister_s;

typedef struct
{
    bool HssiPktError;
    bool LsifPktError;
    bool ParkModeEnable;
    bool HsifEnable;
    bool LsifParityError;
    uint8_t BsaStepDownCounter;
    bool MpsTimeoutMonitor;
} DLPC654X_DmdErrorStatusRegister_s;

typedef struct
{
    bool DccClkEnable;
    bool QuickstartEnable;
    bool QbrEnable;
    bool TestmuxEnable;
    bool PrbsEnable;
    bool TestClkEnable;
    bool HsReset;
    bool LoopEnable;
    uint8_t Loopamp;
    bool LvdsEnable;
    uint8_t FiltDoc;
    uint8_t IpfSel;
    uint8_t ItrimIbuf;
    uint8_t ItrimDcc;
} DLPC654X_DmdHssiConfig1Register_s;

typedef struct
{
    uint8_t Vcm;
    bool Voteclrz;
    bool ScopeEnable;
    bool ScopePolarity;
    bool PhaseInterpolatorEnable;
    bool Txramp;
    bool Txdir;
    bool Txmag;
    bool CoarseOrFine;
    uint8_t PhaseInterpolatorCode;
} DLPC654X_DmdHssiConfig2Register_s;

typedef struct
{
    bool HssiDataLane0Enable;
    bool HssiDataLane1Enable;
    bool HssiDataLane2Enable;
    bool HssiDataLane3Enable;
    bool HssiDataLane4Enable;
    bool HssiDataLane5Enable;
    bool HssiDataLane6Enable;
    bool HssiDataLane7Enable;
} DLPC654X_DmdHssiLaneEnable0Register_s;

typedef struct
{
    bool HssiDataLane0Enable;
    bool HssiDataLane1Enable;
    bool HssiDataLane2Enable;
    bool HssiDataLane3Enable;
    bool HssiDataLane4Enable;
    bool HssiDataLane5Enable;
    bool HssiDataLane6Enable;
    bool HssiDataLane7Enable;
} DLPC654X_DmdHssiLaneEnable1Register_s;

typedef struct
{
    bool HssiDataLane0Enable;
    bool HssiDataLane1Enable;
    bool HssiDataLane2Enable;
    bool HssiDataLane3Enable;
    bool HssiDataLane4Enable;
    bool HssiDataLane5Enable;
    bool HssiDataLane6Enable;
    bool HssiDataLane7Enable;
} DLPC654X_DmdHssiLaneEnable2Register_s;

typedef struct
{
    bool HssiDataLane0Enable;
    bool HssiDataLane1Enable;
    bool HssiDataLane2Enable;
    bool HssiDataLane3Enable;
    bool HssiDataLane4Enable;
    bool HssiDataLane5Enable;
    bool HssiDataLane6Enable;
    bool HssiDataLane7Enable;
} DLPC654X_DmdHssiLaneEnable3Register_s;

typedef struct
{
    bool HssiMacro0Enable;
    bool HssiMacro1Enable;
    bool HssiMacro2Enable;
    bool HssiMacro3Enable;
    bool HssiMacro4Enable;
    bool HssiMacro5Enable;
    bool HssiMacro6Enable;
    bool HssiMacro7Enable;
    bool HssiMacro8Enable;
    bool HssiMacro9Enable;
    bool HssiMacro10Enable;
    bool HssiMacro11Enable;
    bool HssiMacro12Enable;
    bool HssiMacro13Enable;
    bool HssiMacro14Enable;
    bool HssiMacro15Enable;
} DLPC654X_DmdHssiMacroEnableRegister_s;

typedef struct
{
    uint8_t AmuxSel;
    uint8_t DmuxSel;
    uint8_t MacroSel;
    bool CoreAmuxSel;
    bool CoreDmuxSel;
} DLPC654X_DmdAmuxDmuxSelectRegister_s;

typedef struct
{
    bool WlDriver20PercentEnable;
    bool WlDriver40PercentEnable;
    uint8_t BlDriverEnable;
    bool ProgrammableTimingEnable;
    bool BlockClearTransistorsEnable;
} DLPC654X_DmdDriveControlRegister_s;

typedef struct
{
    bool ProgtimeRclkBeginPulse;
    bool ProgtimeRclkEndPulse;
    uint8_t RclkBeginPulseTimeCode;
    uint8_t RclkEndPulseTimeCode;
    bool ProgtimeWenBeginPulse;
    bool ProgtimeWenEndPulse;
    uint8_t WenBeginPulseTimeCode;
    uint8_t WenEndPulseTimeCode;
    bool ProgtimeBl1BeginPulse;
    bool ProgtimeBl1EndPulse;
    uint8_t Bl1BeginPulseTimeCode;
    uint8_t Bl1EndPulseTimeCode;
    bool ProgtimeBl0BeginPulse;
    bool ProgtimeBl0EndPulse;
    uint8_t Bl0BeginPulseTimeCode;
    uint8_t Bl0EndPulseTimeCode;
} DLPC654X_DmdVarTimingRegister_s;

typedef struct
{
    bool Lane7Enable;
    bool Lane6Enable;
    bool Lane5Enable;
    bool Lane4Enable;
    bool Lane3Enable;
    bool Lane2Enable;
    bool Lane1Enable;
    bool Lane0Enable;
} DLPC654X_LaneEnable_s;

typedef struct
{
    bool BitNotLocked;
    bool BitLocked;
    bool ByteNotLocked;
    bool ByteLocked;
    bool DataNotLocked;
    bool DataLocked;
} DLPC654X_InterruptEnable_s;

typedef struct
{
    bool BitNotLocked;
    bool BitLocked;
    bool ByteNotLocked;
    bool ByteLocked;
    bool DataNotLocked;
    bool DataLocked;
} DLPC654X_InterruptStatus_s;

typedef struct
{
    bool BitNotLocked;
    bool BitLocked;
    bool ByteNotLocked;
    bool ByteLocked;
    bool DataNotLocked;
    bool DataLocked;
} DLPC654X_InterruptClear_s;

typedef struct
{
    bool Lane7BitLocked;
    bool Lane6BitLocked;
    bool Lane5BitLocked;
    bool Lane4BitLocked;
    bool Lane3BitLocked;
    bool Lane2BitLocked;
    bool Lane1BitLocked;
    bool Lane0BitLocked;
} DLPC654X_BitLockMonitorLan0ToLane7_s;

typedef struct
{
    bool Lane15BitLocked;
    bool Lane14BitLocked;
    bool Lane13BitLocked;
    bool Lane12BitLocked;
    bool Lane11BitLocked;
    bool Lane10BitLocked;
    bool Lane9BitLocked;
    bool Lane8BitLocked;
} DLPC654X_BitLockMonitorLane8ToLane15_s;

typedef struct
{
    bool Lane7ByteLocked;
    bool Lane6ByteLocked;
    bool Lane5ByteLocked;
    bool Lane4ByteLocked;
    bool Lane3ByteLocked;
    bool Lane2ByteLocked;
    bool Lane1ByteLocked;
    bool Lane0ByteLocked;
} DLPC654X_ByteLockMonitorLane0ToLane7_s;

typedef struct
{
    bool Lane15ByteLocked;
    bool Lane14ByteLocked;
    bool Lane13ByteLocked;
    bool Lane12ByteLocked;
    bool Lane11ByteLocked;
    bool Lane10ByteLocked;
    bool Lane9ByteLocked;
    bool Lane8ByteLocked;
} DLPC654X_ByteLockMonitorLane8ToLane15_s;

typedef struct
{
    bool Lane7DataLocked;
    bool Lane6DataLocked;
    bool Lane5DataLocked;
    bool Lane4DataLocked;
    bool Lane3DataLocked;
    bool Lane2DataLocked;
    bool Lane1DataLocked;
    bool Lane0DataLocked;
} DLPC654X_DataLockMonitorLane0ToLane7_s;

typedef struct
{
    bool Lane15DataLocked;
    bool Lane14DataLocked;
    bool Lane13DataLocked;
    bool Lane12DataLocked;
    bool Lane11DataLocked;
    bool Lane10DataLocked;
    bool Lane9DataLocked;
    bool Lane8DataLocked;
    uint8_t DeskewReferenceLane;
    bool DeskewEnable;
} DLPC654X_DataLockMonitorLane8ToLane15_s;

typedef struct
{
    bool Lane7VideoOutputEnabled;
    bool Lane6VideoOutputEnabled;
    bool Lane5VideoOutputEnabled;
    bool Lane4VideoOutputEnabled;
    bool Lane3VideoOutputEnabled;
    bool Lane2VideoOutputEnabled;
    bool Lane1VideoOutputEnabled;
    bool Lane0VideoOutputEnabled;
} DLPC654X_VideoOutputEnableLane0ToLane7_s;

typedef struct
{
    bool Lane15VideoOutputEnabled;
    bool Lane14VideoOutputEnabled;
    bool Lane13VideoOutputEnabled;
    bool Lane12VideoOutputEnabled;
    bool Lane11VideoOutputEnabled;
    bool Lane10VideoOutputEnabled;
    bool Lane9VideoOutputEnabled;
    bool Lane8VideoOutputEnabled;
} DLPC654X_VideoOutputEnableLane8ToLane15_s;


/**
 * This command returns whether we are in Bootloader or in Main Application.
 *
 * \param[out]  AppMode  Application Mode
 * \param[out]  ControllerConfig  Controller Configuration
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadMode(DLPC654X_CmdModeT_e *AppMode, DLPC654X_CmdControllerConfigT_e *ControllerConfig);

/**
 * This command returns the version of the currently active Application and the version of the underlying API library. The currently active application can be queried using [[#t-mode_read]] command.
 *
 * \param[out]  Version  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadVersion(DLPC654X_Version_s *Version);

/**
 * This command is used to switch between bootloader and application mode.
 *
 * \param[in]  SwitchMode  Application to switch to
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteSwitchMode(DLPC654X_CmdSwitchTypeT_e SwitchMode);

/**
 * Returns the code that specifies the reason for being in bootloader mode.
 *
 * \param[out]  ReasonCode  Reason code<br>0x00    BOOT_HOLD jumper in HOLD position <br> 0x01    Switched to programming mode initated by main app <br> 0x02    Reading flash info failed  <br> 0x03   Flash layout mismatch  <br> 0x04    Can't initialize ARM peripherals  <br> 0x05    Can't allocate memory pool  <br> 0x06    Failure in initialization task  <br> 0x07    Error in USB init  <br> 0x08    Error in i2c init  <br> 0x09    Error getting app configuration  <br> 0x0A    App configuration layout mismatch <br>
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadBootHoldReason(uint8_t *ReasonCode);

/**
 * This command returns the flash device and manufacturer IDs. Only CFI compliant flash devices are supported.
 *
 * \param[out]  ManfId  Manufacturer ID
 * \param[out]  DevId  Device ID
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadFlashId(uint16_t *ManfId, uint64_t *DevId);

/**
 * This command returns the flash sector information read from CFI compliant flash device. If the flash is non-CFI compliant, this command will fail.
 *
 * \param[out]  SectorInfo  Sector Information
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadGetFlashSectorInformation(DLPC654X_SectorInfo_s *SectorInfo[]);

/**
 * This command unlocks the flash update operation (Download, Erase). By default the flash update operations are locked. This is to prevent accidental modification of flash contents. To unlock, the pre-defined key shall be send as the unlock code. Calling this command with any other parameter will lock the flash update commands.
 *
 * \param[in]  Unlock  Flash Update lock/unlock
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteUnlockFlashForUpdate(DLPC654X_CmdFlashUpdateT_e Unlock);

/**
 * This command returns whether the flash is in unlocked state.
 *
 * \param[out]  IsUnlocked  <BR>1 = Unlocked <BR> 0 = Locked <BR>
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadUnlockFlashForUpdate(uint8_t *IsUnlocked);

/**
 * This command erases the sector of the flash where the given address falls in. This command is a flash update command, and requires flash operations to be unlocked using [[#t-unlock_flash_for_update_write]] command. The sector address shall be specified as an offset from flash start address. For example in a flash device where all sectors are 64KB of size, sector addresses shall be specified as follows:- <br> Sector 0 = 0 <br> Sector 1 = 0x10000 <br> Sector 2 = 0x20000 <br> and so on... <br>
 *
 * \param[in]  SectorAddress  Sector Address <br>
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteEraseSector(uint32_t SectorAddress);

/**
 * This command initializes flash read/write operation. This command shall be called before a [[#t-flash_write_command]] command is sent. Note: For Flash Write, the Address and download size set up shall both be even.__
 *
 * \param[in]  StartAddress  Start Address offset to program data to where Offset 0 refers to first byte in the flash, 1 refers to second byte and so on. This offset must be an even number.
 * \param[in]  NumBytes  This specifies the number of bytes Flash Write command should expect or the number of bytes Flash Read command should return.This must be an even number.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteInitializeFlashReadWriteSettings(uint32_t StartAddress, uint32_t NumBytes);

/**
 * This command is used to program data to flash. This command shall be called only after setting the start address and size using the [[#t-initialize_flash_read_write_settings_write]] command. This command is a flash update command, and requires flash operations to be unlocked using [[#t-unlock_flash_for_update_write]] command. <BR> FlashWrite commands can be chained till the initialized number of bytes are programmed. The bootloader will auto-increment the address and size for each command. Only the initialized number of bytes will be programmed even if more data is provided. <BR> __It is important to send only even number of bytes per flash write command to ensure all bytes are written. This is done so that all flash writes are optimized as per the multi-word write supported by the flash device.__ This command supports variable sized payload.
 *
 * \param[in]  DataLength  Byte Length for Data
 * \param[in]  Data  Data to write to flash memory
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteFlashWrite(uint16_t DataLength, uint8_t* Data);

/**
 * This command is used to read data from flash. This command shall be called only after setting the start address and size using the [[#t-initialize_flash_read_write_settings_write]] command. <BR> FlashRead commands can be chained till the initialized number of bytes are returned. The bootloader will auto-increment the address and size for each command. Only the initialized number of bytes will be returned. Calling the function after returning requested data will return in command failure. This command supports variable sized response.
 *
 * \param[in]  NumBytesToRead  Num bytes to read in this command
 * \param[out]  Data  The bytes read from the flash
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadFlashWrite(uint16_t NumBytesToRead, uint8_t *Data);

/**
 * This command computes and returns the checksum starting at the given address for the specified number of bytes. Checksum is calculated as below:- <BR> uint32 SimpleChecksum = 0; <BR> uint32 SumofSumChecksum = 0; <BR> uint08 *Addr = (uint08 *) StartAddress; <BR> <BR> while (NumBytes--) <BR> { <BR> SimpleChecksum += *Addr++; <BR> SumofSumChecksum += SimpleChecksum; <BR> } <BR>
 *
 * \param[in]  StartAddress  Start Address offset for checksum computation where Offset 0 refers to first byte in the flash, 1 refers to second byte and so on.
 * \param[in]  NumBytes  Number of bytes to compute checksum
 * \param[out]  SimpleChecksum  Simple additive checksum
 * \param[out]  SumofSumChecksum  Sum of simple additive checksum calculated at each address
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadChecksum(uint32_t StartAddress, uint32_t NumBytes, uint32_t *SimpleChecksum, uint32_t *SumofSumChecksum);

/**
 * This command attempts a direct write of the given 32-bit value to the given 32-bit memory address. The memory address is not verified whether it is valid to write to the location.
 *
 * \param[in]  Address  Memory address to write to. Address must be a multiple of 4.
 * \param[in]  Value  Value to write
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteMemory(uint32_t Address, uint32_t Value);

/**
 * This command returns the 32-bit value stored at the given 32-bit memory address.
 *
 * \param[in]  Address  Memory Address to read data from. Address must be a multiple of 4.
 * \param[out]  Value  Value read from address
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadMemory(uint32_t Address, uint32_t *Value);

/**
 * Writes a stream of words into the RAM memory (DRAM or IRAM) starting from the address specified. Performs no checks as to whether the specified memory address given is valid or not.
 *
 * \param[in]  DataLength  Byte Length for Data
 * \param[in]  MemoryArray  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteMemoryArray(uint16_t DataLength, DLPC654X_MemoryArray_s *MemoryArray);

/**
 * Reads a stream of words from memory starting from the address specified. Performs no checks as to whether the specified memory address given is valid or not.
 *
 * \param[in]  StartAddress  Start Address from which data is to be read
 * \param[in]  AddressIncrement  Address increment steps. 0 - No increment
 * \param[in]  AccessWidth  Read access width
 * \param[in]  NumberOfWords  Number of words to be read
 * \param[in]  NumberOfBytesPerWord  The number of bytes per word
 * \param[out]  Data  Data
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadMemoryArray(uint32_t StartAddress, uint8_t AddressIncrement, DLPC654X_CmdAccessWidthT_e AccessWidth, uint16_t NumberOfWords, uint8_t NumberOfBytesPerWord, uint8_t *Data);

/**
 * This command sets the current system look. System looks shall be designed and configured via DLP Composer tool. System look determines the current group of sequences and color points to be loaded. This command also initiates the source definition change that corresponds to new look index.
 *
 * \param[in]  System  Look Index
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteSystemLook(uint16_t System);

/**
 * This command gets the current system look.
 *
 * \param[out]  System  Look Index
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadSystemLook(uint16_t *System);

/**
 * Sets the aspect ratio of the displayed image/video.
 *
 * \param[in]  AspectRatio  Aspect Ratio
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteAspectRatio(DLPC654X_DispfmtAspectRatioT_e AspectRatio);

/**
 * Returns the current aspect ratio of the displayed image/video.
 *
 * \param[out]  AspectRatio  Aspect Ratio
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadAspectRatio(DLPC654X_DispfmtAspectRatioT_e *AspectRatio);

/**
 * Gets the resolution of the displayed SFG image.
 *
 * \param[out]  HorizontalResolution  Horizontal resolution of SFG
 * \param[out]  VerticalResolution  Vertical resolution of SFG
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadSfgResolution(uint16_t *HorizontalResolution, uint16_t *VerticalResolution);

/**
 * This command will set one of the pre-defined test patterns stored in flash (configured using DLP Composer tool). The function selects a pattern to load from flash into the test pattern generator hardware. The information retrieved from the flash includes pattern definition, color definition, and the resolution. The pre-defined patterns are included in the flash configuration data. [[#t-display_write]] command must be called to switch the display mode from other modes to TPG prior to or after this command .
 *
 * \param[in]  PatternNumber  Predefined test pattern number to be displayed
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteTpgPredefinedPattern(uint8_t PatternNumber);

/**
 * Returns the current selection for pre-defined test patterns.
 *
 * \param[out]  PatternNumber  Predefined test pattern number selected
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadTpgPredefinedPattern(uint8_t *PatternNumber);

/**
 * Returns DLP Controller ID.
 *
 * \param[out]  ControllerId  Controller ID. (DLPC6540 ID = 0x59)
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadControllerInformation(uint8_t *ControllerId);

/**
 * Returns the 32-bit hard wired DMD device ID.
 *
 * \param[out]  DmdId  DMD(tm) device ID.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDmdDeviceId(uint32_t *DmdId);

/**
 * Returns the DMD width and height in number of pixels and lines respectively.
 *
 * \param[out]  Width  Effective width of DMD in pixels.
 * \param[out]  Height  Effective height of DMD in lines.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDmdResolution(uint16_t *Width, uint16_t *Height);

/**
 * Returns version number that uniquely identifies the flash image.
 *
 * \param[out]  FlashVersionMajor  Flash Version Major
 * \param[out]  FlashVersionMinor  Flash Version Minor
 * \param[out]  FlashVersionSubminor  Flash Version Subminor
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadFlashVersion(uint8_t *FlashVersionMajor, uint8_t *FlashVersionMinor, uint8_t *FlashVersionSubminor);

/**
 * Returns supported Layout revision numbers and hash for flash config and app config layout.
 *
 * \param[out]  FlashCfgLayoutVersion  Flash Cfg Layout Version
 * \param[out]  FlashCfgLayoutHash  Flash Cfg Layout Hash
 * \param[out]  AppCfgLayoutVersion  App Cfg Layout Version
 * \param[out]  AppCfgLayoutHash  App Cfg Layout Hash
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadFlashLayoutVersion(uint16_t *FlashCfgLayoutVersion, uint8_t* *FlashCfgLayoutHash, uint16_t *AppCfgLayoutVersion, uint8_t* *AppCfgLayoutHash);

/**
 * Command to read status information from DLP Controller. If status interrupt is enabled (configurable via default UI tool in DLP Composer), reading back this command will acknowledge/deactivate the interrupt pin until the next change in status.
 *
 * \param[out]  SystemStatus  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadSystemStatus(DLPC654X_SystemStatus_s *SystemStatus);

/**
 * On receipt of this command controller wait for DelayInMilliseconds period before executing the next command. This command to be used in Auto Initialization batchfile configuration. Use this command to insert delay between execution of two commands.
 *
 * \param[in]  DelayInMilliseconds  Delay In Milliseconds
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteGeneralDelayCommand(uint32_t DelayInMilliseconds);

/**
 * Invalidates the user settings portion of EEPROM data or calibration portion of EEPROM data or both as per input arguments and restarts the system. If none of the settings or calibration data is selected, then the command does nothing.
 *
 * \param[in]  EepromInvalidate  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteEepromInvalidate(DLPC654X_EepromInvalidate_s *EepromInvalidate);

/**
 * Captures the current external image displayed on the screen and stores it into the Flash memory as a Splash image.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteSplashCapture();

/**
 * Returns the current status of splash capture.
 *
 * \param[out]  CaptureState  Capture State
 * \param[out]  CaptureCompletionPercentage  Completion Status
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadSplashCaptureStatus(DLPC654X_SplashCaptureStateT_e *CaptureState, uint8_t *CaptureCompletionPercentage);

/**
 * Terminates any ongoing Splash Capture
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteTerminateSplashCapture();

/**
 * This Command Initializes the on the fly splash load operation. It is used to give the information about the image (dimensions, pixel format, compression used and so on) and sets the datapath for the splash load operation. It also resets prior operations. It should be called once before loading an image on the fly.
 *
 * \param[in]  SplashHeader  Header of the pattern Image (same format as the splash header)
 * \param[in]  Width  Width of the splash image, in pixels, to which given pattern is to be replicated
 * \param[in]  Height  Height of the splash image, in pixels, to which given pattern is to be replicated
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteInitializeOnTheFlyLoadSplashImage(uint8_t* SplashHeader, uint16_t Width, uint16_t Height);

/**
 * This Command receives the image data and writes it into the VPS DRAM Buffer. It performs decompression if required. Due to the limitation of USB, the entire image data may not be sent in one go. Hence, this command has to be called multiple times in order to load the entire image. It also stores any bytes that are unprocessed by the API (if the API does not have all the bytes required to perform an operation). These unprocessed bytes are sent over to callback function the next time data is received. The API displays the loaded splash once the entire image is loaded into the memory. The remaining pixels to load the image is also returned by the callback function.
 *
 * \param[in]  ImageDataLength  Byte Length for ImageData
 * \param[in]  ImageData  Image Data
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteLoadSplashImageOnTheFly(uint16_t ImageDataLength, uint8_t* ImageData);

/**
 * Enables 3D functionality.
 *
 * \param[in]  Enable3D  TRUE - Enable Processing,   FALSE - Disable Processing.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteEnableThreeD(bool Enable3D);

/**
 * Returns whether 3D is enabled or not.
 *
 * \param[out]  Enable3D  TRUE - Enable Processing,   FALSE - Disable Processing.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadEnableThreeD(bool *Enable3D);

/**
 * This command configures source of L/R signal is GPIO or internally SW generated.
 *
 * \param[in]  UseLeftRightSignalOnGpio  TRUE L/R Signal on GPIO  FALSE No L/R signal input
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteLeftRightSignalSource(bool UseLeftRightSignalOnGpio);

/**
 * This command tells whether source of L/R signal is GPIO or internally SW generated.
 *
 * \param[out]  UseLeftRightSignalOnGpio  TRUE L/R Signal on GPIO  FALSE No L/R signal input
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadLeftRightSignalSource(bool *UseLeftRightSignalOnGpio);

/**
 * This command inverts the L/R signal polarity.
 *
 * \param[in]  Invert  TRUE  Left / Right Frame are swapped   FALSE  Left / Right Frame are normal
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteLeftRightSignalPolarity(bool Invert);

/**
 * This command tells whether L/R signal polarity is inverted or not.
 *
 * \param[out]  Invert  Invert
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadLeftRightSignalPolarity(bool *Invert);

/**
 * This command sets the orientation number of the actuator position There are 24 possible options 0 - 23; use this commmand while performing XPR calibration using TI provided XPR calibration splash image.
 *
 * \param[in]  XprOrientationNumber  Orientation number. Range 0 - 23.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXpr4WayOrientation(uint8_t XprOrientationNumber);

/**
 * This command retrieves the last set orientation number or the subframe order
 *
 * \param[out]  XprOrientationNumber  Refer software for XPR_Orientation_Number definitions.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXpr4WayOrientation(uint8_t *XprOrientationNumber);

/**
 * This command configure/setup Actuator Waveform Control(AWC) block. Here, AWCx can be AWC 0 or 1. Bytes 2-5 contains the XPR command data as mentioned in Byte 0. While Byte 1 contains AWC channel to be confifure, possible value are 0 or 1.<BR><BR> **Fixed Output Enable** : Configures Actuator in fixed output mode.<BR> Byte    2: 0x00 - Disable 0x01 - Enable  <BR> Bytes 3-5: Reserved must be set to 0x000000<BR><BR> **DAC Gain** : Set Waveform Generator DAC Gain.<BR> Byte    2: Range 0 - 255 format u1.7 (0 to 1.9921875) <BR> Bytes 3-5: Reserved must be set to 0x000000<BR><BR> **Subframe delay** : Subframe delay Bytes 2-5; Range 0 - 262144 and lsb = 133.333ns  <BR><BR> **Actuator Type (READ ONLY)** : Actuator type <BR> Byte    2: <BR>0x00 - NONE <BR> 0x01 - Optotune (XPR-25 Model) <BR> 0x80 - TI Actuator Interface (EEPROM) <BR> 0x81 - TI Actuator Interface (MCU) <BR> Bytes 3-5: Reserved must be set to 0x000000<BR><BR> **Output Enable/Disable** : Actuator output enable/disable <BR> Byte    2: 0x00 - Disable 0x01 - Enable  <BR> Bytes 3-5: Reserved must be set to 0x000000<BR><BR> Note: Both AWC0 and AWC1 disabled/enabled together <BR> **Clock Width** : Defines the high and low width for the output clock; the clock period will be 2*(ClkWidth+1)*8.33ns<BR> Bytes 2-5 : ClkWidth <BR> Example: ClkWidth = 0; will generate clock of 2*(0+1)*8.33 = 16.66ns <BR><BR> **DAC Offset** : DAC Output Offset <BR> Byte    2: Range -128 - +127 format S7 (-128 to +127) <BR> Bytes 3-5: Reserved must be set to 0x000000<BR><BR> **Number of Segments** : Defines number of segments <BR> Byte    2: Range 2 - 255<BR> Bytes 3-5: Reserved must be set to 0x000000<BR><BR> **Segments Length** : Defines size of the segments <BR> Bytes 2-3: Range 19 - 4095<BR> Bytes 4-5: Reserved must be set to 0x0000<BR><BR> **Invert PWM A** : Applicable when AWC is configured to PWM type instead of DAC <BR> Byte    2: 0x00 - No inversion <BR> 0x01 - Inverted  <BR> Bytes 3-5: Reserved must be set to 0x000000<BR><BR> **Invert PWM B** : Applicable when AWC is configured to PWM type instead of DAC <BR> Byte    2: 0x00 - No inversion 0x01 - Inverted  <BR> Bytes 3-5: Reserved must be set to 0x000000<BR><BR> **Subframe Filter Value** : Sets Subframe Filter Value - defines the minimum time between Subframe edges. Edges closer than the set value will be filtered out <BR> Byte    2: 0 = Filter disabled, >0 = Filter time will be Val x 60us, Range: 0 - 255 <BR> Bytes 3-5: Reserved must be set to 0x000000<BR><BR> **Subframe Watch Dog** : Defines the maximum time between Subframe edges; if timer expires, then the WG will automatically output the Fixed Output value, and the normal output will resume on the next subframe edge.<BR> Bytes 2-3: 0 = Subframe watchdog disabled, >0 = Watchdog time will be Time x 60us, Range: Range: 0 - 1023 <BR> Bytes 4-5: Reserved must be set to 0x0000<BR><BR> **Fixed Output Value** : Defines the value to be output on DAC when fixed output mode is selected.<BR> Byte    2: Value to be output on DAC, Range -128 to 127 Bytes 3-5: Reserved must be set to 0x000000<BR><BR>
 *
 * \param[in]  XprCommand  XPR Command
 * \param[in]  AwcChannel  Channel number (0 or 1) of Actuator waveform control for which the command parameter has to be applied
 * \param[in]  Data  Data that needs to be passed to the command
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprActuatorWaveformControlParameter(DLPC654X_Xpr4WayCommand_e XprCommand, uint8_t AwcChannel, uint32_t Data);

/**
 * This command gets the parameter set to the AWC waveform generator.
 *
 * \param[in]  XprCommand  XPR Command
 * \param[in]  AwcChannel  Channel number of Actuator waveform control block for which the command parameter to be readback
 * \param[out]  Data  Parameter value obtained for the command passed
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprActuatorWaveformControlParameter(DLPC654X_Xpr4WayCommand_e XprCommand, uint8_t AwcChannel, uint32_t *Data);

/**
 * This command controls how far open or closed the aperture should be over the light source. The aperture position is adjusted every frame based on the current scene brightness. Note: This function should only be used during calibration and only when the DynamicBlack(tm) II algorithm is disabled.
 *
 * \param[in]  Position  Setting to which aperture should be positioned.   Range = 1 to (Composer configured number of aperture positions).Position = 0 - Aperture is fully closed Position = 0xFFFF - Aperture is fully open
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDbAperturePosition(uint16_t Position);

/**
 * This command gives how far open or closed the aperture is over the light source. The aperture position is adjusted every frame based on the current scene brightness. Note: This function should only be used during calibration and only when the DynamicBlack(tm) II algorithm is disabled.
 *
 * \param[out]  Position  Position
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbAperturePosition(uint16_t *Position);

/**
 * This command controls the minimum and maximum aperture positions. The minimum command typically corresponds to the aperture position for the aperture to be fully closed, while the maximum command typically corresponds to the aperture position for the aperture to be fully open. However, the polarity of these may change depending on the aperture hardware used.
 *
 * \param[in]  AptMin  Position where the aperture is fully open or closed (depending on aperture hardware polarity).   Range = 0 to (Composer configured minimum), and must be less than Apt_Max.
 * \param[in]  AptMax  Position where the aperture is fully open or closed (depending on aperture hardware polarity).   Range = 0 to (Composer configured maximum), and must be greater than Apt_Min.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDbApertureMinMax(uint16_t AptMin, uint16_t AptMax);

/**
 * This command gets the minimum and maximum aperture positions. The minimum command typically corresponds to the aperture position for the aperture to be fully closed, while the maximum command typically corresponds to the aperture position for the aperture to be fully open. However, the polarity of these may change depending on the aperture hardware used.
 *
 * \param[out]  AptMin  Position where the aperture is fully open or closed (depending on aperture hardware polarity).
 * \param[out]  AptMax  Position where the aperture is fully open or closed (depending on aperture hardware polarity).
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbApertureMinMax(uint16_t *AptMin, uint16_t *AptMax);

/**
 * This command enables the DynamicBlack(tm) II Manual Mode of operation. It is intended to be used as a test command. The DB software algorithm will be disabled, and the user can set manual gain values using DB set gain command.
 *
 * \param[in]  Enable  1 = enable manual mode<BR>   0 = disable manual mode
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDbManualMode(uint8_t Enable);

/**
 * This command enables the DynamicBlack(tm) II Manual Mode of operation. It is intended to be used as a test command. The DB software algorithm will be disabled, and the user can set manual gain values using DB set gain command.
 *
 * \param[out]  Enable  1 = enable manual mode<BR>   0 = disable manual mode
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbManualMode(uint8_t *Enable);

/**
 * This command configures area of the DynamicBlack(tm) II border region for the border exclusion function. The border exclusion function allows the user to reduce the letterbox (black border) effect on a primarily bright image where letterbox area reduces the overall scene brightness for the algorithm. It also helps the algorithm better handle images with bright subtitles where the subtitles increase the overall scene brightness. This command will also be used in a multi-ASIC configuration to exclude any image overlap required for other image processing algorithms.
 *
 * \param[in]  Top  number of lines top of border. Range 0 - 4095
 * \param[in]  Bottom  number of lines bottom of border. Range 0 - 4095
 * \param[in]  Left  number of pixels of left border. Range 0 - 2047
 * \param[in]  Right  number of pixels of right border. Range 0 - 2047
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDbBorderConfiguration(uint16_t Top, uint16_t Bottom, uint16_t Left, uint16_t Right);

/**
 * This Command returns the border region area for the DynamicBlack(tm) II border exclusion function.
 *
 * \param[out]  Top  number of lines top of border. Range 0 - 4095
 * \param[out]  Bottom  number of lines bottom of border. Range 0 - 4095
 * \param[out]  Left  number of pixels of left border. Range 0 - 2047
 * \param[out]  Right  number of pixels of right border. Range 0 - 2047
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbBorderConfiguration(uint16_t *Top, uint16_t *Bottom, uint16_t *Left, uint16_t *Right);

/**
 * 
 *
 * \param[in]  BorderWeight  Weight value of border pixels 0 = 0% weighted; 1 = 25% weighted; 2 = 50% weighted; 3 = 75% weighted
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDbBorderWeight(DLPC654X_DbPixelWeightT_e BorderWeight);

/**
 * Sets weight value of the DynamicBlack(tm) II border region for the border exclusion function
 *
 * \param[out]  BorderWeight  Weight value of border pixels 0 = 0% weighted; 1 = 25% weighted; 2 = 50% weighted; 3 = 75% weighted
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbBorderWeight(DLPC654X_DbPixelWeightT_e *BorderWeight);

/**
 * This command returns currently configured number of steps to allow the DynamicBlack(tm) II aperture to move.
 *
 * \param[in]  ClipPixels  Number of pixels that can be clipped.   Range = 0 to 65535.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDbClipPixels(uint16_t ClipPixels);

/**
 * This command returns the currently selected number of pixels that can be clipped.
 *
 * \param[out]  ClipPixels  Clip Pixels
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbClipPixels(uint16_t *ClipPixels);

/**
 * This command controls the DynamicBlack(tm) II gain value. Typical value range is 1.0 to 8.0. Manual Mode needs to be enabled to set the gain as it will override the gain value that is calculated every frame.
 *
 * \param[in]  DbGain  Gain value.   Typical value range is 1.0 to 8.0.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDbGain(double DbGain);

/**
 * This command gets the DynamicBlack(tm) II gain value. Typical value range is 1.0 to 8.0
 *
 * \param[out]  DbGain  Gain value. Typical value range is 1.0 to 8.0.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbGain(double *DbGain);

/**
 * This command sets the number of steps to allow the DynamicBlack(tm) II aperture to move. This allows the number of steps to be adjusted dynamically so that the results can be visually compared.
 *
 * \param[in]  Steps  Number of steps to allow aperture to move.   Range = 1 to (Composer configured number of aperture positions).
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDbNumSteps(uint16_t Steps);

/**
 * This command returns currently configured number of steps to allow the DynamicBlack(tm) II aperture to move.
 *
 * \param[out]  Steps  Steps
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbNumSteps(uint16_t *Steps);

/**
 * This command specifies the number of positions that the DynamicBlack(tm) aperture can move in one frame.
 *
 * \param[in]  Speed  Number of positions that the aperture can move in one frame.   Range = 1 to 256.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDbApertureSpeed(uint16_t Speed);

/**
 * This command returns the number of positions that the DynamicBlack(tm) II aperture can move in one frame.
 *
 * \param[out]  Speed  Speed
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbApertureSpeed(uint16_t *Speed);

/**
 * This function controls the DynamicBlack(tm) II strength. A higher strength value will cause DynamicBlack(tm) II to be more aggressive in closing the aperture for a given scene brightness. The aperture min and max positions have to be set prior to calling this function. The aperture movement strength range is 0 to 3, with a recommended value of 2.
 *
 * \param[in]  Strength  Aperture movement strength setting.   Range = 0 to 3, with a recommended value of 2.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDbStrength(uint8_t Strength);

/**
 * This function returns the DynamicBlack(tm) II strength setting.
 *
 * \param[out]  Strength  Strength
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbStrength(uint8_t *Strength);

/**
 * This command returns the start address of the DynamicBlack(tm) II (DB) histogram data. The histogram contains scene brightness data from the previous frame. The DB histogram contains 34 bins measuring non-overlapping intensity ranges in the displayed image. The value of each bin equals the number of pixels within the bin's intensity range. Each pixel's intensity is calculated as the maximum of its red, green, and blue values. In other words, pixel intensity = MAX( R, G, B ). Each pixel has a format of unsigned 8.8, making 16 bit values. Bins 32 and 33 are special bins that represent pixels that have values of exactly zero and only fractional values respectively. This function can be used independently of aperture control for image improvement in dark scenes.
 *
 * \param[out]  HistPtr  Start address of the DB histogram array. Array size is 34. The LSB of each bin represents 32 pixels. Each bin saturates at 0x0003FFFF.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbHistogram(uint8_t *HistPtr);

/**
 * This function returns the current aperture position.  This position indicates how far open or closed the aperture is over the light source. The aperture position is adjusted every frame based on the current scene brightness.
 *
 * \param[out]  Position  Position
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDbCurrentApertPos(uint32_t *Position);

/**
 * Gets x,y coordinates of system's current white point. WPC should be initialized and calibration data should be set before calling
 *
 * \param[out]  ChromaticX  Chromatic x coordinate in (Transmitted in u1.15 format)
 * \param[out]  ChromaticY  Chromatic y coordinate in (Transmitted in u1.15 format)
 * \param[out]  LuminenceY  Luminence Y coordinate
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadCurrentLedColorPoint(double *ChromaticX, double *ChromaticY, uint32_t *LuminenceY);

/**
 * Searches for and Sets Optimal Duty cycle for Corerct Led White Point. Sensor calibration Data should be set before using this command. @retval PASS Successful @retval WPC_ERR_CALIBRATION_DATA_NOT_SET WPC calibration data not set
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteWpcOptimalDutyCycle();

/**
 * Gets Ideal Duty Cycle for Current Target Color Point and the closest Duty Cycle Avaialable. Sensor calibration Data should be set before using this command.
 *
 * \param[out]  WpcOptimalDutyCycle  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadWpcOptimalDutyCycle(DLPC654X_WpcOptimalDutyCycle_s *WpcOptimalDutyCycle);

/**
 * Returns Output of Integrating Sensor for Red, Blue and Green
 *
 * \param[out]  Red  Red
 * \param[out]  Green  Green
 * \param[out]  Blue  Blue
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadWpcSensorOutput(uint32_t *Red, uint32_t *Green, uint32_t *Blue);

/**
 * Set Maximum Drive Level
 *
 * \param[in]  Color  Color
 * \param[in]  DriveLevel  Drive Level
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteMaximumSsiDriveLevel(DLPC654X_SsiDrvColorT_e Color, uint16_t DriveLevel);

/**
 * Get Maximum Drive Level
 *
 * \param[in]  Color  Color
 * \param[out]  DriveLevel  Drive Level
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadMaximumSsiDriveLevel(DLPC654X_SsiDrvColorT_e Color, uint16_t *DriveLevel);

/**
 * Set enable mask for debug messages. The mask identifies the sources of debug messages which are to be enabled for printing at the UART debug port. The mask bit corresponding to the source has to be set to enable it.
 *
 * \param[in]  DebugMessageMask  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDebugMessageMask(DLPC654X_DebugMessageMask_s *DebugMessageMask);

/**
 * Retrieves the current debug message mask. The mask decides which sources of debug messages are enabled. A value of 1 in the mask bit corresponding to a source means that the source is enabled.
 *
 * \param[out]  DebugMessageMask  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDebugMessageMask(DLPC654X_DebugMessageMask_s *DebugMessageMask);

/**
 * Enables or disables the USB logging of messages. When USB logging is enabled, UART logging is stopped.
 *
 * \param[in]  Enable  1 = Enable debug log on USB port <BR>0 = Disable debug log on USB port
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteEnableUsbDebugLog(uint8_t Enable);

/**
 * Writes data to EEPROM connected to the controller. <BR> The EEPROM holds settings and calibration data. The primary purpose of this function is for the user to write to areas of EEPROM outside of the settings and calibration data. If user wants to overwrite settings/calibration data, the password sent with the command should match the expected password. This is a protection mechanism to prevent accidental overwrite of settings/calibration data.
 *
 * \param[in]  Index  Index of the memory to write to (0 means first byte in EEPROM, 1 means second byte and so on)
 * \param[in]  Size  Number of bytes to be written
 * \param[in]  Pwd  EEPROM password if the user needs to write to the TI specified memory space
 * \param[in]  Data  Data
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteEepromMemory(uint16_t Index, uint16_t Size, uint32_t Pwd, uint8_t* Data);

/**
 * This function reads data from EEPROM connected to the Controller which has settings and calibration data.
 *
 * \param[in]  Index  Index of the memory to read from
 * \param[in]  Size  Number of bytes to read
 * \param[out]  Data  Data
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadEepromMemory(uint16_t Index, uint16_t Size, uint8_t *Data);

/**
 * Command that writes specified value to the specified register address. Refer to DLPA30005 datasheet for register descriptions. http://www.ti.com/lit/gpn/dlpa3005.
 *
 * \param[in]  RegisterAddress  Register Address
 * \param[in]  RegisterValue  Register Value
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDlpa3005Register(uint8_t RegisterAddress, uint8_t RegisterValue);

/**
 * Returns specified register value from DLPA3005. Refer to DLPA30005 datasheet for register descriptions. http://www.ti.com/lit/gpn/dlpa3005.
 *
 * \param[in]  RegisterAddress  Register Address
 * \param[out]  RegisterValue  Register Value
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDlpa3005Register(uint8_t RegisterAddress, uint8_t *RegisterValue);

/**
 * Command used to query actuator related information for debugging purpose. Use this command to retrieve information when actuator not running or system is in standby state.<BR>
 *
 * \param[in]  TiActQueryType  Query type <BR>0 = Query N number of bytes from offset address provided in next two bytes i.e., Bytes 1-2 <BR> 1 = Query Actuator information also print on UART debug port <BR> 2 = Query AWG Data Set for index number provided in next two bytes i.e., Bytes 1-2<BR> 3 = Query AWG Edge table header for index number in next two bytes i.e., Bytes 1-2<BR>
 * \param[in]  TiActAddr  Query type provided in Byte 0; not applicable when Query type = 1
 * \param[in]  TiActNumData  Number of bytes to be read when Query type = 0. Note maximum 32 bytes can be read at a time.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteTiActuatorInterfaceDebug(uint8_t TiActQueryType, uint16_t TiActAddr, uint16_t TiActNumData);

/**
 * Command returns queried data as per the settings made in the set command
 *
 * \param[out]  ActuatorData  Actuator Data
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadTiActuatorInterfaceDebug(uint8_t *ActuatorData);

/**
 * Power up or down the DMD.
 *
 * \param[in]  Enable  0 = Disable <BR>1 = Enable
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDmdPower(bool Enable);

/**
 * Returns DMD power enable state
 *
 * \param[out]  Enable  0 = Disable <BR>1 = Enable
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDmdPower(bool *Enable);

/**
 * Parks/Unparks DMD
 *
 * \param[in]  Park  1 = Park <BR>0 = Unpark
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDmdPark(bool Park);

/**
 * Returns 1 if DMD is Parked, else returns 0
 *
 * \param[out]  Park  1 = DMD is parked <br> 0 = DMD is unparked
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDmdPark(bool *Park);

/**
 * 
 *
 * \param[in]  GlobalResetEnable  1 = True Global Reset Mode Enabled 0 = True Global Reset Mode Disabled
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDmdTrueGlobalReset(bool GlobalResetEnable);

/**
 * 
 *
 * \param[out]  GlobalResetEnable  1 = True Global Reset Mode Enabled 0 = True Global Reset Mode Disabled
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDmdTrueGlobalReset(bool *GlobalResetEnable);

/**
 * 
 *
 * \param[out]  Ssize  ssize
 * \param[out]  Sused  sused
 * \param[out]  Sfree  sfree
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadIntStack(uint32_t *Ssize, uint32_t *Sused, uint32_t *Sfree);

/**
 * Prints, on UART, Information of all tasks defined/created with RTOS @retval PASS Successful
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WritePrintAllTaskInformation();

/**
 * Gives the maximum RTOS resource usage by the application.
 *
 * \param[out]  Resource  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadResource(DLPC654X_Resource_s *Resource);

/**
 * Reports Vx1 source interface status.
 *
 * \param[out]  Vx1HwStatus  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadVx1HwStatus(DLPC654X_Vx1HwStatus_s *Vx1HwStatus);

/**
 * This commands toggles current power mode from standby to active or from active to power down. The Standby state corresponds to Low Power Mode.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WritePower();

/**
 * Returns current system power state.
 *
 * \param[out]  PowerState  Power State
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadPower(DLPC654X_PowerstateEnum_e *PowerState);

/**
 * Displays the specified source.
 *
 * \param[in]  Source  Source
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDisplay(DLPC654X_CmdProjectionModes_e Source);

/**
 * Returns the source which is currently being displayed.
 *
 * \param[out]  Source  The source currently being displayed.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDisplay(DLPC654X_CmdProjectionModes_e *Source);

/**
 * Draws a border around the test pattern of given width and color.
 *
 * \param[in]  Width  Width of the Border
 * \param[in]  BorderColorRed  Border Color Red Value
 * \param[in]  BorderColorGreen  Border Color Green Value
 * \param[in]  BorderColorBlue  Border Color Blue Value
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteTpgBorder(uint8_t Width, uint16_t BorderColorRed, uint16_t BorderColorGreen, uint16_t BorderColorBlue);

/**
 * Returns Width in number of pixels and Color of Border for a test Pattern.
 *
 * \param[out]  Width  Width of the Border
 * \param[out]  BorderColorRed  Border Color Red Value
 * \param[out]  BorderColorGreen  Border Color Green Value
 * \param[out]  BorderColorBlue  Border Color Blue Value
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadTpgBorder(uint8_t *Width, uint16_t *BorderColorRed, uint16_t *BorderColorGreen, uint16_t *BorderColorBlue);

/**
 * Sets horizontal and vertical resolution in number of pixels for current test pattern.
 *
 * \param[in]  HorizontalResolution  Horizontal resolution of test pattern
 * \param[in]  VerticalResolution  Vertical resolution of test pattern
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteTpgResolution(uint16_t HorizontalResolution, uint16_t VerticalResolution);

/**
 * Returns horizontal and vertical resolution in number of pixels for current test pattern.
 *
 * \param[out]  HorizontalResolution  Horizontal resolution of test pattern
 * \param[out]  VerticalResolution  Vertical resolution of test pattern
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadTpgResolution(uint16_t *HorizontalResolution, uint16_t *VerticalResolution);

/**
 * Sets frame rate in Hz for current test pattern.
 *
 * \param[in]  FrameRate  Frame rate of test pattern
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteTpgFrameRate(uint8_t FrameRate);

/**
 * Returns frame rate in Hz for current test pattern.
 *
 * \param[out]  FrameRate  Frame rate of test pattern
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadTpgFrameRate(uint8_t *FrameRate);

/**
 * Configures the solid color to be displayed when display is set to solid field generator (SFG). This command only sets the SFG color and does NOT display it. In order to display the SFG, Display needs to be set with SFG as source(Use [[#t-display_write]] command).
 *
 * \param[in]  Red  Red color level.
 * \param[in]  Green  Green color level.
 * \param[in]  Blue  Blue color level.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteSfgColor(uint16_t Red, uint16_t Green, uint16_t Blue);

/**
 * Returns the solid color which is programmed to be displayed when display is set to SFG.
 *
 * \param[out]  Red  Red color level.
 * \param[out]  Green  Green color level.
 * \param[out]  Blue  Blue color level.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadSfgColor(uint16_t *Red, uint16_t *Green, uint16_t *Blue);

/**
 * Command to set the color to be  used in curtain mode. Use [[#t-display_write]] command to switch to curtain mode.
 *
 * \param[in]  Color  The background color to be set as curtain.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteCurtainColor(DLPC654X_DispBackgroundColorT_e Color);

/**
 * Command that returns the color used in curtain mode.
 *
 * \param[out]  Color  The background color set as curtain. This returns the background colorof the display even if the curtain is not enabled.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadCurtainColor(DLPC654X_DispBackgroundColorT_e *Color);

/**
 * Sets the index of the splash image to be loaded and displayed. This command will only set the Splash image source and NOT display it. In order to display this splash, [[#t-display_write]] has to be called with source as Splash.
 *
 * \param[in]  Index  The 0-based index of Splash Image (0xff for captured splash).
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteSplashLoadImage(uint8_t Index);

/**
 * Gets the index of the splash image to be loaded and displayed.
 *
 * \param[out]  Index  The 0-based index of Splash Image (0xff for captured splash).
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadSplashLoadImage(uint8_t *Index);

/**
 * Flips the data output to the display vertically or horizontally. This feature is provided to support ceiling mount and rear projection use cases..
 *
 * \param[in]  VertFlip  1 = Vertical Flip of the image is enabled <BR>0 = Vertical Flip of the image is disabled
 * \param[in]  HorzFlip  1 = Horizontal Flip of the image is enabled<BR>0 = Horizontal Flip of the image is disabled
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteEnableImageFlip(bool VertFlip, bool HorzFlip);

/**
 * Returns whether image flipping is enabled or not.
 *
 * \param[out]  VertFlip  1 = Vertical Flip of the image is enabled <BR>0 = Vertical Flip of the image is disabled
 * \param[out]  HorzFlip  1 = Horizontal Flip of the image is enabled<BR>0 = Horizontal Flip of the image is disabled
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadEnableImageFlip(bool *VertFlip, bool *HorzFlip);

/**
 * It enables or disables display freeze which freezes the current frame being displayed on the screen.
 *
 * \param[in]  Freeze  1 = Display freeze is enabled <BR>0 = Display freeze is disabled
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteEnableFreeze(bool Freeze);

/**
 * Returns whether the current display is frozen or not.
 *
 * \param[out]  Freeze  1 = Display freeze is enabled <BR>0 = Display freeze is disabled
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadEnableFreeze(bool *Freeze);

/**
 * Configures the Keystone correction when the pitch, yaw, roll, throw ratio and vertical offset of corrected image are known. Keystone correction is used to remove the distortion caused when the projector is not orthogonal to the projection surface (screen). Keystone feature will be automatically enabled when this command is executed.
 *
 * \param[in]  Pitch  Pitch angle in degrees
 * \param[in]  Yaw  Yaw angle in degrees <br> Set to 0 for 1D correction
 * \param[in]  Roll  Roll angle in degrees <br> Set to 0 for 1D/2D correction
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteKeystoneAngles(double Pitch, double Yaw, double Roll);

/**
 * Returns the keystone configuration parameters currently set.
 *
 * \param[out]  Pitch  Pitch
 * \param[out]  Yaw  Yaw
 * \param[out]  Roll  Roll
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadKeystoneAngles(double *Pitch, double *Yaw, double *Roll);

/**
 * 
 *
 * \param[in]  ThrowRatio  Throw Ratio
 * \param[in]  VerticalOffset  Vertical Offset
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteKeystoneConfigOverride(double ThrowRatio, double VerticalOffset);

/**
 * 
 *
 * \param[out]  ThrowRatio  Throw Ratio
 * \param[out]  VerticalOffset  Vertical Offset
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadKeystoneConfigOverride(double *ThrowRatio, double *VerticalOffset);

/**
 * Configures the cropping of input image and resizing of image that is displayed. Cropped area can be equal to or less than the input image size. The display area has has to be within DMD effective number of pixels and lines. Note: - sending this command would override the settings made through Set Aspect Ratio command. <BR> - For TPG, SFG and Splash, croppedArea parameter is ignored. For those sources, cropped area is automatically set as explained below: <BR> - For TPG, cropped area is set to TPG resolution. <BR> - For Splash, cropped area is set to Splash image size. <BR> - For SFG, cropped area is set to SFG resolution which is equal to source area of last stable external source or TPG. <BR>
 *
 * \param[in]  DisplayImageSize  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDisplayImageSize(DLPC654X_DisplayImageSize_s *DisplayImageSize);

/**
 * Returns current cropping and display settings.
 *
 * \param[out]  DisplayImageSize  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDisplayImageSize(DLPC654X_DisplayImageSize_s *DisplayImageSize);

/**
 * Configures the characteristics of the source on the Current active port.
 *
 * \param[in]  SourceConfiguration  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteSourceConfiguration(DLPC654X_SourceConfiguration_s *SourceConfiguration);

/**
 * Retrieves the source characteristics for the current active port.
 *
 * \param[out]  SourceConfiguration  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadSourceConfiguration(DLPC654X_SourceConfiguration_s *SourceConfiguration);

/**
 * This command provides user control to relock to a source or to start/stop autolock algorithm.
 *
 * \param[in]  Action  0 = Resync <BR> 1 = Autolock Algorithm Start <BR> 2 = Autolock Algorithm Stop
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteAutolockSetup(uint8_t Action);

/**
 * Returns Current status of source detection.
 *
 * \param[out]  ScanStatus  Scan Status
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDatapathScanStatus(DLPC654X_DpScanStatus_e *ScanStatus);

/**
 * Configures the characteristics of the Vx1 source.
 *
 * \param[in]  DataMapMode  Data Map Mode
 * \param[in]  ByteMode  Byte Mode
 * \param[in]  NumberOfLanes  Number of lanes can be 1 or 2 or 4 or 8
 * \param[in]  EnablePixelRepeat  Enable Pixel Repeat
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteVboConfiguration(DLPC654X_SrcVboDataMapModeT_e DataMapMode, DLPC654X_SrcVboByteModeT_e ByteMode, uint8_t NumberOfLanes, bool EnablePixelRepeat);

/**
 * Retruns the characteristics of the Vx1 source.
 *
 * \param[out]  DataMapMode  Data Map Mode
 * \param[out]  ByteMode  Byte Mode
 * \param[out]  NumberOfLanes  Number Of Lanes
 * \param[out]  EnablePixelRepeat  Enable Pixel Repeat
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadVboConfiguration(DLPC654X_SrcVboDataMapModeT_e *DataMapMode, DLPC654X_SrcVboByteModeT_e *ByteMode, uint8_t *NumberOfLanes, bool *EnablePixelRepeat);

/**
 * Configures the characteristics of the FPD source.
 *
 * \param[in]  FpdConfiguration  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteFpdConfiguration(DLPC654X_FpdConfiguration_s *FpdConfiguration);

/**
 * Retruns the characteristics of the FPD source.
 *
 * \param[out]  FpdConfiguration  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadFpdConfiguration(DLPC654X_FpdConfiguration_s *FpdConfiguration);

/**
 * This command writes to the warp map table that can be enabled using the [[#t-configure_manual_warp_write]] command. N warp map points (that does not exceed the command packet size) can be loaded at a time to anywhere within the table. Maximum table size is 1952.
 *
 * \param[in]  WarpTableIndex  Start index in the table for the data to be written
 * \param[in]  WarpPointsLength  Byte Length for WarpPoints
 * \param[in]  WarpPoints  Warp map points in X, Y pairs where X, Y are are in 13.3 fixed point format
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteManualWarpTable(uint16_t WarpTableIndex, uint16_t WarpPointsLength, uint16_t WarpPoints[]);

/**
 * This command reads from the warp map table already loaded using [[#t-configure_manual_warp_write]] command. N warp map points (that does not exceed the command packet size) can be read at a time from anywhere within the table. Maximum table size is 1952.
 *
 * \param[in]  WarpTableIndex  Start index in the table from which the data is to be read
 * \param[in]  NumEntries  Number of entries to be read
 * \param[out]  WarpPoints  Warp map points in X, Y pairs where X, Y are in 13.3 fixed point format
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadManualWarpTable(uint16_t WarpTableIndex, uint16_t NumEntries, uint16_t *WarpPoints[]);

/**
 * This command sets up the user defined control points of the warp map that shall be applied on top of the keystone correction, anamorphic scaling and other warp dependent feature settings if they are enabled. The warping map table loaded by the [[#t-manual_warp_table_write]] command is used as a two dimensional array with dimension which is defined based on the argument AreControlPointsEvenlySpaced of this command: - TRUE  = (WarpColumns x WarpRows) - FALSE = (WRP_CTL_POINTS_X x WRP_CTL_POINTS_Y) The points in the map should lie within the display area defined by [[#t-disp_image_size_command]]. Any points lying outside the display area shall get cropped.
 *
 * \param[in]  HorizontalCtrlPointsLength  Byte Length for HorizontalCtrlPoints
 * \param[in]  VerticalCtrlPointsLength  Byte Length for VerticalCtrlPoints
 * \param[in]  ManualWarpControlPoints  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteManualWarpControlPoints(uint16_t HorizontalCtrlPointsLength, uint16_t VerticalCtrlPointsLength, DLPC654X_ManualWarpControlPoints_s *ManualWarpControlPoints);

/**
 * This command gets up the user defined warping map control points.
 *
 * \param[out]  HorizontalCtrlPoints  Warp Horizontal control points position array. <BR>Number of points in this array equal to WRP_CTL_POINTS_X.
 * \param[out]  VerticalCtrlPoints  Warp Vertical control points position array <BR>Number of points in this array equal to WRP_CTL_POINTS_Y.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadManualWarpControlPoints(uint16_t *HorizontalCtrlPoint, uint16_t *VerticalCtrlPoints);

/**
 * This command applies the manual warping control points and map table to the HW defined by [[#t-configure_manual_ctrl_points]] and [[#t-configure_manual_warp_write]] respectively.
 *
 * \param[in]  Enable  TRUE  = Manual warping is enabled and applied. <BR>FALSE = Manual warping is disabled.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteApplyManualWarping(uint8_t Enable);

/**
 * This command returns if manual warping is enabled or disabled.
 *
 * \param[out]  Enable  0 = Manual warping is disabled <BR>1 = Manual warping is enabled.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadManualWarpingEnabled(uint8_t *Enable);

/**
 * This command sets up the user defined 3x3 warping map that shall override the keystone correction, anamorphic scaling and other warping feature settings. a 3x3 warp map table given by the parameter of this command is used but creates a smooth warping output by interpolating the intermediate points. To disable this feature use ConfigureWarpMap command.
 *
 * \param[in]  WarpPointsLength  Byte Length for WarpPoints
 * \param[in]  WarpPoints  3x3 Wamp map points
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteConfigureSmoothWarp(uint16_t WarpPointsLength, uint16_t WarpPoints[]);

/**
 * This command sets Manula Warp Table write Mode
 *
 * \param[in]  Mode  0 = Overwrite Existing <BR>1 = Merge with Existing
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteManualWarpTableUpdateMode(uint8_t Mode);

/**
 * This command returns if manual warping is enabled or disabled.
 *
 * \param[out]  Mode  0 = Overwrite Existing <BR>1 = Merge with Existing
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadManualWarpTableUpdateMode(uint8_t *Mode);

/**
 * Enables the illumination system. In color wheel based lamp or laser systems, the illumination will be turned on only if the system has indicated that the color wheel is spinning.  This interlock is necessary to avoid burning the coatings off the glass, but is less of an issue with printed wheels.
 *
 * \param[in]  Enable  0 - Disabled <BR>1 - Only Red LED Enabled <BR> 2 - Only Green LED Enabled <BR> 3 - Red and Green LEDs Enabled <BR> 4 - Only Blue LED Enabled <BR> 5 - Red and Blue LEDs Enabled <BR> 6 - Green and Blue LEDs Enabled <BR> 7 - All LEDs Enabled <BR>
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteIlluminationEnable(uint8_t Enable);

/**
 * Gets the enable state of lamp/SSI illumination.
 *
 * \param[out]  Enable  0 - Disabled <BR>1 - Only Red LED Enabled <BR> 2 - Only Green LED Enabled <BR> 3 - Red and Green LEDs Enabled <BR> 4 - Only Blue LED Enabled <BR> 5 - Red and Blue LEDs Enabled <BR> 6 - Green and Blue LEDs Enabled <BR> 7 - All LEDs Enabled <BR>
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadIlluminationEnable(uint8_t *Enable);

/**
 * Sets DLPA3005 Drive Current Levels input as 10-bit DriveLevel per LED in the range 0 – 874. Current output in Amps is calculated as described below. <BR> OutputCurrent = ((DriveLevel + 1)/1024)*((0.15/0.004)) Amps <BR> Example: For DriveLevel = 874; OutputCurrent = 32.04345703Amps <BR>
 *
 * \param[in]  DriveLevelRed  Drive Level Red
 * \param[in]  DriveLevelGreen  Drive Level Green
 * \param[in]  DriveLevelBlue  Drive Level Blue
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteDlpa3005IlluminationCurrent(uint16_t DriveLevelRed, uint16_t DriveLevelGreen, uint16_t DriveLevelBlue);

/**
 * Gets DLPA3005 Drive Current Levels.
 *
 * \param[out]  DriveLevelRed  Drive Level Red
 * \param[out]  DriveLevelGreen  Drive Level Green
 * \param[out]  DriveLevelBlue  Drive Level Blue
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDlpa3005IlluminationCurrent(uint16_t *DriveLevelRed, uint16_t *DriveLevelGreen, uint16_t *DriveLevelBlue);

/**
 * Sets the drive current of all the channels of the SSI based illumination. Appicable for DLPA3000 or PWM based LED drivers
 *
 * \param[in]  PwmGroup  Applicable only for PWM based LED driver
 * \param[in]  SsiDriveLevels  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteSsiDriveLevels(uint8_t PwmGroup, DLPC654X_SsiDriveLevels_s *SsiDriveLevels);

/**
 * Gets the drive current of all the channels of the SSI based illumination.  Appicable for DLPA3000 or PWM based LED drivers
 *
 * \param[in]  PwmGroup  Applicable only for PWM based LED driver
 * \param[out]  SsiDriveLevels  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadSsiDriveLevels(uint8_t PwmGroup, DLPC654X_SsiDriveLevels_s *SsiDriveLevels);

/**
 * Sets enable flag for all Image Algorithms.<BR> 0 = Disable <BR> 1 = Enable  <BR> **Chroma Transient Improvement** : <BR> This function enables/disables the Chroma Transient Improvement (CTI) function which filters the 4:4:4 sampled, chrominance (Cr and Cb) data on the B and C data channels. The chroma transient functions performs band pass filtering (supports two center frequencies) and median filtering for ringing minimization. It performs limiting and coring functions for the filtered output. <BR> **Gamma Correction** : <BR> This function enables/disables the Gamma Correction function which implements the removal of gamma transfer function applied at the source, via table lookup process called de-gamma. When enabled, perform de-gamma translation of the 10-bit RGB input to the common 12-bit floating point (S0M8E4) RGB output. When disabled, the full 10 bits of each data input to the Gamma Correction function are zero padded and MSB-aligned to 12-bits and passed through unmodified.<BR> **Color Coordinate Adjustment** : <BR> This function enables/disables the Spatially Adaptive Seven Primaries Color Correction Function Enable. When Disable forces 3x3 CSC (Color Space Conversion) with identity. <BR> **Brilliant Color** : <BR> This function enables/disables the BrilliantColor technology, Brilliant Color uses up to five colors, instead of just the three primary colors, red, green and blue, to improve color accuracy and brightens of secondary colors. This results in a new level of color performance that increases the brightness of the colors. <BR> **White Point Correction** : <BR> This function enables/disables the White Point Correction, typically used on LED type illumination systems. Sometimes due to increase in LED operating temperature or LED aging the LEDs output wavelentgh drifts, therefore white point of the system shifts. This algorithm using active light sensor feedback and factory calibrated values help maintaing white point of the system. <BR> **Dynamic Black** : <BR> Dynamic Black (DB) is an algorithm that reduces the amount of light reaching the projection path by means of LED output power through current control and compensates for reduced light by gaining up the RGB signals. <BR> **HDR** : <BR> High Dynamic Range (HDR) is an algorithm that maps wider brightness and color range of HDR source to the projector display range. HDR is affected by several factors such as illimuniation characterstics, duty cycle distribution and current running sequence. A valid HDR source should be set by HDR_SetHdrSourceConfiguration() before enabling HDR processing.<BR>
 *
 * \param[in]  ImageAlgorithmEnable  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageAlgorithmEnable(DLPC654X_ImageAlgorithmEnable_s *ImageAlgorithmEnable);

/**
 * Returns enable flag for all Image Algorithms <BR> '0' - Disabled or algorithm feature not available. <BR> '1' - Enabled
 *
 * \param[out]  ImageAlgorithmEnable  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageAlgorithmEnable(DLPC654X_ImageAlgorithmEnable_s *ImageAlgorithmEnable);

/**
 * The brightness control provides the ability to add or subtract a fixed bias from each of the input channels.  This may be used to remove any inherent offsets and/or adjust the brightness level. The brightness coefficients are signed, 11-bit (s8.2), 2's complement values between -256 and 255.75, inclusive.  Brightness Control is used after color space conversion.
 *
 * \param[in]  BrightnessAdjustment  Brightness Adjustment
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageBrightness(double BrightnessAdjustment);

/**
 * Returns Image Brightness Level.
 *
 * \param[out]  BrightnessAdjustment  Brightness Adjustment
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageBrightness(double *BrightnessAdjustment);

/**
 * Sets Image Contrast in percentage. Each contrast byte controls the gain applied to the input image data for a given data channel.  The contrast gain has a range from 0 to 200 (0% to 200%) with 100 (100%) being nominal (default).
 *
 * \param[in]  Contrast  Contrast
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageContrast(uint16_t Contrast);

/**
 * Returns Image Contrast in percentage.
 *
 * \param[out]  Contrast  Contrast
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageContrast(uint16_t *Contrast);

/**
 * Sets Image Hue Adjustment angle in degrees and Color Control Gain in percentage.
 *
 * \param[in]  HueAdjustmentAngle  Hue Adjustment Angle
 * \param[in]  ColorControlGain  Color Control Gain
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageHueAndColorControl(int8_t HueAdjustmentAngle, uint16_t ColorControlGain);

/**
 * Returns Image Hue Adjustment angle in degrees and Color Control Gain in percentage.
 *
 * \param[out]  HueAdjustmentAngle  Hue Adjustment Angle
 * \param[out]  ColorControlGain  Color Control Gain
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageHueAndColorControl(int8_t *HueAdjustmentAngle, uint16_t *ColorControlGain);

/**
 * Configures the sharpness filter.  A value of 0 is the least sharp (most smooth), while a value of 31 is the most sharp. This filter is in the back end of the data path, so both video and graphics are affected. TI recommends that the sharpness filters be disabled (sharpness=16) for graphics sources.
 *
 * \param[in]  Sharpness  Sharpness value to apply.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageSharpness(uint8_t Sharpness);

/**
 * Returns the current sharpness value
 *
 * \param[out]  Sharpness  Sharpness value to apply.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageSharpness(uint8_t *Sharpness);

/**
 * Offsets the levels of the RGB channels at a point in the datapath after the following image processing functions have been applied - source offset,  contrast, RGB Gain, brightness and color space conversion (including hue and color adjustment).
 *
 * \param[in]  RedOffset  Red channel offset setting.
 * \param[in]  GreenOffset  Green channel offset setting.
 * \param[in]  BlueOffset  Blue channel offset setting.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageRgbOffset(double RedOffset, double GreenOffset, double BlueOffset);

/**
 * Returns Red, Green and Blue channel offset settings.
 *
 * \param[out]  RedOffset  Red channel offset setting.
 * \param[out]  GreenOffset  Green channel offset setting.
 * \param[out]  BlueOffset  Blue channel offset setting.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageRgbOffset(double *RedOffset, double *GreenOffset, double *BlueOffset);

/**
 * Adjusts individual R, G and B gains of the source image. Gain is specified as a percentage from 0% - 200%, with 100% being nominal (no gain change).  0% will zero out the channel. This function adjusts R, G and B gains by altering the Color Space Converter (CSC) coefficients. This function is only applicable to RGB sources.
 *
 * \param[in]  RedGain  Red channel gain setting.
 * \param[in]  GreenGain  Green channel gain setting.
 * \param[in]  BlueGain  Blue channel gain setting.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageRgbGain(uint16_t RedGain, uint16_t GreenGain, uint16_t BlueGain);

/**
 * Returns gain setting for Red, Green and Blue color channels in percentage.
 *
 * \param[out]  RedGain  Red channel gain setting.
 * \param[out]  GreenGain  Green channel gain setting.
 * \param[out]  BlueGain  Blue channel gain setting.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageRgbGain(uint16_t *RedGain, uint16_t *GreenGain, uint16_t *BlueGain);

/**
 * Sets the Color Space Conversion Matrix with one of the CSC tables stored in flash.
 *
 * \param[in]  Index  Index of the pre-defined CSC table in flash.A value of SRC_CSC_MAXTABLE indicates that no table(from the flash) is being used.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteCscTable(DLPC654X_SrcCscTablesT_e Index);

/**
 * Gets the index of the Color Space Conversion Matrix that is currently cofigured for use.
 *
 * \param[out]  Index  Index of the pre-defined CSC table.A value of SRC_CSC_MAXTABLE indicates that no table(from the flash) is being used.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadCscTable(DLPC654X_SrcCscTablesT_e *Index);

/**
 * This Command allows independent adjustment of the primary, secondary and white coordinates. This call will override any CCA settings performed by prior calls.
 *
 * \param[in]  ImageCcaCoordinates  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageCcaCoordinates(DLPC654X_ImageCcaCoordinates_s *ImageCcaCoordinates);

/**
 * Returns the current color coordinate configuration.
 *
 * \param[out]  ImageCcaCoordinates  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageCcaCoordinates(DLPC654X_ImageCcaCoordinates_s *ImageCcaCoordinates);

/**
 * This command applies the given hue, saturation and gain values for all colors. It does not affect colors having a gain of zero. Note: This call will override any CCA settings performed by prior calls.
 *
 * \param[in]  ImageHsg  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageHsg(DLPC654X_ImageHsg_s *ImageHsg);

/**
 * This command returns the currently applied hue, saturation and gain values for all the colors. If gain for a color is zero then the HSG is not applied on the color.
 *
 * \param[out]  ImageHsg  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageHsg(DLPC654X_ImageHsg_s *ImageHsg);

/**
 * This command loads the specified gamma look-up table into memory from flash. A single load is accomplished by loading data for red, green and blue look-up tables. Use DLP Composer(tm) to create new gamma tables or modify existing gamma tables.
 *
 * \param[in]  GammaLutNumber  Gamma look-up table to load.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageGammaLut(uint8_t GammaLutNumber);

/**
 * Returns the table number of the gamma look-up table currently loaded in memory.
 *
 * \param[out]  GammaLutNumber  Gamma look-up table to load.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageGammaLut(uint8_t *GammaLutNumber);

/**
 * Used to specify the shifts in the gamma curve of Red, Green and Blue. A left shift is a positive offset and a right shift is a negative offset. The effective brightness is increased with a left shift and decreased with a right shift.
 *
 * \param[in]  RedShift  Red gamma curve shift.
 * \param[in]  GreenShift  Green gamma curve shift.
 * \param[in]  BlueShift  Blue gamma curve shift.
 * \param[in]  AllColorShift  Broadcasted shift to gamma curves of all color.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageGammaCurveShift(int8_t RedShift, int8_t GreenShift, int8_t BlueShift, int8_t AllColorShift);

/**
 * Returns Image Gamma Shift for red, green and blue as well as shift to be broadcasted to all colors
 *
 * \param[out]  RedShift  Red gamma curve shift.
 * \param[out]  GreenShift  Green gamma curve shift.
 * \param[out]  BlueShift  Blue gamma curve shift.
 * \param[out]  AllColorShift  Broadcasted shift to gamma curves of all color.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImageGammaCurveShift(int8_t *RedShift, int8_t *GreenShift, int8_t *BlueShift, int8_t *AllColorShift);

/**
 * 
 *
 * \param[in]  WhitePeakingVal  Amount of white processing.   Range 0 to MaxValue (set by IMG_SetWhitePeakingRange()).   The default value of MaxValue is 10.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImgWhitePeakingFactor(uint8_t WhitePeakingVal);

/**
 * 
 *
 * \param[out]  WhitePeakingVal  Amount of white processing.   Range 0 to MaxValue (set by IMG_SetWhitePeakingRange()).   The default value of MaxValue is 10.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadImgWhitePeakingFactor(uint8_t *WhitePeakingVal);

/**
 * HDR maps wider brightness and color range of HDR sources to projector brightness and color range. The mapping requires multiple source groups and system groups define the HDR source and projection device properties respectively. This command sets the source properties and based on this information nearest source group is selected for mapping.
 *
 * \param[in]  HdrSourceConfiguration  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteHdrSourceConfiguration(DLPC654X_HdrSourceConfiguration_s *HdrSourceConfiguration);

/**
 * Includes the metadata information.
 *
 * \param[out]  HdrSourceConfiguration  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadHdrSourceConfiguration(DLPC654X_HdrSourceConfiguration_s *HdrSourceConfiguration);

/**
 * Sets HDR strength which adjusts the electro-optical transfer function that is applied on the input HDR video signal. HDR strength can vary with the ambient brightness level.HDR strength is not applicable for HLG transfer function set by HDR source configuration.
 *
 * \param[in]  HdrStrength  HDR Strength
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteHdrStrengthSetting(uint8_t HdrStrength);

/**
 * 
 *
 * \param[out]  HdrStrength  HDR Strength
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadHdrStrengthSetting(uint8_t *HdrStrength);

/**
 * Sets the system brightness range in nits. These are used in determining the appropriate EOTF and OOTF function to be applied on the HDR source. This need to set only for HDR functionality.
 *
 * \param[in]  MinBrightness  Min Brightness
 * \param[in]  MaxBrightness  Max Brightness
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteSystemBrightnessRangeSetting(double MinBrightness, double MaxBrightness);

/**
 * 
 *
 * \param[out]  MinBrightness  Min Brightness
 * \param[out]  MaxBrightness  Max Brightness
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadSystemBrightnessRangeSetting(double *MinBrightness, double *MaxBrightness);

/**
 * Sets pre-configured Gamma table index and HSG settings as stored in the flash image.
 *
 * \param[in]  ColorProfile  Color Profile
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteImageColorProfile(uint8_t ColorProfile);

/**
 * Programs the direction, logic value and open drain characteristics of a single general purpose I/O pin.
 *
 * \param[in]  PinNumber  Range = 0 to 87.
 * \param[in]  Output  1 = Output(Output buffer enabled)<BR> 0 = Input(Output buffer High Z)
 * \param[in]  LogicVal  1 = LogicVal 1<BR> 0 = LogicVal 0
 * \param[in]  OpenDrain  1 = Open Drain output<BR> 0 = Standard output
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteGpioPinConfig(uint8_t PinNumber, bool Output, bool LogicVal, bool OpenDrain);

/**
 * Returns the direction, logic value and open drain configuration for a single general purpose I/O pin.
 *
 * \param[in]  PinNumber  Range = 0 to 87.
 * \param[out]  Output  1 = Output(Output buffer enabled)<BR> 0 = Input(Output buffer High Z)
 * \param[out]  LogicVal  1 = LogicVal 1<BR> 0 = LogicVal 0
 * \param[out]  OpenDrain  1 = Open Drain output<BR> 0 = Standard output
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadGpioPinConfig(uint8_t PinNumber, bool *Output, bool *LogicVal, bool *OpenDrain);

/**
 * Sets the output logic value for the specified GPIO Pin.
 *
 * \param[in]  PinNumber  Range = 0 to 87.
 * \param[in]  LogicVal  1 = LogicVal 1<BR> 0 = LogicVal 0
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteGpioPin(uint8_t PinNumber, bool LogicVal);

/**
 * Returns the logic value for the specified GPIO pin.
 *
 * \param[in]  PinNumber  Range = 0 to 87.
 * \param[out]  LogicVal  1 = LogicVal 1<BR> 0 = LogicVal 0
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadGpioPin(uint8_t PinNumber, bool *LogicVal);

/**
 * 
 *
 * \param[in]  Clk  Clock to Configure
 * \param[in]  Enabled  TRUE = Enable clock<BR>   FALSE = Disable clock.
 * \param[in]  Freq  Amount to divide the selected clock. This parameter is ignored if the clock is to be disabled. Range 2-127.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteGenPurpseClockEnable(uint8_t Clk, uint8_t Enabled, uint32_t Freq);

/**
 * 
 *
 * \param[in]  Clk  DDP Clock Output.
 * \param[out]  IsEnabled  Is Enabled
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadGenPurpseClockEnable(uint8_t Clk, uint8_t *IsEnabled);

/**
 * 
 *
 * \param[in]  Clk  Clock for which the frequency configuration needs to be returned.
 * \param[out]  Freq  Clock frequency in KHz. Range = 787 to 50,000 KHz.
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadGenPurpseClockFrequency(uint8_t Clk, uint32_t *Freq);

/**
 * Sets the Dutycycle and frequency for the specified PWM port. It also enables or disables the port.
 *
 * \param[in]  Port  Port
 * \param[in]  Frequency  Frequency
 * \param[in]  DutyCycle  Duty Cycle
 * \param[in]  OutputEnabled  1 = Enabled <BR> 0 = Disabled
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WritePwmOutputConfiguration(DLPC654X_Pwmport_e Port, uint32_t Frequency, uint8_t DutyCycle, bool OutputEnabled);

/**
 * Gets the Dutycycle and frequency for the specified PWM port. It also returns whether the port is currently enabled or disabled.
 *
 * \param[in]  Port  Port
 * \param[out]  Frequency  Frequency
 * \param[out]  DutyCycle  Duty Cycle
 * \param[out]  OutputEnabledBit  1 = Enabled <BR> 0 = Disabled
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadPwmOutputConfiguration(DLPC654X_Pwmport_e Port, uint32_t *Frequency, uint8_t *DutyCycle, bool *OutputEnabledBit);

/**
 * Sets the sample rate, dutycycle, high pulse width and low pulse width of the specified PWM incounter port. It also enables or disables the port.
 *
 * \param[in]  Port  Port
 * \param[in]  SampleRate  Sample Rate
 * \param[in]  InCounterEnabled  In Counter Enabled
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WritePwmInputConfiguration(DLPC654X_PwmIncounter_e Port, uint32_t SampleRate, uint8_t InCounterEnabled);

/**
 * Gets the sample rate, dutycycle, high pulse width and low pulse width of the specified PWM incounter port. It also returns whether the port is currently enabled or disabled.
 *
 * \param[in]  Port  Port
 * \param[out]  PwmInputConfiguration  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadPwmInputConfiguration(DLPC654X_PwmIncounter_e Port, DLPC654X_PwmInputConfiguration_s *PwmInputConfiguration);

/**
 * Writes data to specified I2C device address.
 *
 * \param[in]  DataBytesLength  Byte Length for DataBytes
 * \param[in]  I2CPassthrough  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteI2CPassthrough(uint16_t DataBytesLength, DLPC654X_I2CPassthrough_s *I2CPassthrough);

/**
 * Reads data from specified I2C device address.
 *
 * \param[in]  Port  Port
 * \param[in]  Is7Bit  7-bit Address <BR> 0 = 10-bit Address<BR> 1 = 7-bit Address
 * \param[in]  HasSubaddr  SubAddress Present - 0 = No sub-addr present; 1 = sub-addr present
 * \param[in]  ClockRate  Clock Rate
 * \param[in]  DeviceAddress  Device Address
 * \param[in]  ByteCount  Byte Count
 * \param[in]  SubAddr  sub-address (if present)
 * \param[out]  DataBytes  Data Bytes
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadI2CPassthrough(DLPC654X_I2CPortT_e Port, uint8_t Is7Bit, uint8_t HasSubaddr, uint32_t ClockRate, uint16_t DeviceAddress, uint16_t ByteCount, uint8_t* SubAddr, uint8_t *DataBytes);

/**
 * This command applicable only if TMP411A temperature sensor is installed in the system.
 *
 * \param[out]  Temperature  value in degree Celcius
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadDmdTemperature(uint16_t *Temperature);

/**
 * Sets the lock state of EEPROM. When lock is set, all writes to EEPROM settings and/or calibration data from application software will not be actually written to the EEPROM. The locked mode is to be used only in factory where user wants to play around with various settings without actually recording them in the EEPROM. In normal use mode, the lock is not supposed to be set.
 *
 * \param[in]  LockState  '0' - Unlocked<BR>'1' - Locked
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteEepromLockState(uint8_t LockState);

/**
 * Gets the lock state of EEPROM.
 *
 * \param[out]  LockState  '0' - Unlocked<BR>'1' - Locked
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadEepromLockState(uint8_t *LockState);

/**
 * Initializes all programmable parameters for the specified UART port.
 *
 * \param[in]  Port  UART Port
 * \param[in]  UartConfiguration  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteUartConfiguration(DLPC654X_UrtPortT_e Port, DLPC654X_UartConfiguration_s *UartConfiguration);

/**
 * Gets current configuration for the specified UART port.
 *
 * \param[in]  Port  UART Port
 * \param[out]  UartConfiguration  
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadUartConfiguration(DLPC654X_UrtPortT_e Port, DLPC654X_UartConfiguration_s *UartConfiguration);

/**
 * **Fixed Output Enable** : Configures Actuator in fixed output mode. --> Byte 2: 0x00 - Disable 0x01 - Enable --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[in]  FixedOutputEnable  fixed_output_enable
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprFixedOutputEnable(uint8_t ChannelNum, bool FixedOutputEnable);

/**
 * **Fixed Output Enable** : Configures Actuator in fixed output mode. --> Byte 2: 0x00 - Disable 0x01 - Enable --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[out]  FixedOutputEnable  fixed_output_enable
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprFixedOutputEnable(uint8_t ChannelNum, bool *FixedOutputEnable);

/**
 * **DAC Gain** : Set Waveform Generator DAC Gain. --> Byte 2: Range 0 - 255 format u1.7 (0 to 1.9921875)  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[in]  DacGain  dac_gain
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprDacGain(uint8_t ChannelNum, double DacGain);

/**
 * **DAC Gain** : Set Waveform Generator DAC Gain. --> Byte 2: Range 0 - 255 format u1.7 (0 to 1.9921875)  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[out]  DacGain  dac_gain
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprDacGain(uint8_t ChannelNum, double *DacGain);

/**
 * **Subframe delay** : Subframe delay --> Bytes 2-5; Range 0 - 262144 and lsb = 133.333ns
 *
 * \param[in]  ChannelNum  
 * \param[in]  SubframeDelay  subframe_delay
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprSubframeDelay(uint8_t ChannelNum, uint32_t SubframeDelay);

/**
 * **Subframe delay** : Subframe delay --> Bytes 2-5; Range 0 - 262144 and lsb = 133.333ns
 *
 * \param[in]  ChannelNum  
 * \param[out]  SubframeDelay  subframe_delay
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprSubframeDelay(uint8_t ChannelNum, uint32_t *SubframeDelay);

/**
 * **Actuator Type (READ ONLY)** : Actuator type --> Byte 2:  0x00 - NONE  0x01 - Optotune (XPR-25 Model)  0x80 - TI Actuator Interface (EEPROM)  0x81 - TI Actuator Interface (MCU) 
 *
 * \param[out]  ActuatorType  actuator_type
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprActuatorType(uint8_t *ActuatorType);

/**
 * **Output Enable/Disable** : Actuator output enable/disable  --> Byte 2: 0x00 - Disable 0x01 - Enable  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[in]  OutputEnable  output_enable
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprOutputEnable(uint8_t ChannelNum, bool OutputEnable);

/**
 * **Output Enable/Disable** : Actuator output enable/disable  --> Byte 2: 0x00 - Disable 0x01 - Enable  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[out]  OutputEnable  output_enable
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprOutputEnable(uint8_t ChannelNum, bool *OutputEnable);

/**
 * **Clock Width** : Defines the high and low width for the output clock (the clock period will be 2*(ClkWidth+1))  0 = 1 (DAC clock period is two clocks); lsb = 8.33ns  --> Bytes 2-5 : ClkWidth  Example: ClkWidth = 0; will generate clock of 2*(1+1)*8.33 = 33.32ns
 *
 * \param[in]  ChannelNum  
 * \param[in]  ClockWidth  clock_width
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprClockWidth(uint8_t ChannelNum, uint32_t ClockWidth);

/**
 * **Clock Width** : Defines the high and low width for the output clock (the clock period will be 2*(ClkWidth+1))  0 = 1 (DAC clock period is two clocks); lsb = 8.33ns  --> Bytes 2-5 : ClkWidth  Example: ClkWidth = 0; will generate clock of 2*(1+1)*8.33 = 33.32ns
 *
 * \param[in]  ChannelNum  
 * \param[out]  ClockWidth  clock_width
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprClockWidth(uint8_t ChannelNum, uint32_t *ClockWidth);

/**
 * **DAC Offset** : DAC Output Offset  --> Byte 2: Range -128 - +127 format S7 (-128 to +127)  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[in]  DacOffset  dac_offset
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprDacOffset(uint8_t ChannelNum, int8_t DacOffset);

/**
 * **DAC Offset** : DAC Output Offset  --> Byte 2: Range -128 - +127 format S7 (-128 to +127)  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[out]  DacOffset  dac_offset
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprDacOffset(uint8_t ChannelNum, int8_t *DacOffset);

/**
 * **Number of Segments** : Defines number of segments  --> Byte 2: Range 2 - 255  --> Byte 3-5: Reserved must be set to 0x000000 
 *
 * \param[in]  ChannelNum  
 * \param[in]  NumberOfSegments  number_of_segments
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprNumberOfSegments(uint8_t ChannelNum, uint8_t NumberOfSegments);

/**
 * **Number of Segments** : Defines number of segments  --> Byte 2: Range 2 - 255  --> Byte 3-5: Reserved must be set to 0x000000 
 *
 * \param[in]  ChannelNum  
 * \param[out]  NumberOfSegments  number_of_segments
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprNumberOfSegments(uint8_t ChannelNum, uint8_t *NumberOfSegments);

/**
 * **Segments Length** : Defines size of the segments  --> Bytes 2-3: Range 19 - 4095  --> Bytes 4-5: Reserved must be set to 0x0000
 *
 * \param[in]  ChannelNum  
 * \param[in]  SegmentLength  segment_length
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprSegmentLength(uint8_t ChannelNum, uint16_t SegmentLength);

/**
 * **Segments Length** : Defines size of the segments  --> Bytes 2-3: Range 19 - 4095  --> Bytes 4-5: Reserved must be set to 0x0000
 *
 * \param[in]  ChannelNum  
 * \param[out]  SegmentLength  segment_length
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprSegmentLength(uint8_t ChannelNum, uint16_t *SegmentLength);

/**
 * **Invert PWM A** : Applicable when AWC is configured to PWM type instead of DAC  --> Byte 2: 0x00 - No inversion 0x01 - Inverted  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[in]  InvertPwmA  invert_pwm_a
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprInvertPwmA(uint8_t ChannelNum, bool InvertPwmA);

/**
 * **Invert PWM A** : Applicable when AWC is configured to PWM type instead of DAC  --> Byte 2: 0x00 - No inversion 0x01 - Inverted  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[out]  InvertPwmA  invert_pwm_a
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprInvertPwmA(uint8_t ChannelNum, bool *InvertPwmA);

/**
 * **Invert PWM ** : Applicable when AWC is configured to PWM type instead of DAC  --> Byte 2: 0x00 - No inversion 0x01 - Inverted  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[in]  InvertPwmB  invert_pwm_b
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprInvertPwmB(uint8_t ChannelNum, bool InvertPwmB);

/**
 * **Invert PWM ** : Applicable when AWC is configured to PWM type instead of DAC  --> Byte 2: 0x00 - No inversion 0x01 - Inverted  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[out]  InvertPwmB  invert_pwm_b
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprInvertPwmB(uint8_t ChannelNum, bool *InvertPwmB);

/**
 * **Subframe Filter Value** : Sets Subframe Filter Value - defines the minimum time between Subframe edges. Edges closer than the set value will be filtered out.  --> Byte 2: 0 = Filter disabled, >0 = Filter time will be Val x 60us, Range: 0 - 255  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[in]  SubframeFilterValue  subframe_filter_value
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprSubframeFilterValue(uint8_t ChannelNum, uint8_t SubframeFilterValue);

/**
 * **Subframe Filter Value** : Sets Subframe Filter Value - defines the minimum time between Subframe edges. Edges closer than the set value will be filtered out.  --> Byte 2: 0 = Filter disabled, >0 = Filter time will be Val x 60us, Range: 0 - 255  --> Byte 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[out]  SubframeFilterValue  subframe_filter_value
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprSubframeFilterValue(uint8_t ChannelNum, uint8_t *SubframeFilterValue);

/**
 * **Subframe Watch Dog** : Defines the maximum time between Subframe edges; if timer expires, then the WG will automatically output the Fixed Output value.  and the normal output will resume on the next subframe edge.  --> Bytes 2-3: 0 = Subframe watchdog disabled, >0 = Watchdog time will be Time x 60us, Range: Range: 0 - 1023  --> Bytes 4-5: Reserved must be set to 0x0000
 *
 * \param[in]  ChannelNum  
 * \param[in]  SubframeWatchDog  subframe_watch_dog
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprSubframeWatchDog(uint8_t ChannelNum, uint16_t SubframeWatchDog);

/**
 * **Subframe Watch Dog** : Defines the maximum time between Subframe edges; if timer expires, then the WG will automatically output the Fixed Output value.  and the normal output will resume on the next subframe edge.  --> Bytes 2-3: 0 = Subframe watchdog disabled, >0 = Watchdog time will be Time x 60us, Range: Range: 0 - 1023  --> Bytes 4-5: Reserved must be set to 0x0000
 *
 * \param[in]  ChannelNum  
 * \param[out]  SubframeWatchDog  subframe_watch_dog
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprSubframeWatchDog(uint8_t ChannelNum, uint16_t *SubframeWatchDog);

/**
 * **Fixed Output Value** : Defines the value to be output on DAC when fixed output mode is selected. --> Byte 2: Value to be output on DAC, Range -128 to 127. --> Bytes 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[in]  FixedOutputValue  fixed_output_value
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_WriteXprFixedOutputValue(uint8_t ChannelNum, int8_t FixedOutputValue);

/**
 * **Fixed Output Value** : Defines the value to be output on DAC when fixed output mode is selected. --> Byte 2: Value to be output on DAC, Range -128 to 127. --> Bytes 3-5: Reserved must be set to 0x000000
 *
 * \param[in]  ChannelNum  
 * \param[out]  FixedOutputValue  fixed_output_value
 *
 * \return 0 if successful, error code otherwise
 */
uint32_t DLPC654X_ReadXprFixedOutputValue(uint8_t ChannelNum, int8_t *FixedOutputValue);

#ifdef __cplusplus    /* matches __cplusplus construct above */
}
#endif
#endif /* DLPC654X_H */
