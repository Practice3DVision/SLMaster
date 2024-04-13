#[[
  MIT License

  Copyright (C) 2023 by wangwenx190 (Yuhang Zhao)

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
]]

# https://github.com/Chuyu-Team/YY-Thunks

if(NOT MSVC OR DEFINED __YY_THUNKS_CMAKE_INCLUDE_GUARD)
    return()
endif()
set(__YY_THUNKS_CMAKE_INCLUDE_GUARD 1)

if(NOT DEFINED YYTHUNKS_TARGET_OS)
    set(YYTHUNKS_TARGET_OS "Vista" CACHE STRING "Supported values: Vista, WinXP" FORCE)
endif()

set(__yy_thunks_dir "")
if(DEFINED ENV{YYTHUNKS_INSTALL_DIR})
    set(__env "$ENV{YYTHUNKS_INSTALL_DIR}")
    if(NOT "x${__env}" STREQUAL "x" AND EXISTS "${__env}")
        set(__yy_thunks_dir "${__env}")
    endif()
endif()

if("x${__yy_thunks_dir}" STREQUAL "x")
    set(__yy_thunks_reg "")
    cmake_host_system_information(RESULT __yy_thunks_reg
        QUERY WINDOWS_REGISTRY "HKCU/Code/YY-Thunks"
        VALUE "Root")
    if(NOT "x${__yy_thunks_reg}" STREQUAL "x" AND EXISTS "${__yy_thunks_reg}")
        set(__yy_thunks_dir "${__yy_thunks_reg}")
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/YY-Thunks")
        set(__yy_thunks_dir "${CMAKE_CURRENT_LIST_DIR}/YY-Thunks")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/YY-Thunks")
        set(__yy_thunks_dir "${CMAKE_CURRENT_SOURCE_DIR}/YY-Thunks")
    elseif(EXISTS "${PROJECT_SOURCE_DIR}/YY-Thunks")
        set(__yy_thunks_dir "${PROJECT_SOURCE_DIR}/YY-Thunks")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/YY-Thunks")
        set(__yy_thunks_dir "${CMAKE_SOURCE_DIR}/YY-Thunks")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/../YY-Thunks")
        set(__yy_thunks_dir "${CMAKE_SOURCE_DIR}/../YY-Thunks")
    endif()
endif()

if(NOT "x${__yy_thunks_dir}" STREQUAL "x")
    cmake_path(NORMAL_PATH __yy_thunks_dir OUTPUT_VARIABLE __yy_thunks_dir)
endif()

if(NOT "x${__yy_thunks_dir}" STREQUAL "x" AND EXISTS "${__yy_thunks_dir}")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(__yy_thunks_arch x64)
    else()
        set(__yy_thunks_arch x86)
    endif()
    set(__yy_thunks_obj "${__yy_thunks_dir}/objs/${__yy_thunks_arch}/YY_Thunks_for_${YYTHUNKS_TARGET_OS}.obj")
    cmake_path(NORMAL_PATH __yy_thunks_obj OUTPUT_VARIABLE __yy_thunks_obj)
    if(EXISTS "${__yy_thunks_obj}")
        set(YYTHUNKS_FOUND TRUE CACHE BOOL "" FORCE)
        set(YYTHUNKS_INSTALL_DIR "${__yy_thunks_dir}" CACHE STRING "" FORCE)
        set(YYTHUNKS_ARCH "${__yy_thunks_arch}" CACHE STRING "" FORCE)
        set(YYTHUNKS_OBJ_FILE "YY_Thunks_for_${YYTHUNKS_TARGET_OS}.obj" CACHE STRING "" FORCE)
        set(YYTHUNKS_OBJ_PATH "${__yy_thunks_obj}" CACHE STRING "" FORCE)
        add_link_options("${__yy_thunks_obj}")
        message("###################################################################################################")
        message("#                                                                                                 #")
        message("#           ██    ██ ██    ██       ████████ ██   ██ ██    ██ ███    ██ ██   ██ ███████           #")
        message("#            ██  ██   ██  ██           ██    ██   ██ ██    ██ ████   ██ ██  ██  ██                #")
        message("#             ████     ████   █████    ██    ███████ ██    ██ ██ ██  ██ █████   ███████           #")
        message("#              ██       ██             ██    ██   ██ ██    ██ ██  ██ ██ ██  ██       ██           #")
        message("#              ██       ██             ██    ██   ██  ██████  ██   ████ ██   ██ ███████           #")
        message("#                                                                                                 #")
        message("###################################################################################################")
        message("")
        message(" YY-Thunks install dir     :" ${__yy_thunks_dir})
        message(" YY-Thunks target platform :" ${YYTHUNKS_TARGET_OS})
        message(" YY-Thunks obj file path   :" ${__yy_thunks_obj})
        message("")
    else()
        message(WARNING "YY-Thunks's obj file is missing!")
    endif()
else()
    message(WARNING "Can't locate the YY-Thunks installation directory!")
endif()
