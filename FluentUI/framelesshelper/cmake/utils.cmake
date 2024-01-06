#[[
  MIT License

  Copyright (C) 2021-2023 by wangwenx190 (Yuhang Zhao)

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

if(NOT DEFINED __WWX190_CMAKE_UTILS_INCLUDED)
set(__WWX190_CMAKE_UTILS_INCLUDED 1)

function(are_paths_equal lhs rhs out_var)
    if(NOT lhs OR NOT rhs)
        message(AUTHOR_WARNING "are_paths_equal: You have to pass two paths to this function!")
        return()
    endif()
    if("x${lhs}" STREQUAL "x" OR "x${rhs}" STREQUAL "x")
        message(AUTHOR_WARNING "are_paths_equal: Empty paths are not allowed!")
        return()
    endif()
    if(NOT out_var)
        message(AUTHOR_WARNING "are_paths_equal: You need to specify an output variable!")
        return()
    endif()
    cmake_path(NORMAL_PATH lhs OUTPUT_VARIABLE path1)
    cmake_path(NORMAL_PATH rhs OUTPUT_VARIABLE path2)
    if(path1 STREQUAL path2)
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(compute_install_dir)
    cmake_parse_arguments(arg "APPEND_64BIT_SUFFIX;ROOT" "IS_MULTI_CONFIG;BIN_DIR;LIB_DIR;INCLUDE_DIR;QML_DIR;DATA_DIR;INFO_DIR;MAN_DIR;DOC_DIR;LOCALE_DIR" "" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "compute_install_dir: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    set(__dir_suffix "")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND arg_APPEND_64BIT_SUFFIX)
        set(__dir_suffix "64")
    endif()
    include(GNUInstallDirs)
    if("x${CMAKE_INSTALL_QMLDIR}" STREQUAL "x")
        set(CMAKE_INSTALL_QMLDIR "qml")
    endif()
    set(__bin_dir "${CMAKE_INSTALL_BINDIR}${__dir_suffix}")
    set(__lib_dir "${CMAKE_INSTALL_LIBDIR}${__dir_suffix}")
    set(__inc_dir "${CMAKE_INSTALL_INCLUDEDIR}")
    set(__qml_dir "${CMAKE_INSTALL_QMLDIR}")
    set(__data_dir "${CMAKE_INSTALL_DATADIR}")
    set(__info_dir "${CMAKE_INSTALL_INFODIR}")
    set(__man_dir "${CMAKE_INSTALL_MANDIR}")
    set(__doc_dir "${CMAKE_INSTALL_DOCDIR}")
    set(__locale_dir "${CMAKE_INSTALL_LOCALEDIR}")
    if("x${CMAKE_BUILD_TYPE}" STREQUAL "x" AND NOT "x${CMAKE_CONFIGURATION_TYPES}" STREQUAL "x")
        if(arg_IS_MULTI_CONFIG)
            set(${arg_IS_MULTI_CONFIG} TRUE PARENT_SCOPE)
        endif()
        if(NOT arg_ROOT)
            set(__subdir "/$<LOWER_CASE:$<CONFIG>>")
            string(APPEND __bin_dir "${__subdir}")
            string(APPEND __lib_dir "${__subdir}")
        endif()
    endif()
    if(arg_BIN_DIR)
        set(${arg_BIN_DIR} "${__bin_dir}" PARENT_SCOPE)
    endif()
    if(arg_LIB_DIR)
        set(${arg_LIB_DIR} "${__lib_dir}" PARENT_SCOPE)
    endif()
    if(arg_INCLUDE_DIR)
        set(${arg_INCLUDE_DIR} "${__inc_dir}" PARENT_SCOPE)
    endif()
    if(arg_QML_DIR)
        set(${arg_QML_DIR} "${__qml_dir}" PARENT_SCOPE)
    endif()
    if(arg_DATA_DIR)
        set(${arg_DATA_DIR} "${__data_dir}" PARENT_SCOPE)
    endif()
    if(arg_INFO_DIR)
        set(${arg_INFO_DIR} "${__info_dir}" PARENT_SCOPE)
    endif()
    if(arg_MAN_DIR)
        set(${arg_MAN_DIR} "${__man_dir}" PARENT_SCOPE)
    endif()
    if(arg_DOC_DIR)
        set(${arg_DOC_DIR} "${__doc_dir}" PARENT_SCOPE)
    endif()
    if(arg_LOCALE_DIR)
        set(${arg_LOCALE_DIR} "${__locale_dir}" PARENT_SCOPE)
    endif()
endfunction()

function(setup_project)
    cmake_parse_arguments(PROJ_ARGS "QT_PROJECT;ENABLE_LTO;WARNINGS_ARE_ERRORS;MAX_WARNING;NO_WARNING;RTTI;EXCEPTIONS" "QML_IMPORT_DIR;LICENSE_HEADER;MODIFY_LOST_WARNING;CONFIG_PREFIX;VERSION_PREFIX" "LANGUAGES" ${ARGN})
    if(NOT PROJ_ARGS_LANGUAGES)
        message(AUTHOR_WARNING "setup_project: You need to specify at least one language for this function!")
        return()
    endif()
    if(PROJ_ARGS_MAX_WARNING AND PROJ_ARGS_NO_WARNING)
        message(AUTHOR_WARNING "setup_project: MAX_WARNING and NO_WARNING can't be enabled at the same time!")
        return()
    endif()
    if(PROJ_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "setup_project: Unrecognized arguments: ${PROJ_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    set(__in_source_build FALSE)
    are_paths_equal(PROJECT_BINARY_DIR PROJECT_SOURCE_DIR __in_source_build)
    if(__in_source_build)
        message(FATAL_ERROR "In-source builds are not allowed. You must build this project out-of-source.")
        return()
    endif()
    # AUTOMOC include directory is a system include directory by default.
    if(POLICY CMP0151)
        cmake_policy(SET CMP0151 NEW)
    endif()
    # Visual Studio Generators select latest Windows SDK by default.
    if(POLICY CMP0149)
        cmake_policy(SET CMP0149 NEW)
    endif()
    # Visual Studio Generators build custom commands in parallel.
    if(POLICY CMP0147)
        cmake_policy(SET CMP0147 NEW)
    endif()
    # USE_FOLDERS global property is treated as ON by default.
    if(POLICY CMP0143)
        cmake_policy(SET CMP0143 NEW)
    endif()
    # MSVC: Do not add "-Z7", "-Zi" or "-ZI" to CMAKE_<LANG>_FLAGS by default. Let developers decide.
    if(POLICY CMP0141)
        cmake_policy(SET CMP0141 NEW)
    endif()
    if(MSVC)
        # Introduced by CMP0141.
        if(NOT DEFINED CMAKE_MSVC_DEBUG_INFORMATION_FORMAT)
            set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>" PARENT_SCOPE)
        endif()
    endif()
    # The if() command supports path comparisons using PATH_EQUAL operator.
    if(POLICY CMP0139)
        cmake_policy(SET CMP0139 NEW)
    endif()
    # CheckIPOSupported uses flags from calling project.
    if(POLICY CMP0138)
        cmake_policy(SET CMP0138 NEW)
    endif()
    # LINK_LIBRARIES supports the $<LINK_ONLY:...> generator expression.
    if(POLICY CMP0131)
        cmake_policy(SET CMP0131 NEW)
    endif()
    # while() diagnoses condition evaluation errors.
    if(POLICY CMP0130)
        cmake_policy(SET CMP0130 NEW)
    endif()
    # Improve language standard and extension selection.
    if(POLICY CMP0128)
        cmake_policy(SET CMP0128 NEW)
    endif()
    # cmake_dependent_option() supports full Condition Syntax.
    if(POLICY CMP0127)
        cmake_policy(SET CMP0127 NEW)
    endif()
    # The set(CACHE) command does not remove any normal variable of the same name from the current scope.
    if(POLICY CMP0126)
        cmake_policy(SET CMP0126 NEW)
    endif()
    # find_(path|file|library|program) have consistent behavior for cache variables.
    if(POLICY CMP0125)
        cmake_policy(SET CMP0125 NEW)
    endif()
    # foreach() loop variables are only available in the loop scope.
    if(POLICY CMP0124)
        cmake_policy(SET CMP0124 NEW)
    endif()
    # The list command detects invalid indices.
    if(POLICY CMP0121)
        cmake_policy(SET CMP0121 NEW)
    endif()
    # LANGUAGE source file property explicitly compiles as language.
    if(POLICY CMP0119)
        cmake_policy(SET CMP0119 NEW)
    endif()
    # Do not add "-GR" to CMAKE_CXX_FLAGS by default.
    if(POLICY CMP0117)
        cmake_policy(SET CMP0117 NEW)
    endif()
    # Ninja generators transform DEPFILEs from add_custom_command().
    if(POLICY CMP0116)
        cmake_policy(SET CMP0116 NEW)
    endif()
    # Source file extensions must be explicit.
    if(POLICY CMP0115)
        cmake_policy(SET CMP0115 NEW)
    endif()
    # Let AUTOMOC and AUTOUIC process header files that end with a .hh extension.
    if(POLICY CMP0100)
        cmake_policy(SET CMP0100 NEW)
    endif()
    # The project() command preserves leading zeros in version components.
    if(POLICY CMP0096)
        cmake_policy(SET CMP0096 NEW)
    endif()
    # MSVC warning flags are not in CMAKE_<LANG>_FLAGS by default.
    if(POLICY CMP0092)
        cmake_policy(SET CMP0092 NEW)
    endif()
    # MSVC: Do not add "-MT(d)" or "-MD(d)" to CMAKE_<LANG>_FLAGS by default. Let developers decide.
    if(POLICY CMP0091)
        cmake_policy(SET CMP0091 NEW)
    endif()
    if(MSVC)
        # Introduced by CMP0091.
        if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
            set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" PARENT_SCOPE)
        endif()
    endif()
    # Add correct link flags for PIE (Position Independent Executable).
    if(POLICY CMP0083)
        cmake_policy(SET CMP0083 NEW)
    endif()
    if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE "Release" PARENT_SCOPE)
    endif()
    if(PROJ_ARGS_ENABLE_LTO)
        # MinGW has many bugs when LTO is enabled, and they are all very
        # hard to workaround, so just don't enable LTO at all for MinGW.
        if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE AND NOT MINGW)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON PARENT_SCOPE)
        endif()
    endif()
    set(__is_multi_config FALSE)
    set(__bin_dir "")
    set(__lib_dir "")
    compute_install_dir(IS_MULTI_CONFIG __is_multi_config BIN_DIR __bin_dir LIB_DIR __lib_dir)
    if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${__bin_dir}" PARENT_SCOPE)
    endif()
    if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${__lib_dir}" PARENT_SCOPE)
    endif()
    if(NOT DEFINED CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${__lib_dir}" PARENT_SCOPE)
    endif()
    if(__is_multi_config)
        compute_install_dir(ROOT BIN_DIR __bin_dir LIB_DIR __lib_dir)
        foreach(__type ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER ${__type} __type_upper)
            string(TOLOWER ${__type} __type_lower)
            set(__bin_var CMAKE_RUNTIME_OUTPUT_DIRECTORY_${__type_upper})
            if(NOT DEFINED ${__bin_var})
                set(${__bin_var} "${PROJECT_BINARY_DIR}/${__bin_dir}/${__type_lower}" PARENT_SCOPE)
            endif()
            set(__lib_var CMAKE_LIBRARY_OUTPUT_DIRECTORY_${__type_upper})
            if(NOT DEFINED ${__lib_var})
                set(${__lib_var} "${PROJECT_BINARY_DIR}/${__lib_dir}/${__type_lower}" PARENT_SCOPE)
            endif()
            set(__ar_var CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${__type_upper})
            if(NOT DEFINED ${__ar_var})
                set(${__ar_var} "${PROJECT_BINARY_DIR}/${__lib_dir}/${__type_lower}" PARENT_SCOPE)
            endif()
        endforeach()
    endif()
    if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
        if(WIN32)
            if(MINGW)
                # MinGW libraries usually don't have a debug postfix but we
                # need to add one when using multi-config generators to avoid
                # our libraries be overwritten by different configurations.
                if(__is_multi_config)
                    set(CMAKE_DEBUG_POSTFIX "d" PARENT_SCOPE)
                endif()
            else()
                set(CMAKE_DEBUG_POSTFIX "d" PARENT_SCOPE)
            endif()
        else()
            set(CMAKE_DEBUG_POSTFIX "_debug" PARENT_SCOPE)
        endif()
    endif()
    if(APPLE)
        if(NOT DEFINED CMAKE_FRAMEWORK_MULTI_CONFIG_POSTFIX_DEBUG)
            set(CMAKE_FRAMEWORK_MULTI_CONFIG_POSTFIX_DEBUG "_debug" PARENT_SCOPE)
        endif()
    endif()
    if(NOT DEFINED CMAKE_INCLUDE_CURRENT_DIR)
        set(CMAKE_INCLUDE_CURRENT_DIR ON PARENT_SCOPE)
    endif()
    if(NOT DEFINED CMAKE_LINK_DEPENDS_NO_SHARED)
        set(CMAKE_LINK_DEPENDS_NO_SHARED ON PARENT_SCOPE)
    endif()
    if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON PARENT_SCOPE)
    endif()
    if(NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN)
        set(CMAKE_VISIBILITY_INLINES_HIDDEN ON PARENT_SCOPE)
    endif()
    if(NOT DEFINED CMAKE_INSTALL_RPATH_USE_LINK_PATH)
        # We'll handle RPATHs manually later.
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF PARENT_SCOPE)
    endif()
    if(APPLE)
        if(NOT DEFINED CMAKE_MACOSX_RPATH)
            set(CMAKE_MACOSX_RPATH ON PARENT_SCOPE)
        endif()
    else()
        if(NOT DEFINED CMAKE_INSTALL_RPATH)
            # Don't set RPATHs globally. We'll handle RPATHs for each target separately later.
            set(CMAKE_INSTALL_RPATH "" PARENT_SCOPE)
        endif()
    endif()
    if(NOT DEFINED CMAKE_INSTALL_PREFIX)
        if(WIN32)
            set(CMAKE_INSTALL_PREFIX "C:/Packages/${PROJECT_NAME}" PARENT_SCOPE)
        else()
            set(CMAKE_INSTALL_PREFIX "/opt/packages/${PROJECT_NAME}" PARENT_SCOPE)
        endif()
    endif()
    if(PROJ_ARGS_QT_PROJECT)
        if(NOT DEFINED CMAKE_AUTOUIC)
            set(CMAKE_AUTOUIC ON PARENT_SCOPE)
        endif()
        if(NOT DEFINED CMAKE_AUTOMOC)
            set(CMAKE_AUTOMOC ON PARENT_SCOPE)
        endif()
        if(NOT DEFINED CMAKE_AUTORCC)
            set(CMAKE_AUTORCC ON PARENT_SCOPE)
        endif()
    endif()
    if(PROJ_ARGS_QML_IMPORT_DIR)
        list(APPEND QML_IMPORT_PATH "${PROJ_ARGS_QML_IMPORT_DIR}")
        list(REMOVE_DUPLICATES QML_IMPORT_PATH)
        set(QML_IMPORT_PATH ${QML_IMPORT_PATH} CACHE STRING "Qt Creator extra QML import paths" FORCE)
    endif()
    foreach(__lang ${PROJ_ARGS_LANGUAGES})
        string(TOUPPER ${__lang} __lang_upper)
        if(__lang_upper STREQUAL "C")
            enable_language(C)
            if(NOT DEFINED CMAKE_C_STANDARD)
                set(CMAKE_C_STANDARD 11 PARENT_SCOPE)
            endif()
            if(NOT DEFINED CMAKE_C_STANDARD_REQUIRED)
                set(CMAKE_C_STANDARD_REQUIRED ON PARENT_SCOPE)
            endif()
            if(NOT DEFINED CMAKE_C_EXTENSIONS)
                set(CMAKE_C_EXTENSIONS OFF PARENT_SCOPE)
            endif()
            if(NOT DEFINED CMAKE_C_VISIBILITY_PRESET)
                set(CMAKE_C_VISIBILITY_PRESET "hidden" PARENT_SCOPE)
            endif()
            if(MSVC)
                if(NOT ("x${CMAKE_C_FLAGS}" STREQUAL "x"))
                    string(REGEX REPLACE "[-|/]w " " " CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
                    string(REGEX REPLACE "[-|/]W[0|1|2|3|4|all|X] " " " CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
                endif()
                if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
                    if(NOT ("x${CMAKE_C_FLAGS_RELEASE}" STREQUAL "x"))
                        string(REGEX REPLACE "[-|/]O[d|0|1|2|3|fast] " " " CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
                    endif()
                endif()
                if(PROJ_ARGS_NO_WARNING)
                    string(APPEND CMAKE_C_FLAGS " /w ")
                elseif(PROJ_ARGS_MAX_WARNING)
                    string(APPEND CMAKE_C_FLAGS " /W4 ") # /Wall gives me 10000+ warnings!
                else()
                    string(APPEND CMAKE_C_FLAGS " /W3 ")
                endif()
                if(PROJ_ARGS_WARNINGS_ARE_ERRORS)
                    string(APPEND CMAKE_C_FLAGS " /WX ")
                endif()
                set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} PARENT_SCOPE)
                if(MSVC_VERSION GREATER_EQUAL 1920) # Visual Studio 2019 version 16.0
                    if(NOT ("x${CMAKE_C_FLAGS_RELEASE}" STREQUAL "x"))
                        string(REGEX REPLACE "[-|/]Ob[0|1|2|3] " " " CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
                    endif()
                    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        string(APPEND CMAKE_C_FLAGS_RELEASE " /Ob2 ")
                    else()
                        string(APPEND CMAKE_C_FLAGS_RELEASE " /Ob3 ")
                    endif()
                    set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} PARENT_SCOPE)
                endif()
                if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
                    string(APPEND CMAKE_C_FLAGS_RELEASE " /clang:-Ofast ")
                    set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} PARENT_SCOPE)
                endif()
            elseif(NOT (MINGW AND (CMAKE_C_COMPILER_ID STREQUAL "GNU")))
                if(NOT ("x${CMAKE_C_FLAGS}" STREQUAL "x"))
                    string(REPLACE "-w " " " CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
                    string(REGEX REPLACE "-W[all|extra|error|pedantic] " " " CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
                endif()
                if(NOT ("x${CMAKE_C_FLAGS_RELEASE}" STREQUAL "x"))
                    string(REGEX REPLACE "-O[d|0|1|2|3|fast] " " " CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
                endif()
                if(PROJ_ARGS_NO_WARNING)
                    string(APPEND CMAKE_C_FLAGS " -w ")
                elseif(PROJ_ARGS_MAX_WARNING)
                    string(APPEND CMAKE_C_FLAGS " -Wall -Wextra ") # -Wpedantic ?
                else()
                    string(APPEND CMAKE_C_FLAGS " -Wall ")
                endif()
                if(PROJ_ARGS_WARNINGS_ARE_ERRORS)
                    string(APPEND CMAKE_C_FLAGS " -Werror ")
                endif()
                string(APPEND CMAKE_C_FLAGS_RELEASE " -Ofast ")
                set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} PARENT_SCOPE)
                set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} PARENT_SCOPE)
            endif()
        elseif(__lang_upper STREQUAL "CXX")
            enable_language(CXX)
            if(NOT DEFINED CMAKE_CXX_STANDARD)
                set(CMAKE_CXX_STANDARD 20 PARENT_SCOPE)
            endif()
            if(NOT DEFINED CMAKE_CXX_STANDARD_REQUIRED)
                set(CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE)
            endif()
            if(NOT DEFINED CMAKE_CXX_EXTENSIONS)
                set(CMAKE_CXX_EXTENSIONS OFF PARENT_SCOPE)
            endif()
            if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
                set(CMAKE_CXX_VISIBILITY_PRESET "hidden" PARENT_SCOPE)
            endif()
            if(MSVC)
                if(NOT ("x${CMAKE_CXX_FLAGS}" STREQUAL "x"))
                    string(REGEX REPLACE "[-|/]GR-? " " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
                    string(REGEX REPLACE "[-|/]EH(a-?|r-?|s-?|c-?)+ " " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
                    string(REGEX REPLACE "[-|/]w " " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
                    string(REGEX REPLACE "[-|/]W[0|1|2|3|4|all|X] " " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
                endif()
                if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
                    if(NOT ("x${CMAKE_CXX_FLAGS_RELEASE}" STREQUAL "x"))
                        string(REGEX REPLACE "[-|/]O[d|0|1|2|3|fast] " " " CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
                    endif()
                endif()
                if(PROJ_ARGS_NO_WARNING)
                    string(APPEND CMAKE_CXX_FLAGS " /w ")
                elseif(PROJ_ARGS_MAX_WARNING)
                    string(APPEND CMAKE_CXX_FLAGS " /W4 ") # /Wall gives me 10000+ warnings!
                else()
                    string(APPEND CMAKE_CXX_FLAGS " /W3 ")
                endif()
                if(PROJ_ARGS_WARNINGS_ARE_ERRORS)
                    string(APPEND CMAKE_CXX_FLAGS " /WX ")
                endif()
                if(PROJ_ARGS_RTTI)
                    string(APPEND CMAKE_CXX_FLAGS " /GR ")
                else()
                    string(APPEND CMAKE_CXX_FLAGS " /GR- ")
                endif()
                if(PROJ_ARGS_EXCEPTIONS)
                    string(APPEND CMAKE_CXX_FLAGS " /EHsc ")
                else()
                    string(APPEND CMAKE_CXX_FLAGS " /EHs-c- ")
                endif()
                set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} PARENT_SCOPE)
                if(MSVC_VERSION GREATER_EQUAL 1920) # Visual Studio 2019 version 16.0
                    if(NOT ("x${CMAKE_CXX_FLAGS_RELEASE}" STREQUAL "x"))
                        string(REGEX REPLACE "[-|/]Ob[0|1|2|3] " " " CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
                    endif()
                    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
                        string(APPEND CMAKE_CXX_FLAGS_RELEASE " /Ob2 ")
                    else()
                        string(APPEND CMAKE_CXX_FLAGS_RELEASE " /Ob3 ")
                    endif()
                    set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} PARENT_SCOPE)
                endif()
                if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
                    string(APPEND CMAKE_CXX_FLAGS_RELEASE " /clang:-Ofast ")
                    set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} PARENT_SCOPE)
                endif()
            elseif(NOT (MINGW AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")))
                if(NOT ("x${CMAKE_CXX_FLAGS}" STREQUAL "x"))
                    string(REPLACE "-w " " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
                    string(REGEX REPLACE "-W[all|extra|error|pedantic] " " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
                endif()
                if(NOT ("x${CMAKE_CXX_FLAGS_RELEASE}" STREQUAL "x"))
                    string(REGEX REPLACE "-O[d|0|1|2|3|fast] " " " CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
                endif()
                if(PROJ_ARGS_NO_WARNING)
                    string(APPEND CMAKE_CXX_FLAGS " -w ")
                elseif(PROJ_ARGS_MAX_WARNING)
                    string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra ") # -Wpedantic ?
                else()
                    string(APPEND CMAKE_CXX_FLAGS " -Wall ")
                endif()
                if(PROJ_ARGS_WARNINGS_ARE_ERRORS)
                    string(APPEND CMAKE_CXX_FLAGS " -Werror ")
                endif()
                if(PROJ_ARGS_RTTI)
                    string(APPEND CMAKE_CXX_FLAGS " -frtti ")
                else()
                    string(APPEND CMAKE_CXX_FLAGS " -fno-rtti ")
                endif()
                if(PROJ_ARGS_EXCEPTIONS)
                    string(APPEND CMAKE_CXX_FLAGS " -fexceptions ")
                else()
                    string(APPEND CMAKE_CXX_FLAGS " -fno-exceptions ")
                endif()
                string(APPEND CMAKE_CXX_FLAGS_RELEASE " -Ofast ")
                set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} PARENT_SCOPE)
                set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} PARENT_SCOPE)
            endif()
        elseif(__lang_upper STREQUAL "RC")
            if(WIN32)
                enable_language(RC)
            endif()
            if(MSVC)
                # Clang-CL forces us use "-" instead of "/" because it always
                # regard everything begins with "/" as a file path instead of
                # a command line parameter.
                set(CMAKE_RC_FLAGS "-c65001 -DWIN32 -nologo" PARENT_SCOPE)
            endif()
        endif()
    endforeach()
    set(__modify_lost_warning "// Caution: This file is generated by CMake automatically during configure.
// WARNING!!! DO NOT EDIT THIS FILE MANUALLY!!!
// ALL YOUR MODIFICATIONS HERE WILL GET LOST AFTER RE-CONFIGURING!!!")
    if(PROJ_ARGS_MODIFY_LOST_WARNING)
        set(__modify_lost_warning "${PROJ_ARGS_MODIFY_LOST_WARNING}")
    endif()
    set(__config_prefix "${PROJECT_NAME}_FEATURE")
    string(TOUPPER "${__config_prefix}" __config_prefix)
    if(PROJ_ARGS_CONFIG_PREFIX)
        set(__config_prefix "${PROJ_ARGS_CONFIG_PREFIX}")
    endif()
    set(__version_prefix "${PROJECT_NAME}")
    string(TOUPPER "${__version_prefix}" __version_prefix)
    if(PROJ_ARGS_VERSION_PREFIX)
        set(__version_prefix "${PROJ_ARGS_VERSION_PREFIX}")
    endif()
    set(__meta_target ${PROJECT_NAME}_METADATA)
    add_custom_target(${__meta_target})
    set_target_properties(${__meta_target} PROPERTIES __PROJECT_MODIFY_LOST_WARNING "${__modify_lost_warning}")
    set_target_properties(${__meta_target} PROPERTIES __PROJECT_CONFIG_PREFIX "${__config_prefix}")
    set_target_properties(${__meta_target} PROPERTIES __PROJECT_VERSION_PREFIX "${__version_prefix}")
    if(PROJ_ARGS_LICENSE_HEADER)
        set_target_properties(${__meta_target} PROPERTIES __PROJECT_LICENSE_HEADER "${PROJ_ARGS_LICENSE_HEADER}")
    endif()
endfunction()

function(get_commit_hash)
    cmake_parse_arguments(GIT_ARGS "" "RESULT" "" ${ARGN})
    if(NOT GIT_ARGS_RESULT)
        message(AUTHOR_WARNING "get_commit_hash: You need to specify a result variable for this function!")
        return()
    endif()
    if(GIT_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "get_commit_hash: Unrecognized arguments: ${GIT_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    set(__hash)
    # We do not want to use git command here because we don't want to make git a build-time dependency.
    if(EXISTS "${PROJECT_SOURCE_DIR}/.git/HEAD")
        file(READ "${PROJECT_SOURCE_DIR}/.git/HEAD" __hash)
        string(STRIP "${__hash}" __hash)
        if(__hash MATCHES "^ref: (.*)")
            set(HEAD "${CMAKE_MATCH_1}")
            if(EXISTS "${PROJECT_SOURCE_DIR}/.git/${HEAD}")
                file(READ "${PROJECT_SOURCE_DIR}/.git/${HEAD}" __hash)
                string(STRIP "${__hash}" __hash)
            else()
                file(READ "${PROJECT_SOURCE_DIR}/.git/packed-refs" PACKED_REFS)
                string(REGEX REPLACE ".*\n([0-9a-f]+) ${HEAD}\n.*" "\\1" __hash "\n${PACKED_REFS}")
            endif()
        endif()
    endif()
    if(__hash)
        set(${GIT_ARGS_RESULT} "${__hash}" PARENT_SCOPE)
    endif()
endfunction()

function(setup_qt_stuff)
    cmake_parse_arguments(QT_ARGS "ALLOW_KEYWORD;NO_DEPRECATED_API" "" "TARGETS" ${ARGN})
    if(NOT QT_ARGS_TARGETS)
        message(AUTHOR_WARNING "setup_qt_stuff: You need to specify at least one target for this function!")
        return()
    endif()
    if(QT_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "setup_qt_stuff: Unrecognized arguments: ${QT_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    foreach(__target ${QT_ARGS_TARGETS})
        if(NOT TARGET ${__target})
            message(AUTHOR_WARNING "${__target} is not a valid CMake target!")
            continue()
        endif()
        target_compile_definitions(${__target} PRIVATE
            QT_NO_CAST_TO_ASCII
            QT_NO_CAST_FROM_ASCII
            QT_NO_CAST_FROM_BYTEARRAY
            QT_NO_URL_CAST_FROM_STRING
            QT_NO_NARROWING_CONVERSIONS_IN_CONNECT
            QT_NO_FOREACH
            QT_NO_JAVA_STYLE_ITERATORS
            QT_NO_AS_CONST
            QT_NO_QEXCHANGE
            QT_NO_USING_NAMESPACE
            QT_NO_CONTEXTLESS_CONNECT
            QT_EXPLICIT_QFILE_CONSTRUCTION_FROM_PATH
            #QT_TYPESAFE_FLAGS # QtQuick private headers prevent us from enabling this flag.
            QT_USE_QSTRINGBUILDER
            QT_USE_FAST_OPERATOR_PLUS
            QT_DEPRECATED_WARNINGS # Have no effect since 5.13
            QT_DEPRECATED_WARNINGS_SINCE=0x070000 # Deprecated since 6.5
            QT_WARN_DEPRECATED_UP_TO=0x070000 # Available since 6.5
        )
        if(QT_ARGS_NO_DEPRECATED_API)
            target_compile_definitions(${__target} PRIVATE
                QT_DISABLE_DEPRECATED_BEFORE=0x070000 # Deprecated since 6.5
                QT_DISABLE_DEPRECATED_UP_TO=0x070000 # Available since 6.5
            )
        endif()
        # On Windows enabling this flag requires us re-compile Qt with this flag enabled,
        # so only enable it on non-Windows platforms.
        if(NOT WIN32)
            target_compile_definitions(${__target} PRIVATE
                QT_STRICT_ITERATORS
            )
        endif()
        # We handle this flag specially because some Qt headers may still use the
        # traditional keywords (especially some private headers).
        if(NOT QT_ARGS_ALLOW_KEYWORD)
            target_compile_definitions(${__target} PRIVATE
                QT_NO_KEYWORDS
            )
        endif()
    endforeach()
endfunction()

function(setup_compile_params)
    if(MINGW AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        message(WARNING "setup_compile_params: Current toolchain is not supported. Only LLVM-MinGW (https://github.com/mstorsjo/llvm-mingw) has partial support.")
        return()
    endif()
    cmake_parse_arguments(COM_ARGS "SPECTRE;EHCONTGUARD;PERMISSIVE;INTELCET;INTELJCC;CFGUARD;FORCE_LTO;SECURE_CODE" "" "TARGETS" ${ARGN})
    if(NOT COM_ARGS_TARGETS)
        message(AUTHOR_WARNING "setup_compile_params: You need to specify at least one target for this function!")
        return()
    endif()
    if(COM_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "setup_compile_params: Unrecognized arguments: ${COM_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    foreach(__target ${COM_ARGS_TARGETS})
        if(NOT TARGET ${__target})
            message(AUTHOR_WARNING "${__target} is not a valid CMake target!")
            continue()
        endif()
        set(__target_type "UNKNOWN")
        get_target_property(__target_type ${__target} TYPE)
        if((__target_type STREQUAL "STATIC_LIBRARY") AND (NOT COM_ARGS_FORCE_LTO))
            # Turn off LTCG/LTO for static libraries, because enabling it for the static libraries
            # will destroy the binary compatibility of them and some compilers will also produce
            # some way too large object files which we can't accept in most cases.
            set_target_properties(${__target} PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION OFF
                INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL OFF
                INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF
                INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO OFF
            )
        endif()
        # Needed by both MSVC and MinGW, otherwise some APIs we need will not be available.
        if(WIN32)
            set(_WIN32_WINNT_WIN10 0x0A00)
            set(NTDDI_WIN10_NI 0x0A00000C)
            # According to MS docs, both "WINVER" and "_WIN32_WINNT" should be defined
            # at the same time and they should use exactly the same value.
            target_compile_definitions(${__target} PRIVATE
                WINVER=${_WIN32_WINNT_WIN10} _WIN32_WINNT=${_WIN32_WINNT_WIN10}
                _WIN32_IE=${_WIN32_WINNT_WIN10} NTDDI_VERSION=${NTDDI_WIN10_NI}
                GDIPVER=0x0110 # Enable GDI+ v1.1, which is available since Windows Vista.
            )
        endif()
        if(MSVC)
            target_compile_definitions(${__target} PRIVATE
                _CRT_NON_CONFORMING_SWPRINTFS
                _CRT_SECURE_NO_WARNINGS _CRT_SECURE_NO_DEPRECATE
                _CRT_NONSTDC_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE
                _SCL_SECURE_NO_WARNINGS _SCL_SECURE_NO_DEPRECATE
                _ENABLE_EXTENDED_ALIGNED_STORAGE # STL fixed a bug which breaks binary compatibility, thus need to be enabled manually by defining this.
                _USE_MATH_DEFINES # Enable the PI constant define for the math headers and also fix the redefinition error caused by Windows SDK's unguarded math macros.
                NOMINMAX # Avoid the Win32 macros min/max conflict with std::min()/std::max().
                UNICODE _UNICODE # Use the -W APIs by default (the -A APIs are just wrappers of the -W APIs internally, so calling the -W APIs directly is more efficient).
                STRICT # https://learn.microsoft.com/en-us/windows/win32/winprog/enabling-strict
                WIN32_LEAN_AND_MEAN WINRT_LEAN_AND_MEAN # Filter out some rarely used headers, to increase compilation speed.
            )
            if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
                target_compile_options(${__target} PRIVATE
                    /bigobj /FS /MP /utf-8 $<$<CONFIG:Release>:/fp:fast /GT /Gw /Gy /Oi /Ot /Oy /Zc:inline>
                )
                if(COM_ARGS_SECURE_CODE)
                    target_compile_options(${__target} PRIVATE
                        /GS /sdl
                    )
                    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
                        target_link_options(${__target} PRIVATE /SAFESEH)
                    endif()
                else()
                    target_compile_options(${__target} PRIVATE
                        /GS- /sdl-
                    )
                    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
                        target_link_options(${__target} PRIVATE /SAFESEH:NO)
                    endif()
                endif()
                target_link_options(${__target} PRIVATE
                    # We want to get the PDB files so we add the "/DEBUG" parameter here,
                    # however, MSVC will disable some optimizations if it found this flag,
                    # but we can override this behavior by explicitly adding the flags which
                    # enable the optimizations.
                    $<$<CONFIG:Release>:/DEBUG /OPT:REF /OPT:ICF /OPT:LBR>
                    /DYNAMICBASE /FIXED:NO /NXCOMPAT /LARGEADDRESSAWARE /WX
                )
                if(__target_type STREQUAL "EXECUTABLE")
                    target_compile_options(${__target} PRIVATE $<$<CONFIG:Release>:/GA>)
                    target_link_options(${__target} PRIVATE /TSAWARE)
                endif()
                if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                    target_link_options(${__target} PRIVATE /HIGHENTROPYVA)
                endif()
                if(MSVC_VERSION GREATER_EQUAL 1910) # Visual Studio 2017 version 15.0 ~ 15.2
                    #target_link_options(${__target} PRIVATE /DEPENDENTLOADFLAG:0x800)
                    target_compile_options(${__target} PRIVATE $<$<CONFIG:Debug,RelWithDebInfo>:/Zf>)
                endif()
                if(MSVC_VERSION GREATER_EQUAL 1915) # Visual Studio 2017 version 15.8
                    target_compile_options(${__target} PRIVATE $<$<CONFIG:Debug,RelWithDebInfo>:/JMC>)
                endif()
                if(MSVC_VERSION GREATER_EQUAL 1920) # Visual Studio 2019 version 16.0
                    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                        target_compile_options(${__target} PRIVATE /d2FH4)
                    endif()
                    if(COM_ARGS_INTELCET)
                        target_link_options(${__target} PRIVATE $<$<CONFIG:Release>:/CETCOMPAT>)
                    else()
                        target_link_options(${__target} PRIVATE $<$<CONFIG:Release>:/CETCOMPAT:NO>)
                    endif()
                endif()
                if(MSVC_VERSION GREATER_EQUAL 1924) # Visual Studio 2019 version 16.4
                    target_compile_options(${__target} PRIVATE $<$<CONFIG:Debug,RelWithDebInfo>:/ZH:SHA_256>)
                endif()
                if(MSVC_VERSION GREATER_EQUAL 1925) # Visual Studio 2019 version 16.5
                    target_compile_options(${__target} PRIVATE $<$<CONFIG:Release>:/QIntel-jcc-erratum>)
                endif()
                if(MSVC_VERSION GREATER_EQUAL 1927) # Visual Studio 2019 version 16.7
                    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                        if(COM_ARGS_EHCONTGUARD)
                            target_compile_options(${__target} PRIVATE $<$<CONFIG:Release>:/guard:ehcont>)
                            target_link_options(${__target} PRIVATE $<$<CONFIG:Release>:/guard:ehcont>)
                        else()
                            target_compile_options(${__target} PRIVATE $<$<CONFIG:Release>:/guard:ehcont->)
                        endif()
                    endif()
                endif()
                if(MSVC_VERSION GREATER_EQUAL 1929) # Visual Studio 2019 version 16.10
                    target_compile_options(${__target} PRIVATE /await:strict)
                elseif(MSVC_VERSION GREATER_EQUAL 1900) # Visual Studio 2015
                    target_compile_options(${__target} PRIVATE /await)
                endif()
                if(MSVC_VERSION GREATER_EQUAL 1930) # Visual Studio 2022 version 17.0
                    target_compile_options(${__target} PRIVATE /options:strict)
                endif()
                if(COM_ARGS_CFGUARD)
                    target_compile_options(${__target} PRIVATE $<$<CONFIG:Release>:/guard:cf>)
                    target_link_options(${__target} PRIVATE $<$<CONFIG:Release>:/GUARD:CF>)
                else()
                    target_compile_options(${__target} PRIVATE $<$<CONFIG:Release>:/guard:cf->)
                    target_link_options(${__target} PRIVATE $<$<CONFIG:Release>:/GUARD:NO>)
                endif()
                if(COM_ARGS_SPECTRE)
                    if(MSVC_VERSION GREATER_EQUAL 1925) # Visual Studio 2019 version 16.5
                        target_compile_options(${__target} PRIVATE /Qspectre-load)
                    elseif(MSVC_VERSION GREATER_EQUAL 1912) # Visual Studio 2017 version 15.5
                        target_compile_options(${__target} PRIVATE /Qspectre)
                    endif()
                endif()
                if(COM_ARGS_PERMISSIVE)
                    target_compile_options(${__target} PRIVATE
                        /Zc:auto /Zc:forScope /Zc:implicitNoexcept /Zc:noexceptTypes /Zc:referenceBinding
                        /Zc:rvalueCast /Zc:sizedDealloc /Zc:strictStrings /Zc:throwingNew /Zc:trigraphs
                        /Zc:wchar_t
                    )
                    if(MSVC_VERSION GREATER_EQUAL 1900) # Visual Studio 2015
                        target_compile_options(${__target} PRIVATE /Zc:threadSafeInit)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1910) # Visual Studio 2017 version 15.0
                        target_compile_options(${__target} PRIVATE /permissive- /Zc:ternary)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1912) # Visual Studio 2017 version 15.5
                        target_compile_options(${__target} PRIVATE /Zc:alignedNew)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1913) # Visual Studio 2017 version 15.6
                        target_compile_options(${__target} PRIVATE /Zc:externConstexpr)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1914) # Visual Studio 2017 version 15.7
                        target_compile_options(${__target} PRIVATE /Zc:__cplusplus)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1921) # Visual Studio 2019 version 16.1
                        target_compile_options(${__target} PRIVATE /Zc:char8_t)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1923) # Visual Studio 2019 version 16.3
                        target_compile_options(${__target} PRIVATE /Zc:externC)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1924) # Visual Studio 2019 version 16.4
                        target_compile_options(${__target} PRIVATE /Zc:hiddenFriend)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1925) # Visual Studio 2019 version 16.5
                        target_compile_options(${__target} PRIVATE /Zc:preprocessor /Zc:tlsGuards)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1928) # Visual Studio 2019 version 16.8 & 16.9
                        target_compile_options(${__target} PRIVATE /Zc:lambda /Zc:zeroSizeArrayNew)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1931) # Visual Studio 2022 version 17.1
                        target_compile_options(${__target} PRIVATE /Zc:static_assert)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1932) # Visual Studio 2022 version 17.2
                        target_compile_options(${__target} PRIVATE /Zc:__STDC__)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1934) # Visual Studio 2022 version 17.4
                        target_compile_options(${__target} PRIVATE /Zc:enumTypes /Zc:gotoScope /Zc:nrvo)
                    endif()
                    if(MSVC_VERSION GREATER_EQUAL 1935) # Visual Studio 2022 version 17.5
                        target_compile_options(${__target} PRIVATE /Zc:templateScope /Zc:checkGwOdr)
                    endif()
                endif()
            endif()
        else()
            target_compile_options(${__target} PRIVATE
                -fthreadsafe-statics
                $<$<CONFIG:Release>:-fmerge-all-constants -ffp-contract=fast -fomit-frame-pointer
                    -ffunction-sections -fdata-sections -funroll-all-loops>
            )
            if(APPLE)
                target_link_options(${__target} PRIVATE
                    -Wl,-fatal_warnings -Wl,-undefined,error
                    $<$<CONFIG:Release>:-Wl,-dead_strip> # -Wl,-no_data_in_code_info -Wl,-no_function_starts
                )
            else()
                target_link_options(${__target} PRIVATE
                    -Wl,--fatal-warnings -Wl,--build-id=sha1 -Wl,--no-undefined -Wl,--as-needed -Wl,-z,defs
                    $<$<CONFIG:Release>:-Wl,--strip-all -Wl,--gc-sections -Wl,-O3 -Wl,--no-keep-memory -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack>
                )
            endif()
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                target_compile_options(${__target} PRIVATE
                    -pipe # Use pipes for communicating between sub-processes. Faster. Have no effect for Clang.
                    $<$<CONFIG:Release>:-Wa,-mbranches-within-32B-boundaries>
                )
            endif()
            if(COM_ARGS_INTELCET)
                target_compile_options(${__target} PRIVATE
                    $<$<CONFIG:Release>:-mshstk>
                )
            endif()
            if(COM_ARGS_CFGUARD)
                if(MINGW)
                    target_compile_options(${__target} PRIVATE
                        $<$<CONFIG:Release>:-mguard=cf>
                    )
                    target_link_options(${__target} PRIVATE
                        $<$<CONFIG:Release>:-mguard=cf>
                    )
                elseif(APPLE OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
                    target_compile_options(${__target} PRIVATE
                        $<$<CONFIG:Release>:-fcf-protection=full>
                    )
                endif()
            endif()
            if(COM_ARGS_SPECTRE)
                if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                    #[[target_compile_options(${__target} PRIVATE
                        $<$<CONFIG:Release>:
                            # These parameters are not compatible with -fcf-protection=full
                            -mindirect-branch=thunk
                            -mfunction-return=thunk
                            -mindirect-branch-register
                            -mindirect-branch-cs-prefix
                        >
                    )]]
                endif()
            endif()
        endif()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(__lto_enabled)
            if(DEFINED CMAKE_BUILD_TYPE)
                set(__upper_type)
                string(TOUPPER ${CMAKE_BUILD_TYPE} __upper_type)
                get_target_property(__lto_enabled ${__target} INTERPROCEDURAL_OPTIMIZATION_${__upper_type})
            endif()
            if(NOT __lto_enabled)
                get_target_property(__lto_enabled ${__target} INTERPROCEDURAL_OPTIMIZATION)
            endif()
            if(__lto_enabled)
                #target_compile_options(${__target} PRIVATE
                #    $<$<CONFIG:Release>:-fsplit-lto-unit -funified-lto>
                #)
                if(__target_type STREQUAL "EXECUTABLE")
                    target_compile_options(${__target} PRIVATE
                        $<$<CONFIG:Release>:-fwhole-program-vtables>
                    )
                endif()
                if(MSVC)
                    target_link_options(${__target} PRIVATE
                        $<$<CONFIG:Release>:/OPT:lldltojobs=all /OPT:lldlto=3 /OPT:lldltocgo=3>
                    )
                else()
                    if(__target_type STREQUAL "EXECUTABLE")
                        target_link_options(${__target} PRIVATE
                            $<$<CONFIG:Release>:-fwhole-program-vtables>
                        )
                    endif()
                    target_link_options(${__target} PRIVATE
                        $<$<CONFIG:Release>:-Wl,--thinlto-jobs=all -Wl,--lto-O3 -Wl,--lto-CGO3>
                    )
                endif()
            endif()
            #[[target_compile_options(${__target} PRIVATE
                $<$<CONFIG:Release>:-fsanitize=shadow-call-stack -fno-stack-protector>
            )
            target_link_options(${__target} PRIVATE
                $<$<CONFIG:Release>:-fsanitize=shadow-call-stack -fno-stack-protector>
            )]]
            target_compile_options(${__target} PRIVATE -fcolor-diagnostics)
            if(MSVC)
                # Required to make the 19041 SDK compatible with clang-cl.
                target_compile_definitions(${__target} PRIVATE __WRL_ENABLE_FUNCTION_STATICS__)
                target_compile_options(${__target} PRIVATE
                    /bigobj /utf-8 /FS
                    -fmsc-version=1935 # Tell clang-cl to emulate Visual Studio 2022 version 17.5
                    # This flag enforces that member pointer base types are complete.
                    # It helps prevent us from running into problems in the Microsoft C++ ABI.
                    -fcomplete-member-pointers
                    # Consistently use backslash as the path separator when expanding the
                    # __FILE__ macro when targeting Windows regardless of the build environment.
                    -ffile-reproducible
                    # Enable ANSI escape codes if something emulating them is around (cmd.exe
                    # doesn't understand ANSI escape codes by default).
                    -fansi-escape-codes
                    /Zc:dllexportInlines- # Do not export inline member functions. This is similar to "-fvisibility-inlines-hidden".
                    /Zc:char8_t /Zc:sizedDealloc /Zc:strictStrings /Zc:threadSafeInit /Zc:trigraphs /Zc:twoPhase
                    $<$<CONFIG:Release>:/clang:-mbranches-within-32B-boundaries /fp:fast /Gw /Gy /Oi /Ot /Zc:inline>
                )
                if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                    target_compile_options(${__target} PRIVATE
                        /clang:-mcx16 # Needed by _InterlockedCompareExchange128() from CPP/WinRT.
                    )
                endif()
                if(COM_ARGS_SECURE_CODE)
                    target_compile_options(${__target} PRIVATE /GS)
                else()
                    target_compile_options(${__target} PRIVATE /GS-)
                endif()
                target_link_options(${__target} PRIVATE
                    --color-diagnostics
                    $<$<CONFIG:Release>:/DEBUG /OPT:REF /OPT:ICF /OPT:LBR /OPT:lldtailmerge>
                    /DYNAMICBASE /FIXED:NO /NXCOMPAT /LARGEADDRESSAWARE
                )
                if(__target_type STREQUAL "EXECUTABLE")
                    target_compile_options(${__target} PRIVATE $<$<CONFIG:Release>:/GA>)
                    target_link_options(${__target} PRIVATE /TSAWARE)
                endif()
                if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                    target_link_options(${__target} PRIVATE /HIGHENTROPYVA)
                endif()
                if(COM_ARGS_CFGUARD)
                    target_compile_options(${__target} PRIVATE $<$<CONFIG:Release>:/guard:cf>)
                    target_link_options(${__target} PRIVATE $<$<CONFIG:Release>:/GUARD:CF>)
                endif()
                if(COM_ARGS_INTELCET)
                    target_link_options(${__target} PRIVATE $<$<CONFIG:Release>:/CETCOMPAT>)
                endif()
                if(COM_ARGS_EHCONTGUARD)
                    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                        target_compile_options(${__target} PRIVATE $<$<CONFIG:Release>:/guard:ehcont>)
                        target_link_options(${__target} PRIVATE $<$<CONFIG:Release>:/guard:ehcont>)
                    endif()
                endif()
            else()
                target_link_options(${__target} PRIVATE -fuse-ld=lld -Wl,--color-diagnostics $<$<CONFIG:Release>:-Wl,--icf=all>)
                if(APPLE)
                    # TODO: -fobjc-arc, -fobjc-arc-exceptions (http://clang.llvm.org/docs/AutomaticReferenceCounting.html)
                    target_compile_options(${__target} PRIVATE -fobjc-call-cxx-cdtors)
                    target_link_options(${__target} PRIVATE $<$<CONFIG:Release>:-Wl,--strict-auto-link>)
                else()
                    target_link_options(${__target} PRIVATE -Wl,-z,keep-text-section-prefix)
                endif()
                if(COM_ARGS_SPECTRE)
                    target_compile_options(${__target} PRIVATE
                        $<$<CONFIG:Release>:-mretpoline -mspeculative-load-hardening>
                    )
                endif()
                if(COM_ARGS_CFGUARD)
                    if(NOT APPLE)
                        target_compile_options(${__target} PRIVATE
                            $<$<CONFIG:Release>:-fsanitize=cfi -fsanitize-cfi-cross-dso>
                        )
                    endif()
                endif()
            endif()
        endif()
    endforeach()
endfunction()

function(setup_gui_app)
    # TODO: macOS bundle icon
    # TODO: Info.plist
    # TODO: sign
    cmake_parse_arguments(GUI_ARGS "" "BUNDLE_ID;BUNDLE_VERSION;BUNDLE_VERSION_SHORT" "TARGETS" ${ARGN})
    if(NOT GUI_ARGS_TARGETS)
        message(AUTHOR_WARNING "setup_gui_app: You need to specify at least one target for this function!")
        return()
    endif()
    if(GUI_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "setup_gui_app: Unrecognized arguments: ${GUI_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    foreach(__target ${GUI_ARGS_TARGETS})
        if(NOT TARGET ${__target})
            message(AUTHOR_WARNING "${__target} is not a valid CMake target!")
            continue()
        endif()
        set_target_properties(${__target} PROPERTIES
            WIN32_EXECUTABLE TRUE
            MACOSX_BUNDLE TRUE
        )
        if(GUI_ARGS_BUNDLE_ID)
            set_target_properties(${__target} PROPERTIES
                MACOSX_BUNDLE_GUI_IDENTIFIER ${GUI_ARGS_BUNDLE_ID}
            )
        endif()
        if(GUI_ARGS_BUNDLE_VERSION)
            set_target_properties(${__target} PROPERTIES
                MACOSX_BUNDLE_BUNDLE_VERSION ${GUI_ARGS_BUNDLE_VERSION}
            )
        endif()
        if(GUI_ARGS_BUNDLE_VERSION_SHORT)
            set_target_properties(${__target} PROPERTIES
                MACOSX_BUNDLE_SHORT_VERSION_STRING ${GUI_ARGS_BUNDLE_VERSION_SHORT}
            )
        endif()
    endforeach()
endfunction()

function(prepare_package_export)
    cmake_parse_arguments(PKG_ARGS "NO_INSTALL" "PACKAGE_NAME;PACKAGE_VERSION" "" ${ARGN})
    if(NOT PKG_ARGS_PACKAGE_NAME)
        message(AUTHOR_WARNING "prepare_package_export: You need to specify the package name for this function!")
        return()
    endif()
    if(NOT PKG_ARGS_PACKAGE_VERSION)
        message(AUTHOR_WARNING "prepare_package_export: You need to specify the package version for this function!")
        return()
    endif()
    if(PKG_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "prepare_package_export: Unrecognized arguments: ${PKG_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    set(__cmake_dir "")
    compute_install_dir(ROOT LIB_DIR __cmake_dir)
    string(APPEND __cmake_dir "/cmake")
    set(__config_name "${PKG_ARGS_PACKAGE_NAME}Config")
    set(__config_file "${__config_name}.cmake")
    set(__version_name "${__config_name}Version")
    set(__versoin_file "${__version_name}.cmake")
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${__versoin_file}"
        VERSION ${PKG_ARGS_PACKAGE_VERSION}
        COMPATIBILITY AnyNewerVersion
    )
    configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/${__config_file}.in"
        "${CMAKE_CURRENT_BINARY_DIR}/${__config_file}"
        INSTALL_DESTINATION "${__cmake_dir}/${PKG_ARGS_PACKAGE_NAME}"
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )
    if(NOT PKG_ARGS_NO_INSTALL)
        install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/${__config_file}"
            "${CMAKE_CURRENT_BINARY_DIR}/${__versoin_file}"
            DESTINATION "${__cmake_dir}/${PKG_ARGS_PACKAGE_NAME}"
        )
    endif()
endfunction()

function(setup_package_export)
    cmake_parse_arguments(PKG_ARGS ""
        "NAMESPACE;PACKAGE_NAME;COMPONENT"
        "TARGETS;PUBLIC_HEADERS;PRIVATE_HEADERS;ALIAS_HEADERS" ${ARGN})
    if(NOT PKG_ARGS_TARGETS)
        message(AUTHOR_WARNING "setup_package_export: You need to specify at least one target for this function!")
        return()
    endif()
    if(NOT PKG_ARGS_NAMESPACE)
        message(AUTHOR_WARNING "setup_package_export: You need to specify an export namespace for this function!")
        return()
    endif()
    if(NOT PKG_ARGS_PACKAGE_NAME)
        message(AUTHOR_WARNING "setup_package_export: You need to specify a package name for this function!")
        return()
    endif()
    if(PKG_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "setup_package_export: Unrecognized arguments: ${PKG_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    set(__bin_dir "")
    set(__lib_dir "")
    set(__include_dir "")
    set(__cmake_dir "")
    compute_install_dir(BIN_DIR __bin_dir LIB_DIR __lib_dir INCLUDE_DIR __include_dir)
    compute_install_dir(ROOT LIB_DIR __cmake_dir)
    string(APPEND __cmake_dir "/cmake")
    set(__inc_dir "${__include_dir}/${PKG_ARGS_PACKAGE_NAME}")
    set(__inc_dir_list "${__include_dir}" "${__inc_dir}")
    if(PKG_ARGS_COMPONENT)
        set(__inc_dir "${__inc_dir}/${PKG_ARGS_COMPONENT}")
        list(APPEND __inc_dir_list "${__inc_dir}")
    endif()
    set(__inc_priv_dir "${__inc_dir}/private")
    list(APPEND __inc_dir_list "${__inc_priv_dir}")
    # If "PKG_ARGS_COMPONENT" is not defined, it's OK, it will be treated as an empty string.
    set(__export_name "${PKG_ARGS_PACKAGE_NAME}${PKG_ARGS_COMPONENT}Targets")
    set(__export_file "${__export_name}.cmake")
    install(
        TARGETS ${PKG_ARGS_TARGETS}
        EXPORT ${__export_name}
        PUBLIC_HEADER DESTINATION "${__inc_dir}"
        PRIVATE_HEADER DESTINATION "${__inc_priv_dir}"
        INCLUDES DESTINATION ${__inc_dir_list}
        BUNDLE DESTINATION .
        RUNTIME DESTINATION "${__bin_dir}"
        LIBRARY DESTINATION "${__lib_dir}"
        ARCHIVE DESTINATION "${__lib_dir}"
    )
    if(MSVC)
        foreach(__target ${PKG_ARGS_TARGETS})
            get_target_property(__target_type ${__target} TYPE)
            if(__target_type STREQUAL "EXECUTABLE" OR __target_type STREQUAL "SHARED_LIBRARY")
                install(FILES "$<TARGET_PDB_FILE:${__target}>" DESTINATION "${__bin_dir}" OPTIONAL)
            endif()
        endforeach()
    endif()
    export(EXPORT ${__export_name}
        FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${__export_file}"
        NAMESPACE ${PKG_ARGS_NAMESPACE}::
    )
    if(PKG_ARGS_PUBLIC_HEADERS)
        install(FILES ${PKG_ARGS_PUBLIC_HEADERS} DESTINATION "${__inc_dir}")
    endif()
    if(PKG_ARGS_PRIVATE_HEADERS)
        install(FILES ${PKG_ARGS_PRIVATE_HEADERS} DESTINATION "${__inc_priv_dir}")
    endif()
    if(PKG_ARGS_ALIAS_HEADERS)
        install(FILES ${PKG_ARGS_ALIAS_HEADERS} DESTINATION "${__inc_dir}")
    endif()
    install(EXPORT ${__export_name}
        FILE ${__export_file}
        NAMESPACE ${PKG_ARGS_NAMESPACE}::
        DESTINATION "${__cmake_dir}/${PKG_ARGS_PACKAGE_NAME}"
    )
endfunction()

function(install2)
    cmake_parse_arguments(arg "" "" "TARGETS" ${ARGN})
    if(NOT arg_TARGETS)
        message(AUTHOR_WARNING "install2: you have to specify at least one target!")
        return()
    endif()
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "install2: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    set(__bin_dir "")
    set(__lib_dir "")
    compute_install_dir(BIN_DIR __bin_dir LIB_DIR __lib_dir)
    install(TARGETS ${arg_TARGETS}
        BUNDLE  DESTINATION .
        RUNTIME DESTINATION "${__bin_dir}"
        LIBRARY DESTINATION "${__lib_dir}"
        ARCHIVE DESTINATION "${__lib_dir}"
    )
    if(MSVC)
        foreach(__target ${arg_TARGETS})
            get_target_property(__target_type ${__target} TYPE)
            if(__target_type STREQUAL "EXECUTABLE" OR __target_type STREQUAL "SHARED_LIBRARY")
                install(FILES "$<TARGET_PDB_FILE:${__target}>" DESTINATION "${__bin_dir}" OPTIONAL)
            endif()
        endforeach()
    endif()
endfunction()

function(deploy_qt_runtime)
    cmake_parse_arguments(DEPLOY_ARGS "NO_INSTALL" "TARGET;QML_SOURCE_DIR;QML_IMPORT_DIR;TRANSLATION_DEPLOY_DIR;QML_DEPLOY_DIR;PLUGIN_DEPLOY_DIR" "" ${ARGN})
    if(NOT DEPLOY_ARGS_TARGET)
        message(AUTHOR_WARNING "deploy_qt_runtime: You need to specify a target for this function!")
        return()
    endif()
    if(DEPLOY_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "deploy_qt_runtime: Unrecognized arguments: ${DEPLOY_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT TARGET Qt6::Core AND NOT TARGET Qt5::Core)
        find_package(Qt6 QUIET COMPONENTS Core)
        if(NOT TARGET Qt6::Core)
            find_package(Qt5 QUIET COMPONENTS Core)
            if(NOT TARGET Qt5::Core)
                message(AUTHOR_WARNING "deploy_qt_runtime: You need to install the QtCore module first to be able to deploy the Qt libraries.")
                return()
            endif()
        endif()
    endif()
    # "QT_QMAKE_EXECUTABLE" is usually defined by QtCreator.
    if(NOT DEFINED QT_QMAKE_EXECUTABLE)
        get_target_property(QT_QMAKE_EXECUTABLE Qt::qmake IMPORTED_LOCATION)
    endif()
    if(NOT EXISTS "${QT_QMAKE_EXECUTABLE}")
        message(WARNING "deploy_qt_runtime: Can't locate the QMake executable.")
        return()
    endif()
    cmake_path(GET QT_QMAKE_EXECUTABLE PARENT_PATH __qt_bin_dir)
    find_program(__deploy_tool NAMES windeployqt macdeployqt HINTS "${__qt_bin_dir}")
    if(NOT EXISTS "${__deploy_tool}")
        message(WARNING "deploy_qt_runtime: Can't locate the deployqt tool.")
        return()
    endif()
    set(__bin_dir "$<TARGET_FILE_DIR:${DEPLOY_ARGS_TARGET}>")
    set(__is_quick_app FALSE)
    set(__full_deploy_command "")
    if(WIN32)
        set(__old_deploy_params "")
        if(QT_VERSION_MAJOR LESS 6)
            set(__old_deploy_params
                --no-webkit2 # Only needed by some very old Qt5 versions, we don't want it.
                #--no-angle # We'll most likely need ANGLE when we use OpenGL, so we'd better ship it.
            )
        endif()
        if(QT_VERSION VERSION_LESS "6.5")
            set(__old_deploy_params
                ${__old_deploy_params}
                --no-virtualkeyboard # windeployqt always copy virtual keyboard libraries if they can be found. But almost all Qt applications don't need them, we also don't want them either.
            )
        endif()
        set(__quick_deploy_params "")
        if(DEPLOY_ARGS_QML_SOURCE_DIR)
            set(__is_quick_app TRUE)
            set(__quick_deploy_params
                --qmldir "${DEPLOY_ARGS_QML_SOURCE_DIR}"
            )
            set(__qml_dir "${__bin_dir}/../qml")
            if(DEPLOY_ARGS_QML_DEPLOY_DIR)
                set(__qml_dir "${DEPLOY_ARGS_QML_DEPLOY_DIR}")
            endif()
            if(QT_VERSION VERSION_GREATER_EQUAL "6.5")
                set(__quick_deploy_params
                    ${__quick_deploy_params}
                    --qml-deploy-dir "${__qml_dir}"
                )
            else()
                set(__quick_deploy_params
                    ${__quick_deploy_params}
                    --dir "${__qml_dir}"
                )
            endif()
        endif()
        if(DEPLOY_ARGS_QML_IMPORT_DIR)
            set(__is_quick_app TRUE)
            set(__quick_deploy_params
                ${__quick_deploy_params}
                --qmlimport "${DEPLOY_ARGS_QML_IMPORT_DIR}"
            )
        endif()
        set(__extra_deploy_params "")
        if(QT_VERSION VERSION_GREATER_EQUAL "6.5")
            set(__translations_dir "${__bin_dir}/../translations")
            if(DEPLOY_ARGS_TRANSLATION_DEPLOY_DIR)
                set(__translations_dir "${DEPLOY_ARGS_TRANSLATION_DEPLOY_DIR}")
            endif()
            set(__extra_deploy_params
                --translationdir "${__translations_dir}"
            )
        endif()
        set(__plugins_dir "${__bin_dir}/../plugins")
        if(DEPLOY_ARGS_PLUGIN_DEPLOY_DIR)
            set(__plugins_dir "${DEPLOY_ARGS_PLUGIN_DEPLOY_DIR}")
        endif()
        set(__extra_deploy_params
            ${__extra_deploy_params}
            --plugindir "${__plugins_dir}" # windeployqt by default will copy all plugins to the application root folder which is very bad.
        )
        set(__full_deploy_params
            $<$<CONFIG:Debug>:--debug> # Sometimes windeployqt can't determine the build type, we tell it explicitly.
            $<$<NOT:$<CONFIG:Debug>>:--release> # Same as above.
            --libdir "${__bin_dir}" # Explicitly set the library deploy path (where to copy Qt6XXX.dll) because if may be affected by other parameters we use.
            #--no-translations # It's better to ship Qt translations altogether, otherwise the strings from Qt will stay English.
            #--no-system-d3d-compiler # QtGui will need d3dcompiler_XX.dll. If we don't ship them, you'll have to make sure the target machine has these libraries.
            #--no-compiler-runtime # The target machine may not installed the same MSVC runtime as the develop machine, so it's better to ship it as well.
            --compiler-runtime # Tell windeployqt we need the MSVC runtime explicitly, otherwise it may not copy it.
            --no-opengl-sw # Mesa3D library llvmpipe backend, mostly for PCs without GPU support. We don't need it.
            --force # Always overwrite existing files, to make sure we always get the most updated files.
            --verbose 2 # Output detailed running log, to help locate the deploy issues if any.
            ${__quick_deploy_params}
            ${__old_deploy_params}
            ${__extra_deploy_params}
        )
        set(__full_deploy_command
            "${__deploy_tool}"
            ${__full_deploy_params}
            "$<TARGET_FILE:${DEPLOY_ARGS_TARGET}>"
        )
    elseif(APPLE)
        set(__quick_deploy_params "")
        if(DEPLOY_ARGS_QML_SOURCE_DIR)
            set(__is_quick_app TRUE)
            set(__quick_deploy_params
                -qmldir="${DEPLOY_ARGS_QML_SOURCE_DIR}"
            )
        endif()
        if(DEPLOY_ARGS_QML_IMPORT_DIR)
            set(__is_quick_app TRUE)
            set(__quick_deploy_params
                ${__quick_deploy_params}
                -qmlimport="${DEPLOY_ARGS_QML_IMPORT_DIR}"
            )
        endif()
        set(__full_deploy_params
            -verbose=2
            ${__quick_deploy_params}
        )
        set(__full_deploy_command
            "${__deploy_tool}"
            "$<TARGET_BUNDLE_DIR:${DEPLOY_ARGS_TARGET}>"
            ${__full_deploy_params}
        )
    elseif(UNIX)
        # TODO: Linux
    endif()
    set(__deploy_target ${DEPLOY_ARGS_TARGET}_deployqt)
    add_custom_target(${__deploy_target}
        COMMAND ${__full_deploy_command}
        WORKING_DIRECTORY "${__bin_dir}"
        COMMENT "Deploying Qt dependencies for target ${DEPLOY_ARGS_TARGET} ..."
        #VERBATIM # This parameter actually makes our command line unusable ...
    )
    # Normally CMake will do this for us automatically, but in case it doesn't ...
    add_dependencies(${__deploy_target} ${DEPLOY_ARGS_TARGET})
    set(__qt_conf "${__bin_dir}/qt.conf")
    # Without qt.conf, our application won't start, because we don't put plugins
    # alongside with our main executable file. We need to explicitly tell Qt where
    # to find them.
    add_custom_command(TARGET ${__deploy_target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E echo "[Paths]" > "${__qt_conf}"
        COMMAND "${CMAKE_COMMAND}" -E echo "Prefix=.." >> "${__qt_conf}"
        COMMENT "Generating qt.conf ..."
    )
    if(NOT DEPLOY_ARGS_NO_INSTALL)
        install2(TARGETS ${DEPLOY_ARGS_TARGET})
        if(QT_VERSION VERSION_GREATER_EQUAL "6.3")
            set(__deploy_script "")
            if(${__is_quick_app})
                qt_generate_deploy_qml_app_script(
                    TARGET ${DEPLOY_ARGS_TARGET}
                    OUTPUT_SCRIPT __deploy_script
                    #MACOS_BUNDLE_POST_BUILD
                    NO_UNSUPPORTED_PLATFORM_ERROR
                    DEPLOY_USER_QML_MODULES_ON_UNSUPPORTED_PLATFORM
                )
            else()
                qt_generate_deploy_app_script(
                    TARGET ${DEPLOY_ARGS_TARGET}
                    OUTPUT_SCRIPT __deploy_script
                    NO_UNSUPPORTED_PLATFORM_ERROR
                )
            endif()
            install(SCRIPT "${__deploy_script}")
        endif()
    endif()
endfunction()

function(setup_translations)
    cmake_parse_arguments(TRANSLATION_ARGS "NO_INSTALL" "TARGET;TS_DIR;QM_DIR;INSTALL_DIR" "LOCALES" ${ARGN})
    if(NOT TRANSLATION_ARGS_TARGET)
        message(AUTHOR_WARNING "setup_translations: You need to specify a target for this function!")
        return()
    endif()
    if(NOT TRANSLATION_ARGS_LOCALES)
        message(AUTHOR_WARNING "setup_translations: You need to specify at least one locale for this function!")
        return()
    endif()
    if(TRANSLATION_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "setup_translations: Unrecognized arguments: ${TRANSLATION_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    # Qt5's CMake functions to create translations lack many features
    # we need and what's worse, they also have a severe bug which will
    # wipe out our .ts files' contents every time we call them, so we
    # really can't use them until Qt6 (the functions have been completely
    # re-written in Qt6 and according to my experiments they work reliably
    # now finally).
    if(NOT TARGET Qt6::LinguistTools)
        find_package(Qt6 QUIET COMPONENTS LinguistTools)
        if(NOT TARGET Qt6::LinguistTools)
            message(AUTHOR_WARNING "setup_translations: You need to install the Qt Linguist Tools first to be able to create translations.")
            return()
        endif()
    endif()
    set(__ts_dir translations)
    if(TRANSLATION_ARGS_TS_DIR)
        set(__ts_dir "${TRANSLATION_ARGS_TS_DIR}")
    endif()
    set(__qm_dir "${PROJECT_BINARY_DIR}/translations")
    if(TRANSLATION_ARGS_QM_DIR)
        set(__qm_dir "${TRANSLATION_ARGS_QM_DIR}")
    endif()
    set(__ts_files)
    foreach(__locale ${TRANSLATION_ARGS_LOCALES})
        list(APPEND __ts_files "${__ts_dir}/${TRANSLATION_ARGS_TARGET}_${__locale}.ts")
    endforeach()
    set_source_files_properties(${__ts_files} PROPERTIES
        OUTPUT_LOCATION "${__qm_dir}"
    )
    set(__qm_files)
    qt_add_translations(${TRANSLATION_ARGS_TARGET}
        TS_FILES ${__ts_files}
        QM_FILES_OUTPUT_VARIABLE __qm_files
        LUPDATE_OPTIONS
            -no-obsolete # Don't keep vanished translation contexts.
        LRELEASE_OPTIONS
            -compress # Compress the QM file if the file size can be decreased siginificantly.
            -nounfinished # Don't include unfinished translations (to save file size).
            -removeidentical # Don't include translations that are the same with their original texts (to save file size).
    )
    if(NOT TRANSLATION_ARGS_NO_INSTALL)
        set(__inst_dir translations)
        if(TRANSLATION_ARGS_INSTALL_DIR)
            set(__inst_dir "${TRANSLATION_ARGS_INSTALL_DIR}")
        endif()
        install(FILES ${__qm_files} DESTINATION "${__inst_dir}")
    endif()
endfunction()

function(generate_win32_rc_file)
    cmake_parse_arguments(RC_ARGS "LIBRARY" "PATH;COMMENTS;COMPANY;DESCRIPTION;VERSION;INTERNAL_NAME;COPYRIGHT;TRADEMARK;ORIGINAL_FILENAME;PRODUCT" "ICONS" ${ARGN})
    if(NOT RC_ARGS_PATH)
        message(AUTHOR_WARNING "generate_win32_rc_file: You need to specify where to put the generated rc file for this function!")
        return()
    endif()
    if(RC_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "generate_win32_rc_file: Unrecognized arguments: ${RC_ARGS_UNPARSED_ARGUMENTS}")
        return()
    endif()
    set(__file_type)
    if(RC_ARGS_LIBRARY)
        set(__file_type "VFT_DLL")
    else()
        set(__file_type "VFT_APP")
    endif()
    set(__icons)
    if(RC_ARGS_ICONS)
        set(__index 1)
        foreach(__icon IN LISTS RC_ARGS_ICONS)
            string(APPEND __icons "IDI_ICON${__index}    ICON    \"${__icon}\"\n")
            math(EXPR __index "${__index} +1")
        endforeach()
    endif()
    set(__comments)
    if(RC_ARGS_COMMENTS)
        set(__comments "${RC_ARGS_COMMENTS}")
    endif()
    set(__company)
    if(RC_ARGS_COMPANY)
        set(__company "${RC_ARGS_COMPANY}")
    endif()
    set(__description)
    if(RC_ARGS_DESCRIPTION)
        set(__description "${RC_ARGS_DESCRIPTION}")
    endif()
    set(__version)
    if(RC_ARGS_VERSION)
        if(RC_ARGS_VERSION MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+")
            set(__version "${RC_ARGS_VERSION}")
        elseif(RC_ARGS_VERSION MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+")
            set(__version "${RC_ARGS_VERSION}.0")
        elseif(RC_ARGS_VERSION MATCHES "[0-9]+\\.[0-9]+")
            set(__version "${RC_ARGS_VERSION}.0.0")
        elseif(RC_ARGS_VERSION MATCHES "[0-9]+")
            set(__version "${RC_ARGS_VERSION}.0.0.0")
        else()
            message(FATAL_ERROR "generate_win32_rc_file: Invalid version format: '${RC_ARGS_VERSION}'")
        endif()
    else()
        set(__version "0.0.0.0")
    endif()
    set(__version_comma)
    string(REPLACE "." "," __version_comma ${__version})
    set(__internal_name)
    if(RC_ARGS_INTERNAL_NAME)
        set(__internal_name "${RC_ARGS_INTERNAL_NAME}")
    endif()
    set(__copyright)
    if(RC_ARGS_COPYRIGHT)
        set(__copyright "${RC_ARGS_COPYRIGHT}")
    endif()
    set(__trademark)
    if(RC_ARGS_TRADEMARK)
        set(__trademark "${RC_ARGS_TRADEMARK}")
    endif()
    set(__original_filename)
    if(RC_ARGS_ORIGINAL_FILENAME)
        set(__original_filename "${RC_ARGS_ORIGINAL_FILENAME}")
    endif()
    set(__product)
    if(RC_ARGS_PRODUCT)
        set(__product "${RC_ARGS_PRODUCT}")
    endif()
    set(__contents "// This file is auto-generated by CMake. DO NOT EDIT! ALL MODIFICATIONS WILL BE LOST!

#include <windows.h> // Use lower-cased file names to be compatible with MinGW.

${__icons}

VS_VERSION_INFO VERSIONINFO
FILEVERSION     ${__version_comma}
PRODUCTVERSION  ${__version_comma}
FILEFLAGSMASK   0x3fL
#ifdef _DEBUG
    FILEFLAGS   VS_FF_DEBUG
#else // !_DEBUG
    FILEFLAGS   0x0L
#endif // _DEBUG
FILEOS          VOS_NT_WINDOWS32
FILETYPE        ${__file_type}
FILESUBTYPE     VFT2_UNKNOWN
BEGIN
    BLOCK \"StringFileInfo\"
    BEGIN
        BLOCK \"040904b0\"
        BEGIN
            VALUE \"CompanyName\",      \"${__company}\\0\"
            VALUE \"FileDescription\",  \"${__description}\\0\"
            VALUE \"FileVersion\",      \"${__version}\\0\"
            VALUE \"LegalCopyright\",   \"${__copyright}\\0\"
            VALUE \"OriginalFilename\", \"${__original_filename}\\0\"
            VALUE \"ProductName\",      \"${__product}\\0\"
            VALUE \"ProductVersion\",   \"${__version}\\0\"
            VALUE \"Comments\",         \"${__comments}\\0\"
            VALUE \"LegalTrademarks\",  \"${__trademark}\\0\"
            VALUE \"InternalName\",     \"${__internal_name}\\0\"
        END
    END
    BLOCK \"VarFileInfo\"
    BEGIN
        VALUE \"Translation\", 0x0409, 1200
    END
END
")
    file(GENERATE OUTPUT "${RC_ARGS_PATH}" CONTENT "${__contents}")
endfunction()

function(generate_win32_manifest_file)
    cmake_parse_arguments(MF_ARGS "UTF8_CODEPAGE;VISTA_COMPAT;WIN7_COMPAT;WIN8_COMPAT;WIN8_1_COMPAT;WIN10_COMPAT;WIN11_COMPAT;XAML_ISLANDS_COMPAT;REQUIRE_ADMIN" "PATH;ID;VERSION;DESCRIPTION;ARCHITECTURE;LANGUAGE;PUBLIC_KEY_TOKEN" "" ${ARGN})
    if(NOT MF_ARGS_PATH)
        message(AUTHOR_WARNING "generate_win32_manifest_file: You need to specify where to put the generated rc file for this function!")
        return()
    endif()
    if(NOT MF_ARGS_ID)
        message(AUTHOR_WARNING "generate_win32_manifest_file: You need to specify your application identifier for this function!")
        return()
    endif()
    if(MF_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "generate_win32_manifest_file: Unrecognized arguments: ${MF_ARGS_UNPARSED_ARGUMENTS}")
        return()
    endif()
    set(__id "${MF_ARGS_ID}")
    set(__version "")
    if(MF_ARGS_VERSION)
        if(MF_ARGS_VERSION MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+")
            set(__version "${MF_ARGS_VERSION}")
        elseif(MF_ARGS_VERSION MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+")
            set(__version "${MF_ARGS_VERSION}.0")
        elseif(MF_ARGS_VERSION MATCHES "[0-9]+\\.[0-9]+")
            set(__version "${MF_ARGS_VERSION}.0.0")
        elseif(MF_ARGS_VERSION MATCHES "[0-9]+")
            set(__version "${MF_ARGS_VERSION}.0.0.0")
        else()
            message(FATAL_ERROR "generate_win32_manifest_file: Invalid version format: '${MF_ARGS_VERSION}'")
        endif()
    else()
        set(__version "0.0.0.0")
    endif()
    set(__architecture "")
    if(MF_ARGS_ARCHITECTURE)
        set(__architecture "processorArchitecture=\"${MF_ARGS_ARCHITECTURE}\"")
    endif()
    set(__language "")
    if(MF_ARGS_LANGUAGE)
        set(__language "language=\"${MF_ARGS_LANGUAGE}\"")
    endif()
    set(__public_key_token "")
    if(MF_ARGS_PUBLIC_KEY_TOKEN)
        set(__public_key_token "publicKeyToken=\"${MF_ARGS_PUBLIC_KEY_TOKEN}\"")
    endif()
    set(__description "")
    if(MF_ARGS_DESCRIPTION)
        set(__description "<description>${MF_ARGS_DESCRIPTION}</description>")
    endif()
    set(__execution_level "")
    if(MF_ARGS_REQUIRE_ADMIN)
        set(__execution_level "requireAdministrator")
    else()
        set(__execution_level "asInvoker")
    endif()
    set(__vista_compat "")
    if(MF_ARGS_VISTA_COMPAT)
        set(__vista_compat "<!-- Windows Vista and Windows Server 2008 -->
      <supportedOS Id=\"{e2011457-1546-43c5-a5fe-008deee3d3f0}\"/>")
    endif()
    set(__win7_compat "")
    if(MF_ARGS_WIN7_COMPAT)
        set(__win7_compat "<!-- Windows 7 and Windows Server 2008 R2 -->
      <supportedOS Id=\"{35138b9a-5d96-4fbd-8e2d-a2440225f93a}\"/>")
    endif()
    set(__win8_compat "")
    if(MF_ARGS_WIN8_COMPAT)
        set(__win8_compat "<!-- Windows 8 and Windows Server 2012 -->
      <supportedOS Id=\"{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}\"/>")
    endif()
    set(__win8_1_compat "")
    if(MF_ARGS_WIN8_1_COMPAT)
        set(__win8_1_compat "<!-- Windows 8.1 and Windows Server 2012 R2 -->
      <supportedOS Id=\"{1f676c76-80e1-4239-95bb-83d0f6d0da78}\"/>")
    endif()
    set(__win10_11_compat "")
    if(MF_ARGS_WIN10_COMPAT OR MF_ARGS_WIN11_COMPAT)
        set(__win10_11_compat "<!-- Windows 10, Windows 11, Windows Server 2016, Windows Server 2019 and Windows Server 2022 -->
      <supportedOS Id=\"{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}\"/>")
    endif()
    set(__xaml_islands_compat "")
    if(MF_ARGS_XAML_ISLANDS_COMPAT)
        set(__xaml_islands_compat "<!-- Windows 10 Version 1809 (October 2018 Update) -->
      <maxversiontested Id=\"10.0.17763.0\"/>
      <!-- Windows 10 Version 1903 (May 2019 Update) -->
      <maxversiontested Id=\"10.0.18362.0\"/>
      <!-- Windows 10 Version 1909 (November 2019 Update) -->
      <maxversiontested Id=\"10.0.18363.0\"/>
      <!-- Windows 10 Version 2004 (May 2020 Update) -->
      <maxversiontested Id=\"10.0.19041.0\"/>
      <!-- Windows 10 Version 20H2 (October 2020 Update) -->
      <maxversiontested Id=\"10.0.19042.0\"/>
      <!-- Windows 10 Version 21H1 (May 2021 Update) -->
      <maxversiontested Id=\"10.0.19043.0\"/>
      <!-- Windows 10 Version 21H2 (November 2021 Update) -->
      <maxversiontested Id=\"10.0.19044.0\"/>
      <!-- Windows 10 Version 22H2 (October 2022 Update) -->
      <maxversiontested Id=\"10.0.19045.0\"/>
      <!-- Windows 11 Version 21H2 -->
      <maxversiontested Id=\"10.0.22000.0\"/>
      <!-- Windows 11 Version 22H2 (October 2022 Update) -->
      <maxversiontested Id=\"10.0.22621.0\"/>")
    endif()
    set(__utf8_codepage "")
    if(MF_ARGS_UTF8_CODEPAGE)
        set(__utf8_codepage "<activeCodePage xmlns=\"http://schemas.microsoft.com/SMI/2019/WindowsSettings\">UTF-8</activeCodePage>")
    endif()
    set(__contents "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>

<!-- This file is auto-generated by CMake. DO NOT EDIT! ALL MODIFICATIONS WILL BE LOST! -->

<assembly xmlns=\"urn:schemas-microsoft-com:asm.v1\" manifestVersion=\"1.0\">
  <assemblyIdentity type=\"win32\" name=\"${__id}\" version=\"${__version}\" ${__architecture} ${__public_key_token} ${__language}/>
  ${__description}
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type=\"win32\" name=\"Microsoft.Windows.Common-Controls\" version=\"6.0.0.0\" processorArchitecture=\"*\" publicKeyToken=\"6595b64144ccf1df\" language=\"*\"/>
    </dependentAssembly>
  </dependency>
  <trustInfo xmlns=\"urn:schemas-microsoft-com:asm.v3\">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level=\"${__execution_level}\" uiAccess=\"false\"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <compatibility xmlns=\"urn:schemas-microsoft-com:compatibility.v1\">
    <application>
      ${__xaml_islands_compat}
      ${__vista_compat}
      ${__win7_compat}
      ${__win8_compat}
      ${__win8_1_compat}
      ${__win10_11_compat}
    </application>
  </compatibility>
  <application xmlns=\"urn:schemas-microsoft-com:asm.v3\">
    <windowsSettings>
      <dpiAware xmlns=\"http://schemas.microsoft.com/SMI/2005/WindowsSettings\">True/PM</dpiAware>
      <printerDriverIsolation xmlns=\"http://schemas.microsoft.com/SMI/2011/WindowsSettings\">true</printerDriverIsolation>
      <disableWindowFiltering xmlns=\"http://schemas.microsoft.com/SMI/2011/WindowsSettings\">true</disableWindowFiltering>
      <highResolutionScrollingAware xmlns=\"http://schemas.microsoft.com/SMI/2013/WindowsSettings\">true</highResolutionScrollingAware>
      <ultraHighResolutionScrollingAware xmlns=\"http://schemas.microsoft.com/SMI/2013/WindowsSettings\">true</ultraHighResolutionScrollingAware>
      <dpiAwareness xmlns=\"http://schemas.microsoft.com/SMI/2016/WindowsSettings\">PerMonitorV2, PerMonitor</dpiAwareness>
      <longPathAware xmlns=\"http://schemas.microsoft.com/SMI/2016/WindowsSettings\">true</longPathAware>
      <!-- <gdiScaling xmlns=\"http://schemas.microsoft.com/SMI/2017/WindowsSettings\">true</gdiScaling> -->
      ${__utf8_codepage}
      <heapType xmlns=\"http://schemas.microsoft.com/SMI/2020/WindowsSettings\">SegmentHeap</heapType>
    </windowsSettings>
  </application>
</assembly>
")
    file(GENERATE OUTPUT "${MF_ARGS_PATH}" CONTENT "${__contents}")
endfunction()

function(query_qt_paths)
    cmake_parse_arguments(QT_ARGS "" "SDK_DIR;BIN_DIR;DOC_DIR;INCLUDE_DIR;LIB_DIR;PLUGINS_DIR;QML_DIR;TRANSLATIONS_DIR" "" ${ARGN})
    if(QT_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "query_qt_paths: Unrecognized arguments: ${QT_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT TARGET Qt6::Core AND NOT TARGET Qt5::Core)
        find_package(Qt6 QUIET COMPONENTS Core)
        if(NOT TARGET Qt6::Core)
            find_package(Qt5 QUIET COMPONENTS Core)
            if(NOT TARGET Qt5::Core)
                message(AUTHOR_WARNING "You need to install the QtCore module first to be able to query Qt installation paths.")
                return()
            endif()
        endif()
    endif()
    # /whatever/Qt/6.5.0/gcc_64/lib/cmake/Qt6
    set(__qt_inst_dir "${Qt${QT_VERSION_MAJOR}_DIR}")
    cmake_path(GET __qt_inst_dir PARENT_PATH __qt_inst_dir)
    cmake_path(GET __qt_inst_dir PARENT_PATH __qt_inst_dir)
    cmake_path(GET __qt_inst_dir PARENT_PATH __qt_inst_dir)
    if(QT_ARGS_SDK_DIR)
        set(${QT_ARGS_SDK_DIR} "${__qt_inst_dir}" PARENT_SCOPE)
    endif()
    if(QT_ARGS_BIN_DIR)
        set(${QT_ARGS_BIN_DIR} "${__qt_inst_dir}/bin" PARENT_SCOPE)
    endif()
    if(QT_ARGS_DOC_DIR)
        set(${QT_ARGS_DOC_DIR} "${__qt_inst_dir}/doc" PARENT_SCOPE)
    endif()
    if(QT_ARGS_INCLUDE_DIR)
        set(${QT_ARGS_INCLUDE_DIR} "${__qt_inst_dir}/include" PARENT_SCOPE)
    endif()
    if(QT_ARGS_LIB_DIR)
        set(${QT_ARGS_LIB_DIR} "${__qt_inst_dir}/lib" PARENT_SCOPE)
    endif()
    if(QT_ARGS_PLUGINS_DIR)
        set(${QT_ARGS_PLUGINS_DIR} "${__qt_inst_dir}/plugins" PARENT_SCOPE)
    endif()
    if(QT_ARGS_QML_DIR)
        set(${QT_ARGS_QML_DIR} "${__qt_inst_dir}/qml" PARENT_SCOPE)
    endif()
    if(QT_ARGS_TRANSLATIONS_DIR)
        set(${QT_ARGS_TRANSLATIONS_DIR} "${__qt_inst_dir}/translations" PARENT_SCOPE)
    endif()
endfunction()

function(query_qt_library_info)
    cmake_parse_arguments(QT_ARGS "" "STATIC;SHARED;VERSION" "" ${ARGN})
    if(QT_ARGS_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "query_qt_library_info: Unrecognized arguments: ${QT_ARGS_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT TARGET Qt6::Core AND NOT TARGET Qt5::Core)
        find_package(Qt6 QUIET COMPONENTS Core)
        if(NOT TARGET Qt6::Core)
            find_package(Qt5 QUIET COMPONENTS Core)
            if(NOT TARGET Qt5::Core)
                message(AUTHOR_WARNING "You need to install the QtCore module first to be able to query Qt library information.")
                return()
            endif()
        endif()
    endif()
    if(QT_ARGS_STATIC OR QT_ARGS_SHARED)
        get_target_property(__lib_type Qt${QT_VERSION_MAJOR}::Core TYPE)
        if(QT_ARGS_STATIC)
            if(__lib_type STREQUAL "STATIC_LIBRARY")
                set(${QT_ARGS_STATIC} ON PARENT_SCOPE)
            elseif(__lib_type STREQUAL "SHARED_LIBRARY")
                set(${QT_ARGS_STATIC} OFF PARENT_SCOPE)
            endif()
        endif()
        if(QT_ARGS_SHARED)
            if(__lib_type STREQUAL "STATIC_LIBRARY")
                set(${QT_ARGS_SHARED} OFF PARENT_SCOPE)
            elseif(__lib_type STREQUAL "SHARED_LIBRARY")
                set(${QT_ARGS_SHARED} ON PARENT_SCOPE)
            endif()
        endif()
    endif()
    if(QT_ARGS_VERSION)
        set(${QT_ARGS_VERSION} "${QT_VERSION}" PARENT_SCOPE)
    endif()
endfunction()

function(dump_target_info)
    cmake_parse_arguments(arg "" "" "TARGETS" ${ARGN})
    if(NOT arg_TARGETS)
        message(AUTHOR_WARNING "dump_target_info: you have to specify at least one target!")
        return()
    endif()
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "dump_target_info: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    foreach(__target ${arg_TARGETS})
        if(NOT TARGET ${__target})
            message(AUTHOR_WARNING "${__target} is not a valid CMake target!")
            continue()
        endif()
        set(__compile_options "")
        set(__compile_definitions "")
        set(__link_options "")
        get_target_property(__compile_options ${__target} COMPILE_OPTIONS)
        get_target_property(__compile_definitions ${__target} COMPILE_DEFINITIONS)
        get_target_property(__link_options ${__target} LINK_OPTIONS)
        message("${__target}'s compile options: ${__compile_options}")
        message("${__target}'s compile definitions: ${__compile_definitions}")
        message("${__target}'s link options: ${__link_options}")
    endforeach()
endfunction()

function(compute_relative_rpath)
    cmake_parse_arguments(arg "" "RPATH;INSTALL_LOCATION;OUTPUT_VARIABLE" "" ${ARGN})
    if(NOT arg_RPATH)
        message(AUTHOR_WARNING "compute_relative_rpath: You need to specify the RPATH parameter!")
        return()
    endif()
    if(NOT arg_INSTALL_LOCATION)
        message(AUTHOR_WARNING "compute_relative_rpath: You need to specify the INSTALL_LOCATION parameter!")
        return()
    endif()
    if(NOT arg_OUTPUT_VARIABLE)
        message(AUTHOR_WARNING "compute_relative_rpath: You need to specify the OUTPUT_VARIABLE parameter!")
        return()
    endif()
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "compute_relative_rpath: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if("x${CMAKE_INSTALL_PREFIX}" STREQUAL "x")
        message(AUTHOR_WARNING "compute_relative_rpath: CMAKE_INSTALL_PREFIX is empty!")
        return()
    endif()
    set(__lib_dir "")
    compute_install_dir(LIB_DIR __lib_dir)
    set(__lib_dir_absolute "${CMAKE_INSTALL_PREFIX}/${__lib_dir}")
    set(__rpath_absolute "")
    cmake_path(ABSOLUTE_PATH arg_RPATH BASE_DIRECTORY __lib_dir_absolute OUTPUT_VARIABLE __rpath_absolute)
    set(__install_location_absolute "${arg_INSTALL_LOCATION}")
    if(NOT IS_ABSOLUTE "${__install_location_absolute}")
        set(__install_location_absolute "${CMAKE_INSTALL_PREFIX}/${arg_INSTALL_LOCATION}")
    endif()
    set(__rpath_relative "")
    cmake_path(RELATIVE_PATH __rpath_absolute BASE_DIRECTORY __install_location_absolute OUTPUT_VARIABLE __rpath_relative)
    if("x${__rpath_relative}" STREQUAL "x")
        # We may get an empty string if the given absolute paths are equal.
        set(__rpath_relative ".")
    endif()
    if(APPLE)
        set(__rpath_base "@loader_path")
    else()
        set(__rpath_base "$ORIGIN")
    endif()
    if(__rpath_relative STREQUAL ".")
        set(__rpath_relative "${__rpath_base}")
    else()
        set(__rpath_relative "${__rpath_base}/${__rpath_relative}")
    endif()
    set(${arg_OUTPUT_VARIABLE} "${__rpath_relative}" PARENT_SCOPE)
endfunction()

function(setup_target_rpaths)
    # RPATH is not supported on Windows and Android platform.
    if(WIN32 OR ANDROID)
        return()
    endif()
    cmake_parse_arguments(arg "" "" "TARGETS" ${ARGN})
    if(NOT arg_TARGETS)
        message(AUTHOR_WARNING "setup_target_rpaths: you have to specify at least one target!")
        return()
    endif()
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "setup_target_rpaths: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if("x${CMAKE_INSTALL_PREFIX}" STREQUAL "x")
        message(AUTHOR_WARNING "setup_target_rpaths: CMAKE_INSTALL_PREFIX is empty!")
        return()
    endif()
    set(__lib_dir "")
    compute_install_dir(LIB_DIR __lib_dir)
    set(__lib_dir_absolute "${CMAKE_INSTALL_PREFIX}/${__lib_dir}")
    set(__is_system_dir -1)
    set(__scopes PLATFORM CXX C)
    foreach(__scope ${__scopes})
        set(__var_name "CMAKE_${__scope}_IMPLICIT_LINK_DIRECTORIES")
        if("x${__var_name}" STREQUAL "x")
            continue()
        endif()
        list(FIND ${__var_name} "${__lib_dir_absolute}" __is_system_dir)
        if(__is_system_dir GREATER_EQUAL 0)
            break()
        endif()
    endforeach()
    if(__is_system_dir LESS 0)
       set(__default_install_rpath "${__lib_dir_absolute}")
    endif()
    foreach(__target ${arg_TARGETS})
        if(NOT TARGET ${__target})
            message(AUTHOR_WARNING "${__target} is not a valid CMake target!")
            continue()
        endif()
        get_target_property(__target_type ${__target} TYPE)
        if(__target_type STREQUAL "INTERFACE_LIBRARY")
            message(AUTHOR_WARNING "You should not set RPATH for ${__target}, which is an interface library.")
            continue()
        endif()
        set(__install_rpath "")
        if(APPLE)
            get_target_property(__is_framework ${__target} FRAMEWORK)
            if(__is_framework)
                set(__install_rpath "/${__target}.framework/Versions/Current")
            endif()
            get_target_property(__is_bundle ${__target} MACOSX_BUNDLE)
            if(__is_bundle AND NOT __is_framework)
                set(__install_rpath "/${__target}.app/Contents/MacOS")
            endif()
        else()
            compute_relative_rpath(RPATH "${__default_install_rpath}" INSTALL_LOCATION "${__lib_dir}" OUTPUT_VARIABLE __install_rpath)
        endif()
        if(NOT "x${__install_rpath}" STREQUAL "x")
            set(__install_prefix "")
            if(NOT "x${CMAKE_STAGING_PREFIX}" STREQUAL "x")
                set(__install_prefix "${CMAKE_STAGING_PREFIX}")
            else()
                set(__install_prefix "${CMAKE_INSTALL_PREFIX}")
            endif()
            set(__will_install FALSE)
            if("x${__install_prefix}" STREQUAL "x")
            else()
                set(__equal FALSE)
                are_paths_equal(__install_prefix PROJECT_BINARY_DIR __equal)
                if(__equal)
                else()
                    set(__win_path "C:/Program Files/${PROJECT_NAME}")
                    are_paths_equal(__install_prefix __win_path __equal)
                    if(__equal)
                    else()
                        set(__win_path "C:/Program Files (x86)/${PROJECT_NAME}")
                        are_paths_equal(__install_prefix __win_path __equal)
                        if(__equal)
                        else()
                            set(__unix_path "/usr/local")
                            are_paths_equal(__install_prefix __unix_path __equal)
                            if(__equal)
                            else()
                                set(__will_install TRUE)
                            endif()
                        endif()
                    endif()
                endif()
            endif()
            if(__will_install)
                set(__prop_name "INSTALL_RPATH")
            else()
                set(__prop_name "BUILD_RPATH")
            endif()
            set_target_properties(${__target} PROPERTIES "${__prop_name}" "${__install_rpath}")
        endif()
    endforeach()
endfunction()

function(add_project_config)
    set(__target ${PROJECT_NAME}_METADATA)
    if(NOT TARGET ${__target})
        message(AUTHOR_WARNING "add_project_config: You need to call setup_project() first!")
        return()
    endif()
    cmake_parse_arguments(arg "" "KEY;VALUE" "CONDITION" ${ARGN})
    if(NOT arg_KEY)
        message(AUTHOR_WARNING "add_project_config: You need to specify the KEY name!")
        return()
    endif()
    if(NOT arg_VALUE AND NOT arg_CONDITION)
        message(AUTHOR_WARNING "add_project_config: You need to specify either the VALUE value or the CONDITION value!")
        return()
    endif()
    if(arg_VALUE AND arg_CONDITION)
        message(AUTHOR_WARNING "add_project_config: VALUE and CONDITION can't be both specified at the same time!")
        return()
    endif()
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "add_project_config: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    set(__prefix "")
    get_target_property(__prefix ${__target} __PROJECT_CONFIG_PREFIX)
    if("x${__prefix}" STREQUAL "x")
        set(__prefix "${PROJECT_NAME}_FEATURE")
        string(TOUPPER "${__prefix}" __prefix)
        message(AUTHOR_WARNING "add_project_config: config prefix is not set, using ${__prefix} as default.")
    endif()
    set(__value "${arg_VALUE}")
    if(arg_CONDITION)
        if(${arg_CONDITION})
            set(__value "TRUE")
        else()
            set(__value "FALSE")
        endif()
    endif()
    set_property(TARGET ${__target} APPEND PROPERTY __PROJECT_CONFIG_OPTIONS "${__prefix}_${arg_KEY}=${__value}")
endfunction()

function(enable_project_config)
    cmake_parse_arguments(arg "" "KEY" "" ${ARGN})
    if(NOT arg_KEY)
        message(AUTHOR_WARNING "enable_project_config: You need to specify the KEY name!")
        return()
    endif()
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "enable_project_config: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    add_project_config(KEY "${arg_KEY}" VALUE "TRUE")
endfunction()

function(disable_project_config)
    cmake_parse_arguments(arg "" "KEY" "" ${ARGN})
    if(NOT arg_KEY)
        message(AUTHOR_WARNING "disable_project_config: You need to specify the KEY name!")
        return()
    endif()
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "disable_project_config: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    add_project_config(KEY "${arg_KEY}" VALUE "FALSE")
endfunction()

function(generate_project_config)
    set(__target ${PROJECT_NAME}_METADATA)
    if(NOT TARGET ${__target})
        message(AUTHOR_WARNING "generate_project_config: You need to call setup_project() first!")
        return()
    endif()
    cmake_parse_arguments(arg "OVERWRITE" "PATH" "" ${ARGN})
    if(NOT arg_PATH)
        message(AUTHOR_WARNING "generate_project_config: You need to specify the file path!")
        return()
    endif()
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "generate_project_config: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(EXISTS "${arg_PATH}" AND NOT arg_OVERWRITE)
        return()
    endif()
    set(__raw_options "")
    get_target_property(__raw_options ${__target} __PROJECT_CONFIG_OPTIONS)
    if("x${__raw_options}" STREQUAL "x")
        message(AUTHOR_WARNING "generate_project_config: You need to specify at least one configuration option!")
        return()
    endif()
    set(__license_header "")
    get_target_property(__license_header ${__target} __PROJECT_LICENSE_HEADER)
    if("x${__license_header}" STREQUAL "x")
        message(AUTHOR_WARNING "generate_project_config: The license header is not set.")
    endif()
    set(__modify_lost_warning "")
    get_target_property(__modify_lost_warning ${__target} __PROJECT_MODIFY_LOST_WARNING)
    if("x${__modify_lost_warning}" STREQUAL "x")
        message(AUTHOR_WARNING "generate_project_config: The modification lost warning is not set.")
    endif()
    set(__inc_guard "_${PROJECT_NAME}_CONFIG_INCLUDE_GUARD_")
    string(TOUPPER "${__inc_guard}" __inc_guard)
    set(__options "")
    foreach(__raw_option ${__raw_options})
        set(__length -1)
        string(LENGTH "${__raw_option}" __length)
        if(__length LESS_EQUAL 0)
            continue()
        endif()
        set(__index -1)
        string(FIND "${__raw_option}" "=" __index)
        if(__index LESS_EQUAL 0)
            continue()
        endif()
        set(__option "")
        string(SUBSTRING "${__raw_option}" 0 ${__index} __option)
        if("x${__option}" STREQUAL "x")
            continue()
        endif()
        math(EXPR __value_index "${__index} + 1")
        math(EXPR __value_length "${__length} - ${__value_index}")
        set(__value "")
        string(SUBSTRING "${__raw_option}" ${__value_index} ${__value_length} __value)
        if("x${__value}" STREQUAL "xTRUE" OR "x${__value}" STREQUAL "xENABLE" OR "x${__value}" STREQUAL "xON"
            OR "x${__value}" STREQUAL "xYES" OR "x${__value}" STREQUAL "xWANT" OR "x${__value}" STREQUAL "xHAS"
            OR "x${__value}" STREQUAL "xHAVE" OR "x${__value}" STREQUAL "xHAD" OR "x${__value}" STREQUAL "x1")
            set(__value "1")
        else()
            set(__value "-1")
        endif()
        string(APPEND __options "#define ${__option} ${__value}\n")
    endforeach()
    set(__config_content "${__license_header}

${__modify_lost_warning}

#pragma once

#ifndef ${__inc_guard}
#define ${__inc_guard}

${__options}
#endif // ${__inc_guard}
")
    file(GENERATE OUTPUT "${arg_PATH}" CONTENT "${__config_content}")
endfunction()

function(query_git_information)
    cmake_parse_arguments(arg "" "HASH;SUBJECT;AUTHOR;DATETIME;BRANCH" "" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "query_git_information: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT Git_FOUND)
        find_package(Git QUIET)
    endif()
    if(NOT Git_FOUND)
        message(AUTHOR_WARNING "query_git_information: Can't find the Git package!")
        return()
    endif()
    #message(STATUS "Git found: ${GIT_EXECUTABLE} (version ${GIT_VERSION_STRING})")
    if(arg_HASH)
        set(__hash "")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" log -1 --pretty=format:%H
            OUTPUT_VARIABLE __hash
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            RESULT_VARIABLE __result_code
        )
        if(__result_code EQUAL 0)
            set(${arg_HASH} "${__hash}" PARENT_SCOPE)
        endif()
    endif()
    if(arg_SUBJECT)
        set(__subject "")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" log -1 --pretty=format:%s
            OUTPUT_VARIABLE __subject
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            RESULT_VARIABLE __result_code
        )
        if(__result_code EQUAL 0)
            set(${arg_SUBJECT} "${__subject}" PARENT_SCOPE)
        endif()
    endif()
    if(arg_AUTHOR)
        set(__author "")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" log -1 "--pretty=format:%aN (%aE)"
            OUTPUT_VARIABLE __author
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            RESULT_VARIABLE __result_code
        )
        if(__result_code EQUAL 0)
            set(${arg_AUTHOR} "${__author}" PARENT_SCOPE)
        endif()
    endif()
    if(arg_DATETIME)
        set(__datetime "")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" log -1 --pretty=format:%aI
            OUTPUT_VARIABLE __datetime
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            RESULT_VARIABLE __result_code
        )
        if(__result_code EQUAL 0)
            set(${arg_DATETIME} "${__datetime}" PARENT_SCOPE)
        endif()
    endif()
    if(arg_BRANCH)
        set(__branch "")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" symbolic-ref --quiet --short HEAD
            OUTPUT_VARIABLE __branch
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            RESULT_VARIABLE __result_code
        )
        if(__result_code EQUAL 0)
            set(${arg_BRANCH} "${__branch}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(query_compiler_information)
    cmake_parse_arguments(arg "" "NAME;VENDOR;VERSION;HOMEPAGE" "" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "query_compiler_information: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    set(__lang "CXX")
    set(__id "${CMAKE_${__lang}_COMPILER_ID}")
    if("x${__id}" STREQUAL "x")
        set(__lang "C")
        set(__id "${CMAKE_${__lang}_COMPILER_ID}")
    endif()
    if("x${__id}" STREQUAL "x")
        message(AUTHOR_WARNING "query_compiler_information: Can't find a valid C/C++ toolchain!")
        return()
    endif()
    set(__sim_id "${CMAKE_${__lang}_SIMULATE_ID}")
    set(__native FALSE)
    if("x${__sim_id}" STREQUAL "x")
        set(__native TRUE)
    endif()
    set(__msvc_like FALSE)
    if("x${__sim_id}" STREQUAL "xMSVC")
        set(__msvc_like TRUE)
    endif()
    set(__gnu_like FALSE)
    if("x${__sim_id}" STREQUAL "xGNU")
        set(__gnu_like TRUE)
    endif()
    # Is this better than the simulate ID ?
    set(__frontend "${CMAKE_${__lang}_COMPILER_FRONTEND_VARIANT}")
    # The target for cross-compiling, if supported.
    set(__triple "${CMAKE_${__lang}_COMPILER_TARGET}")
    if(arg_VERSION)
        set(__version "${CMAKE_${__lang}_COMPILER_VERSION}")
        if("x${__version}" STREQUAL "x")
            set(__version "UNKNOWN")
        endif()
        set(${arg_VERSION} "${__version}" PARENT_SCOPE)
    endif()
    if(__id MATCHES "Clang")
        if(__native)
            if(arg_NAME)
                set(${arg_NAME} "Clang" PARENT_SCOPE)
            endif()
            if(__id STREQUAL "AppleClang")
                if(arg_VENDOR)
                    set(${arg_VENDOR} "Apple®" PARENT_SCOPE)
                endif()
                if(arg_HOMEPAGE)
                    set(${arg_HOMEPAGE} "https://opensource.apple.com/source/clang/" PARENT_SCOPE)
                endif()
            elseif(__id STREQUAL "ARMClang")
                if(arg_VENDOR)
                    set(${arg_VENDOR} "Arm®" PARENT_SCOPE)
                endif()
                if(arg_HOMEPAGE)
                    set(${arg_HOMEPAGE} "https://developer.arm.com/documentation/100748/latest/" PARENT_SCOPE)
                endif()
            elseif(__id STREQUAL "IBMClang" OR __id STREQUAL "XLClang")
                if(arg_VENDOR)
                    set(${arg_VENDOR} "IBM®" PARENT_SCOPE)
                endif()
                if(arg_HOMEPAGE)
                    set(${arg_HOMEPAGE} "https://www.ibm.com/docs/en/openxl-c-and-cpp-aix" PARENT_SCOPE)
                endif()
            elseif(__id STREQUAL "Clang")
                if(arg_VENDOR)
                    set(${arg_VENDOR} "LLVM" PARENT_SCOPE)
                endif()
                if(arg_HOMEPAGE)
                    set(${arg_HOMEPAGE} "https://clang.llvm.org/" PARENT_SCOPE)
                endif()
            else()
                if(arg_NAME)
                    set(${arg_NAME} "Unknown Clang Compiler" PARENT_SCOPE)
                endif()
                if(arg_VENDOR)
                    set(${arg_VENDOR} "UNKNOWN" PARENT_SCOPE)
                endif()
                if(arg_HOMEPAGE)
                    set(${arg_HOMEPAGE} "UNKNOWN" PARENT_SCOPE)
                endif()
            endif()
        elseif(__msvc_like)
            if(arg_NAME)
                set(${arg_NAME} "clang-cl" PARENT_SCOPE)
            endif()
            if(arg_VENDOR)
                set(${arg_VENDOR} "LLVM" PARENT_SCOPE)
            endif()
            if(arg_HOMEPAGE)
                set(${arg_HOMEPAGE} "https://clang.llvm.org/docs/UsersManual.html#clang-cl" PARENT_SCOPE)
            endif()
        elseif(__gnu_like)
            if(arg_NAME)
                set(${arg_NAME} "LLVM MinGW" PARENT_SCOPE)
            endif()
            if(arg_VENDOR)
                set(${arg_VENDOR} "Martin Storsjö" PARENT_SCOPE)
            endif()
            if(arg_HOMEPAGE)
                set(${arg_HOMEPAGE} "https://github.com/mstorsjo/llvm-mingw/" PARENT_SCOPE)
            endif()
        else()
            if(arg_NAME)
                set(${arg_NAME} "Unknown Clang Compiler" PARENT_SCOPE)
            endif()
            if(arg_VENDOR)
                set(${arg_VENDOR} "UNKNOWN" PARENT_SCOPE)
            endif()
            if(arg_HOMEPAGE)
                set(${arg_HOMEPAGE} "UNKNOWN" PARENT_SCOPE)
            endif()
        endif()
    elseif(__id STREQUAL "GNU")
        if(MINGW)
            if(arg_NAME)
                set(${arg_NAME} "MinGW-w64" PARENT_SCOPE)
            endif()
            if(arg_VENDOR)
                set(${arg_VENDOR} "The MinGW-w64 Project" PARENT_SCOPE)
            endif()
            if(arg_HOMEPAGE)
                set(${arg_HOMEPAGE} "https://www.mingw-w64.org/" PARENT_SCOPE)
            endif()
        else()
            if(arg_NAME)
                set(${arg_NAME} "GNU Compiler Collection" PARENT_SCOPE)
            endif()
            if(arg_VENDOR)
                set(${arg_VENDOR} "GNU" PARENT_SCOPE)
            endif()
            if(arg_HOMEPAGE)
                set(${arg_HOMEPAGE} "https://gcc.gnu.org/" PARENT_SCOPE)
            endif()
        endif()
    elseif(__id STREQUAL "MSVC")
        if(arg_NAME)
            set(${arg_NAME} "Microsoft® Visual C/C++ Compiler" PARENT_SCOPE)
        endif()
        if(arg_VENDOR)
            set(${arg_VENDOR} "Microsoft®" PARENT_SCOPE)
        endif()
        if(arg_HOMEPAGE)
            set(${arg_HOMEPAGE} "https://visualstudio.microsoft.com/" PARENT_SCOPE)
        endif()
    elseif(__id MATCHES "Intel")
        if(arg_VENDOR)
            set(${arg_VENDOR} "Intel®" PARENT_SCOPE)
        endif()
        if(__native)
            if(__id STREQUAL "IntelLLVM")
                if(arg_NAME)
                    set(${arg_NAME} "Intel® oneAPI DPC++/C++ Compiler" PARENT_SCOPE)
                endif()
                if(arg_HOMEPAGE)
                    set(${arg_HOMEPAGE} "https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html" PARENT_SCOPE)
                endif()
            elseif(__id STREQUAL "Intel")
                if(arg_NAME)
                    set(${arg_NAME} "Intel® C++ Compiler Classic" PARENT_SCOPE)
                endif()
                if(arg_HOMEPAGE)
                    set(${arg_HOMEPAGE} "https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/overview.html" PARENT_SCOPE)
                endif()
            else()
                if(arg_NAME)
                    set(${arg_NAME} "Unknown Intel Compiler" PARENT_SCOPE)
                endif()
                if(arg_HOMEPAGE)
                    set(${arg_HOMEPAGE} "UNKNOWN" PARENT_SCOPE)
                endif()
            endif()
        elseif(__msvc_like)
            if(arg_NAME)
                set(${arg_NAME} "intel-cl" PARENT_SCOPE)
            endif()
            if(arg_HOMEPAGE)
                set(${arg_HOMEPAGE} "UNKNOWN" PARENT_SCOPE)
            endif()
        elseif(__gnu_like)
            if(arg_NAME)
                set(${arg_NAME} "intel-gcc" PARENT_SCOPE)
            endif()
            if(arg_HOMEPAGE)
                set(${arg_HOMEPAGE} "UNKNOWN" PARENT_SCOPE)
            endif()
        else()
            if(arg_NAME)
                set(${arg_NAME} "Unknown Intel Compiler" PARENT_SCOPE)
            endif()
            if(arg_HOMEPAGE)
                set(${arg_HOMEPAGE} "UNKNOWN" PARENT_SCOPE)
            endif()
        endif()
    else()
        if(arg_NAME)
            set(${arg_NAME} "Unknown C/C++ Compiler" PARENT_SCOPE)
        endif()
        if(arg_VENDOR)
            set(${arg_VENDOR} "UNKNOWN" PARENT_SCOPE)
        endif()
        if(arg_HOMEPAGE)
            set(${arg_HOMEPAGE} "UNKNOWN" PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(generate_project_version)
    set(__target ${PROJECT_NAME}_METADATA)
    if(NOT TARGET ${__target})
        message(AUTHOR_WARNING "generate_project_version: You need to call setup_project() first!")
        return()
    endif()
    cmake_parse_arguments(arg "OVERWRITE;COMMIT_HASH;COMMIT_SUBJECT;COMMIT_AUTHOR;COMMIT_DATETIME;COMMIT_BRANCH;COMPILER_NAME;COMPILER_VENDOR;COMPILER_VERSION;COMPILER_HOMEPAGE;BUILD_DATETIME;ARCHITECTURE;CMAKE_VERSION;GENERATOR" "PATH" "" ${ARGN})
    if(NOT arg_PATH)
        message(AUTHOR_WARNING "generate_project_version: You need to specify the file path!")
        return()
    endif()
    if(arg_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "generate_project_version: Unrecognized arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(EXISTS "${arg_PATH}" AND NOT arg_OVERWRITE)
        return()
    endif()
    set(__license_header "")
    get_target_property(__license_header ${__target} __PROJECT_LICENSE_HEADER)
    if("x${__license_header}" STREQUAL "x")
        message(AUTHOR_WARNING "generate_project_version: The license header is not set.")
    endif()
    set(__modify_lost_warning "")
    get_target_property(__modify_lost_warning ${__target} __PROJECT_MODIFY_LOST_WARNING)
    if("x${__modify_lost_warning}" STREQUAL "x")
        message(AUTHOR_WARNING "generate_project_version: The modification lost warning is not set.")
    endif()
    set(__prefix "")
    get_target_property(__prefix ${__target} __PROJECT_VERSION_PREFIX)
    if("x${__prefix}" STREQUAL "x")
        set(__prefix "${PROJECT_NAME}")
        string(TOUPPER "${__prefix}" __prefix)
        message(AUTHOR_WARNING "generate_project_version: config prefix is not set, using ${__prefix} as default.")
    endif()
    set(__major "${PROJECT_VERSION_MAJOR}")
    if("x${__major}" STREQUAL "x")
        message(AUTHOR_WARNING "generate_project_version: PROJECT_VERSION_MAJOR is not set!")
        set(__major "0")
    endif()
    set(__minor "${PROJECT_VERSION_MINOR}")
    if("x${__minor}" STREQUAL "x")
        message(AUTHOR_WARNING "generate_project_version: PROJECT_VERSION_MINOR is not set!")
        set(__minor "0")
    endif()
    set(__patch "${PROJECT_VERSION_PATCH}")
    if("x${__patch}" STREQUAL "x")
        message(AUTHOR_WARNING "generate_project_version: PROJECT_VERSION_PATCH is not set!")
        set(__patch "0")
    endif()
    set(__tweak "${PROJECT_VERSION_TWEAK}")
    if("x${__tweak}" STREQUAL "x")
        message(AUTHOR_WARNING "generate_project_version: PROJECT_VERSION_TWEAK is not set!")
        set(__tweak "0")
    endif()
    set(__full_ver "${PROJECT_VERSION}")
    if("x${__full_ver}" STREQUAL "x")
        set(__full_ver "0.0.0.0")
    endif()
    set(__full_ver_hex "0x00000000")
    math(EXPR __full_ver_hex "((${__major} & 0xff) << 24) | ((${__minor} & 0xff) << 16) | ((${__patch} & 0xff) << 8) | (${__tweak} & 0xff)" OUTPUT_FORMAT HEXADECIMAL)
    set(__full_ver_hex_length "")
    string(LENGTH "${__full_ver_hex}" __full_ver_hex_length)
    set(__full_ver_hex_suffix_length "")
    math(EXPR __full_ver_hex_suffix_length "${__full_ver_hex_length} - 2")
    if(__full_ver_hex_suffix_length LESS 8)
        set(__full_ver_hex_suffix_length_diff "")
        math(EXPR __full_ver_hex_suffix_length_diff "8 - ${__full_ver_hex_suffix_length}")
        set(__full_ver_hex_suffix "")
        string(SUBSTRING "${__full_ver_hex}" 2 ${__full_ver_hex_suffix_length} __full_ver_hex_suffix)
        foreach(__dummy RANGE 1 ${__full_ver_hex_suffix_length_diff})
            string(PREPEND __full_ver_hex_suffix "0")
        endforeach()
        set(__full_ver_hex "0x${__full_ver_hex_suffix}")
    endif()
    set(__inc_guard "_${PROJECT_NAME}_VERSION_INCLUDE_GUARD_")
    string(TOUPPER "${__inc_guard}" __inc_guard)
    set(__versions "")
    set(__common_num "[[maybe_unused]] inline constexpr const unsigned long")
    set(__common_str "[[maybe_unused]] inline constexpr const char")
    string(APPEND __versions "#define __${__prefix}_VERSION_MAJOR__ ${__major}\n")
    string(APPEND __versions "#define __${__prefix}_VERSION_MINOR__ ${__minor}\n")
    string(APPEND __versions "#define __${__prefix}_VERSION_PATCH__ ${__patch}\n")
    string(APPEND __versions "#define __${__prefix}_VERSION_TWEAK__ ${__tweak}\n")
    string(APPEND __versions "#define __${__prefix}_VERSION__ ${__full_ver_hex}\n\n")
    string(APPEND __versions "${__common_num} ${__prefix}_VERSION_MAJOR = ${__major};\n")
    string(APPEND __versions "${__common_num} ${__prefix}_VERSION_MINOR = ${__minor};\n")
    string(APPEND __versions "${__common_num} ${__prefix}_VERSION_PATCH = ${__patch};\n")
    string(APPEND __versions "${__common_num} ${__prefix}_VERSION_TWEAK = ${__tweak};\n")
    string(APPEND __versions "${__common_num} ${__prefix}_VERSION = ${__full_ver_hex};\n")
    string(APPEND __versions "${__common_str} ${__prefix}_VERSION_STR[] = \"${__full_ver}\";\n")
    if(arg_COMMIT_HASH)
        set(__hash "")
        query_git_information(HASH __hash)
        string(APPEND __versions "${__common_str} ${__prefix}_COMMIT_HASH_STR[] = \"${__hash}\";\n")
    endif()
    if(arg_COMMIT_SUBJECT)
        set(__subject "")
        query_git_information(SUBJECT __subject)
        string(APPEND __versions "${__common_str} ${__prefix}_COMMIT_SUBJECT_STR[] = \"${__subject}\";\n")
    endif()
    if(arg_COMMIT_AUTHOR)
        set(__author "")
        query_git_information(AUTHOR __author)
        string(APPEND __versions "${__common_str} ${__prefix}_COMMIT_AUTHOR_STR[] = \"${__author}\";\n")
    endif()
    if(arg_COMMIT_DATETIME)
        set(__datetime "")
        query_git_information(DATETIME __datetime)
        string(APPEND __versions "${__common_str} ${__prefix}_COMMIT_DATETIME_STR[] = \"${__datetime}\";\n")
    endif()
    if(arg_COMMIT_BRANCH)
        set(__branch "")
        query_git_information(BRANCH __branch)
        string(APPEND __versions "${__common_str} ${__prefix}_COMMIT_BRANCH_STR[] = \"${__branch}\";\n")
    endif()
    if(arg_COMPILER_NAME)
        set(__name "")
        query_compiler_information(NAME __name)
        string(APPEND __versions "${__common_str} ${__prefix}_COMPILER_NAME_STR[] = \"${__name}\";\n")
    endif()
    if(arg_COMPILER_VERSION)
        set(__version "")
        query_compiler_information(VERSION __version)
        string(APPEND __versions "${__common_str} ${__prefix}_COMPILER_VERSION_STR[] = \"${__version}\";\n")
    endif()
    if(arg_COMPILER_VENDOR)
        set(__vendor "")
        query_compiler_information(VENDOR __vendor)
        string(APPEND __versions "${__common_str} ${__prefix}_COMPILER_VENDOR_STR[] = \"${__vendor}\";\n")
    endif()
    if(arg_COMPILER_HOMEPAGE)
        set(__homepage "")
        query_compiler_information(HOMEPAGE __homepage)
        string(APPEND __versions "${__common_str} ${__prefix}_COMPILER_HOMEPAGE_STR[] = \"${__homepage}\";\n")
    endif()
    if(arg_BUILD_DATETIME)
        set(__datetime "")
        string(TIMESTAMP __datetime UTC)
        string(APPEND __versions "${__common_str} ${__prefix}_BUILD_DATETIME_STR[] = \"${__datetime}\";\n")
    endif()
    if(arg_ARCHITECTURE)
        set(__arch "x86")
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(__arch "x64")
        endif()
        string(APPEND __versions "${__common_str} ${__prefix}_ARCHITECTURE_STR[] = \"${__arch}\";\n")
    endif()
    if(arg_CMAKE_VERSION)
        string(APPEND __versions "${__common_str} ${__prefix}_CMAKE_VERSION_STR[] = \"${CMAKE_VERSION}\";\n")
    endif()
    if(arg_GENERATOR)
        string(APPEND __versions "${__common_str} ${__prefix}_CMAKE_GENERATOR_STR[] = \"${CMAKE_GENERATOR}\";\n")
    endif()
    set(__version_content "${__license_header}

${__modify_lost_warning}

#pragma once

#ifndef ${__inc_guard}
#define ${__inc_guard}

${__versions}
#endif // ${__inc_guard}
")
    file(GENERATE OUTPUT "${arg_PATH}" CONTENT "${__version_content}")
endfunction()

endif() # __WWX190_CMAKE_UTILS_INCLUDED
