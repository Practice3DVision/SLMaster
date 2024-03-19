/**
 * @file common.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __COMMON_H_
#define __COMMON_H_

#define IN
#define OUT

# ifdef BUILD_SHARED_LIBS 
#   ifdef _WIN32
#       ifdef DLL_EXPORTS
#           define SLMASTER_API _declspec(dllexport)
#       else
#           define SLMASTER_API _declspec(dllimport)
#       endif
#   else
#       define SLMASTER_API
#   endif
# else
#   define SLMASTER_API
# endif

#endif //! __COMMON_H_
