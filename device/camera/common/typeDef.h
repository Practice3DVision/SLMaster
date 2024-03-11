#ifndef IN
#define IN //输入
#endif

#ifndef OUT
#define OUT //输出
#endif

# ifdef BUILD_SHARED_LIBS
#   ifdef _WIN32
#       ifdef DLL_EXPORTS
#           define DEVICE_API _declspec(dllexport)
#       else
#           define DEVICE_API _declspec(dllimport)
#       endif
#   else
#       define DEVICE_API
#   endif
# else
#   define DEVICE_API
# endif
