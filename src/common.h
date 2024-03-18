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
