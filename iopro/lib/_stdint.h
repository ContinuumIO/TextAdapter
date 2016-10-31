#ifndef STDINT_H
#define STDINT_H


#if defined(_MSC_VER) && _MSC_VER < 1600
/* Visual Studio before 2010 didn't have stdint.h */
#include <limits.h>
typedef signed char      int8_t;
typedef short            int16_t;
typedef int              int32_t;
typedef __int64          int64_t;
typedef unsigned char    uint8_t;
typedef unsigned short   uint16_t;
typedef unsigned int     uint32_t;
typedef unsigned __int64 uint64_t;
#define INT8_MIN SCHAR_MIN
#define INT8_MAX SCHAR_MAX
#define INT16_MIN SHRT_MIN
#define INT16_MAX SHRT_MAX
#define INT32_MIN INT_MIN
#define INT32_MAX INT_MAX
#define UINT8_MAX UCHAR_MAX
#define UINT16_MAX USHRT_MAX
#define UINT32_MAX UINT_MAX
#define UINT64_MAX _UI64_MAX
#else
#include <stdint.h>
#endif


#endif
