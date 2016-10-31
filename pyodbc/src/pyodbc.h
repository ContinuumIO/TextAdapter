
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
// OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef PYODBC_H
#define PYODBC_H


// Python definitions ----------------------------------------------------

// first include Python.h to avoid warnings.
#define PY_SSIZE_T_CLEAN 1

#include <Python.h>
#include <floatobject.h>
#include <longobject.h>
#include <boolobject.h>
#include <unicodeobject.h>
#include <structmember.h>

#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN
#define PyInt_AsSsize_t PyInt_AsLong
#define lenfunc inquiry
#define ssizeargfunc intargfunc
#define ssizeobjargproc intobjargproc
#endif

// System definitions ----------------------------------------------------

#ifdef _MSC_VER
#  define _CRT_SECURE_NO_WARNINGS
#  include <windows.h>
#  include <malloc.h>
typedef __int64 INT64;
typedef unsigned __int64 UINT64;
#else
typedef unsigned char byte;
typedef unsigned int UINT;
typedef long long INT64;
typedef unsigned long long UINT64;
#  define _strcmpi strcasecmp
#  ifdef __MINGW32__
#    include <windef.h>
#    include <malloc.h>
#  else
inline int max(int lhs, int rhs) { return (rhs > lhs) ? rhs : lhs; }
#  endif
#endif

#ifdef __SUN__
#  include <alloca.h>
#endif

#if defined(_MSC_VER)
  #if _MSC_VER < 1600
  /* Visual Studio before 2010 didn't have stdint.h */
  typedef signed char      int8_t;
  typedef short            int16_t;
  typedef int              int32_t;
  typedef __int64          int64_t;
  typedef unsigned char    uint8_t;
  typedef unsigned short   uint16_t;
  typedef unsigned int     uint32_t;
  typedef unsigned __int64 uint64_t;
  #else
  #include <stdint.h>
  #endif
#endif

#if defined(__SUNPRO_CC) || defined(__SUNPRO_C) || (defined(__GNUC__) && !defined(__MINGW32__))
#  include <alloca.h>
#  include <ctype.h>
#  define CDECL cdecl
#  define min(X,Y) ((X) < (Y) ? (X) : (Y))
#  define max(X,Y) ((X) > (Y) ? (X) : (Y))
#  define _alloca alloca
inline void _strlwr(char* name)
{
    while (*name) { *name = tolower(*name); name++; }
}
#else
#  define CDECL
#endif


// ODBC definitions ------------------------------------------------------

#include <sql.h>
#include <sqlext.h>


// Utility functions/definitions  ----------------------------------------

#ifndef _countof
#define _countof(a) (sizeof(a) / sizeof(a[0]))
#endif

inline bool IsSet(DWORD grf, DWORD flags)
{
    return (grf & flags) == flags;
}

#ifdef UNUSED
#undef UNUSED
#endif
inline void UNUSED(...) { }

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)


// Debugging support -----------------------------------------------------

// Building an actual debug version of Python is so much of a pain that it never happens.  I'm providing release-build
// versions of assertions.

extern bool pyodbc_tracing_enabled;
extern bool pyodbc_alloc_guards;
void pyodbc_assertion_failed(const char *file, int line, const char *expr);
void pyodbc_trace_func(const char *file, int line, const char* fmt, ...);
void *pyodbc_guarded_alloc(const char *file, int line, size_t size);
void pyodbc_guarded_dealloc(const char *file, int line, void* ptr);
void pyodbc_check_guards(const char* file, int line, void* ptr, const char *fmt, ...);

#if defined(PYODBC_ASSERT)
  #define I(expr) if (!(expr)) pyodbc_assertion_failed(__FILE__, __LINE__, #expr);
  #define N(expr) if (expr) pyodbc_assertion_failed(__FILE__, __LINE__, #expr);
#else
  #define I(expr)
  #define N(expr)
#endif


#define TRACE(...)                                      \
    if (pyodbc_tracing_enabled)                         \
        pyodbc_trace_func(__FILE__, __LINE__, __VA_ARGS__) 

#define TRACE_NOLOC(...)                        \
    if (pyodbc_tracing_enabled)                 \
        pyodbc_trace_func(NULL, 0, __VA_ARGS__)

#define GUARDED_ALLOC(...)                                  \
    ((!pyodbc_alloc_guards)?                                \
     malloc(__VA_ARGS__) :                                  \
     pyodbc_guarded_alloc(__FILE__, __LINE__, __VA_ARGS__))

#define GUARDED_DEALLOC(...)                                        \
    do if (!pyodbc_alloc_guards) {                                  \
        free(__VA_ARGS__);                                          \
    }                                                               \
    else {                                                          \
        pyodbc_guarded_dealloc(__FILE__, __LINE__, __VA_ARGS__);    \
    } while(0)

#define CHECK_ALLOC_GUARDS(...)                                 \
    if (pyodbc_alloc_guards)                                    \
        pyodbc_check_guards(__FILE__, __LINE__, __VA_ARGS__, "")

#ifdef PYODBC_LEAK_CHECK
#define pyodbc_malloc(len) _pyodbc_malloc(__FILE__, __LINE__, len)
void* _pyodbc_malloc(const char* filename, int lineno, size_t len);
void pyodbc_free(void* p);
void pyodbc_leak_check();
#else
#define pyodbc_malloc malloc
#define pyodbc_free free
#endif


void PrintBytes(void* p, size_t len);


// Python 3 compatibility definitions ------------------------------------
#include "pyodbccompat.h"


#endif // pyodbc_h
