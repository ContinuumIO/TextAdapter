// This file contains the implementation of several pyodbc debugging facilities

#include "pyodbc.h"
#include <stdarg.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <crtdbg.h>
#  define DEBUG_BREAK __debugbreak
#else
#  include <signal.h>
#  define DEBUG_BREAK() raise(SIGTRAP)
#endif

void
pyodbc_assertion_failed(const char* file, int line, const char* expr)
{
    pyodbc_trace_func(file, line, expr);
    DEBUG_BREAK();
}

// tracing function that can be enabled at runtime:
bool pyodbc_tracing_enabled = false;

// place guards on selected allocation functions (internal use only)
bool pyodbc_alloc_guards = false;

#define GUARD_SIZE_IN_BYTES 64 
#define GUARD_MASK 0xd3adb33f

namespace{
void
fill_guard(uint32_t* ptr, uint32_t mask, size_t count_in_bytes)
{
    TRACE_NOLOC("+ fill_guard ptr: %p size: %u\n",
                ptr, (unsigned int)count_in_bytes);
    size_t count = count_in_bytes / sizeof(uint32_t);
    
    for (size_t i=0; i<count; ++i)
        ptr[i] = mask;
}

bool
check_guard(uint32_t* ptr, uint32_t mask, size_t count_in_bytes)
{
    TRACE_NOLOC("+ check_guard ptr: %p size: %u\n",
                ptr, (unsigned int)count_in_bytes);
    size_t count = count_in_bytes / sizeof(uint32_t);
    uint32_t acc = 0;
    for (size_t i=0; i<count;++i)
        acc |= (ptr[i]^mask);

    return acc == 0;
}

}


void
pyodbc_check_guards(const char* file, int line, void* ptr, const char* fmt, ...)
{
    char *base = ((char*)ptr - GUARD_SIZE_IN_BYTES);
    char *tail = *(char**)base;
    base+=sizeof(char*);

    int i = 0;
    i += check_guard((uint32_t*)base, GUARD_MASK, GUARD_SIZE_IN_BYTES - sizeof(char*)) ? 0 : 1;
    i += check_guard((uint32_t*)tail, GUARD_MASK, GUARD_SIZE_IN_BYTES) ? 0 : 2;

    if (i) {
        char buff [4096];
        va_list args;
        va_start(args, fmt);

        char* type[] = { "", "lower guard", "higher guard", "both guards" };

        size_t c = snprintf(buff, sizeof(buff),
                            "allocation guard for block %p overwritten (%s)\n",
                            ptr, type[i]);
        
        if (c > sizeof(buff)) {
            vsnprintf(buff+c, sizeof(buff) - c, fmt, args);
        }
        pyodbc_assertion_failed(file, line, buff);
        /* code does not reach here */
    }
}

void*
pyodbc_guarded_alloc(const char */*file*/, int /*line*/, size_t orig_size)
{
    size_t size = orig_size + 2*GUARD_SIZE_IN_BYTES;
    char* mem = (char*)malloc(size);
    if (mem) {
        void *user_mem = (void*)(mem + GUARD_SIZE_IN_BYTES);

        TRACE_NOLOC("guarded alloc - base: %p user: %p size: %u\n",
                    mem, user_mem, (unsigned int)orig_size);
        *(char**)mem = mem + size - GUARD_SIZE_IN_BYTES; // tail guard
        fill_guard((uint32_t*)(mem + sizeof(char*)), GUARD_MASK, GUARD_SIZE_IN_BYTES - sizeof(char*));
        fill_guard((uint32_t*)(mem + size - GUARD_SIZE_IN_BYTES), GUARD_MASK, GUARD_SIZE_IN_BYTES);

        return user_mem;
    }
    return 0;
}

void
pyodbc_guarded_dealloc(const char* file, int line, void * user_ptr)
{
    if (user_ptr)
    {
        void *base_ptr = (char*)user_ptr - GUARD_SIZE_IN_BYTES;
        TRACE_NOLOC("guarded dealloc - user: %p base: %p\n",
                    user_ptr, base_ptr);

        pyodbc_check_guards(file, line, user_ptr, "when deallocating");
        free(base_ptr);
    }
}

void
pyodbc_trace_func(const char* file, int line, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    if (0 != file) {
        const char* rel_file = strstr(file, "iopro");
        printf("%s:%d\n", rel_file, line);
    }

    vprintf(fmt, args);
}

void PrintBytes(void* p, size_t len)
{
    unsigned char* pch = (unsigned char*)p;
    for (size_t i = 0; i < len; i++)
        printf("%02x ", (int)pch[i]);
    printf("\n");
}

#ifdef PYODBC_LEAK_CHECK

// THIS IS NOT THREAD SAFE: This is only designed for the
// single-threaded unit tests!

struct Allocation
{
    const char* filename;
    int lineno;
    size_t len;
    void* pointer;
    int counter;
};

static Allocation* allocs = 0;
static int bufsize = 0;
static int count = 0;
static int allocCounter = 0;

void* _pyodbc_malloc(const char* filename, int lineno, size_t len)
{
    void* p = malloc(len);
    if (p == 0)
        return 0;

    if (count == bufsize)
    {
        allocs = (Allocation*)realloc(allocs,
                                      (bufsize + 20) * sizeof(Allocation));
        if (allocs == 0)
        {
            // Yes we just lost the original pointer, but we don't care
            // since everything is about to fail.  This is a debug leak
            // check, not a production malloc that needs to be robust in
            // low memory.
            bufsize = 0;
            count   = 0;
            return 0;
        }
        bufsize += 20;
    }

    allocs[count].filename = filename;
    allocs[count].lineno   = lineno;
    allocs[count].len      = len;
    allocs[count].pointer  = p;
    allocs[count].counter  = allocCounter++;

    printf("malloc(%d): %s(%d) %d %p\n", allocs[count].counter, filename,
           lineno, (int)len, p);

    count += 1;

    return p;
}

void pyodbc_free(void* p)
{
    if (p == 0)
        return;

    for (int i = 0; i < count; i++)
    {
        if (allocs[i].pointer == p)
        {
            printf("free(%d): %s(%d) %d %p i=%d\n", allocs[i].counter,
                   allocs[i].filename, allocs[i].lineno, (int)allocs[i].len,
                   allocs[i].pointer, i);
            memmove(&allocs[i], &allocs[i + 1],
                    sizeof(Allocation) * (count - i - 1));
            count -= 1;
            free(p);
            return;
        }
    }

    printf("FREE FAILED: %p\n", p);
    free(p);
}

void pyodbc_leak_check()
{
    if (count == 0)
    {
        printf("NO LEAKS\n");
    }
    else
    {
        printf("********************************************************************************\n");
        printf("%d leaks\n", count);
        for (int i = 0; i < count; i++)
            printf("LEAK: %d %s(%d) len=%d\n", allocs[i].counter, allocs[i].filename, allocs[i].lineno, allocs[i].len);
    }
}

#endif
