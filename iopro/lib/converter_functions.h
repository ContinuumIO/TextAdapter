#ifndef CONVERTERS_H
#define CONVERTERS_H

#if defined(_MSC_VER) && _MSC_VER < 1600
/* Visual Studio before 2010 didn't have stdint.h */
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
#else
#include <stdint.h>
#endif

#include <string.h>


typedef enum
{
    CONVERT_SUCCESS,
    CONVERT_SUCCESS_TYPE_CHANGED,
    CONVERT_ERROR,
    CONVERT_ERROR_OVERFLOW,
    CONVERT_ERROR_TRUNCATE,
    CONVERT_ERROR_INPUT_TYPE,
    CONVERT_ERROR_INPUT_SIZE,
    CONVERT_ERROR_OUTPUT_SIZE,
    CONVERT_ERROR_INPUT_STRING,
    CONVERT_ERROR_USER_CONVERTER,
    CONVERT_ERROR_OBJECT_CONVERTER,
    CONVERT_ERROR_NUMBA,
    CONVERT_ERROR_LAST
} ConvertError;


typedef enum
{
    UINT_CONVERTER_FUNC,
    INT_CONVERTER_FUNC,
    FLOAT_CONVERTER_FUNC,
    STRING_CONVERTER_FUNC,
    STRING_OBJECT_CONVERTER_FUNC,
    NUM_CONVERTER_FUNCS
} DefaultConverterFuncs;


/* 
 * converter function signature for functions that convert strings to a specific
 * data type and stores in output buffer
 * Inputs:
 *   input: null terminated C string representing value to convert
 *   input_len: length of input (redundant but input string originally was not
 *              null terminated
 *   input_type: indicates type of input (not used by every converter func)
 *   output: pointer to memory block where output value should be stored
 *   output_len: length of output reserved for output value
 *   arg: optional arg value/struct specific to each converter func
 * Output:
 *   error code defined above in ConvertError enum
 */
typedef ConvertError (*converter_func_ptr)(const char *input,
                                           uint32_t input_len,
                                           int input_type,
                                           void *output,
                                           uint32_t output_len,
                                           void *arg);

/* 
 * The following conversion functions follow conversion function signature
 * defined above
 */

/* Convert null terminated C string to signed int */
ConvertError str2int_converter(const char *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg);
/* Convert null terminated C string to unsigned int */
ConvertError str2uint_converter(const char *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg);
/* Convert null terminated C string to float/double */
ConvertError str2float_converter(const char *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg);
/* Copy null terminated C string to output of possibly different length */
ConvertError str2str_converter(void *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg);
/* Convert null terminated C string to complex number */
ConvertError str2complex_converter(void *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg);


/*
 * Extract signed int of various sizes from memory block and cast to
 * signed int64 if needed. Input integer size is specified by input_len argument.
 */
ConvertError get_int_value(void *input, uint32_t input_len, int64_t *value);

/*
 * Extract unsigned int of various sizes from memory block and cast to
 * unsigned int64 if needed. Input integer size is specified by input_len argument.
 */
ConvertError get_uint_value(void *input, uint32_t input_len, uint64_t *value);

/*
 * Extract double/float from from memory block and cast to
 * double if needed. Input floating point size is specified by input_len argument.
 */
ConvertError get_float_value(void *input, uint32_t input_len, double *value);

/*
 * Save signed int64 value to memory block, casting to appropriate output integer
 * size if needed. Output integer size is specified by output_len arg.
 */
ConvertError put_int_value(void *output, uint32_t output_len, int64_t value);

/*
 * Save unsigned int64 value to memory block, casting to appropriate output integer
 * size if needed. Output integer size is specified by output_len arg.
 */
ConvertError put_uint_value(void *output, uint32_t output_len, uint64_t value);

/*
 * Save double/float value to memory block, casting to appropriate output floating
 * point size if needed. Output float size is specified by output_len arg.
 */
ConvertError put_float_value(void *output, uint32_t output_len, double value);

#endif
