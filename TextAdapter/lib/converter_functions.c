#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include "converter_functions.h"


typedef enum float_state_t
{
    INTEGER,
    FRACTION,
    EXPONENT_SIGN,
    EXPONENT,
    TRAILING_WHITESPACE
} FloatState;

/* Custom string to float conversion function, because strtod is REALLY slow */
ConvertError string_to_float(const char *input, uint32_t input_len, void *output,
    uint32_t output_len)
{
    uint32_t offset = 0;
    uint64_t fraction = 1;
    uint8_t exponent = 0;
    int sign = 1;
    int exponent_sign = 1;
    FloatState state = INTEGER;
    double value = 0.0;

    if (output != NULL)
        memset((char*)output, 0, output_len);

    if (input_len == 0)
        return CONVERT_ERROR_INPUT_STRING;

    while (offset < input_len
           && (input[offset] == ' ' || input[offset] == '\t'))
    {
        offset++;
    }

    if (input[offset] == '-')
    {
        sign = -1;
        offset++;
    }
    else if (input[offset] == '+')
    {
        offset++;
    }

    while (offset < input_len)
    {
        switch (state)
        {
            case INTEGER:
                if (input[offset] >= '0' && input[offset] <= '9')
                {
                    value *= 10;
                    value += input[offset] - '0';
                }
                else if (input[offset] == '.')
                {
                    state = FRACTION;
                }
                else if (input[offset] == 'E' || input[offset] == 'e')
                {
                    state = EXPONENT_SIGN;
                }
                else if (input[offset] == ' ' || input[offset] == '\t')
                {
                    state = TRAILING_WHITESPACE;
                }
                else
                {
                    return CONVERT_ERROR_INPUT_STRING;
                }
                break;
            case FRACTION:
                if (input[offset] >= '0' && input[offset] <= '9')
                {
                    value *= 10;
                    value += input[offset] - '0';
                    fraction *= 10;
                }
                else if (input[offset] == 'E' || input[offset] == 'e')
                {
                    state = EXPONENT_SIGN;
                }
                else if (input[offset] == ' ' || input[offset] == '\t')
                {
                    state = TRAILING_WHITESPACE;
                }
                else
                {
                    return CONVERT_ERROR_INPUT_STRING;
                }
                break;
            case EXPONENT_SIGN:
                if (input[offset] == '+')
                {
                    exponent_sign = 1;
                    state = EXPONENT;
                }
                else if (input[offset] == '-')
                {
                    exponent_sign = -1;
                    state = EXPONENT;
                }
                else
                {
                    return CONVERT_ERROR_INPUT_STRING;
                }
                break;
            case EXPONENT:
                if (input[offset] >= '0' && input[offset] <= '9')
                {
                    exponent *= 10;
                    exponent += input[offset] - '0';
                }
                else if (input[offset] == ' ' || input[offset] == '\t')
                {
                    state = TRAILING_WHITESPACE;
                }
                else
                {
                    return CONVERT_ERROR_INPUT_STRING;
                }
                break;
            case TRAILING_WHITESPACE:
                if (input[offset] == ' ' || input[offset] == '\t')
                {
                }
                else
                {
                    return CONVERT_ERROR_INPUT_STRING;
                }
                break;
        }

        offset++;
    }

    value /= fraction;

    while (exponent > 0)
    {
        if (exponent_sign == 1)
            value *= 10;
        else if (exponent_sign == -1)
            value /= 10;
        exponent--;
    }

    if (output != NULL)
    {
        if (output_len == sizeof(double))
        {
            *(double*)output = value * sign;
        }
        else if (output_len == sizeof(float))
        {
            if (value < -FLT_MAX || value > FLT_MAX)
                return CONVERT_ERROR_OVERFLOW;
            *(float*)output = (float)value * sign;
        }
        else
        {
            return CONVERT_ERROR_OUTPUT_SIZE;
        }
    }

    return CONVERT_SUCCESS;
}


ConvertError get_int_value(void *input, uint32_t input_len, int64_t *value)
{
    ConvertError result = CONVERT_SUCCESS;

    if (input_len == sizeof(int8_t))
    {
        *value = *(int8_t*)input;
    }
    else if (input_len == sizeof(int16_t))
    {
        *value = *(int16_t*)input;
    }
    else if (input_len == sizeof(int32_t))
    {
        *value = *(int32_t*)input;
    }
    else if (input_len == sizeof(int64_t))
    {
        *value = *(int64_t*)input;
    }
    else
    {
        *value = 0;
        result = CONVERT_ERROR_INPUT_SIZE;
    }
    
    return result;
}

ConvertError get_uint_value(void *input, uint32_t input_len, uint64_t *value)
{
    ConvertError result = CONVERT_SUCCESS;

    if (input_len == sizeof(uint8_t))
    {
        *value = *(uint8_t*)input;
    }
    else if (input_len == sizeof(uint16_t))
    {
        *value = *(uint16_t*)input;
    }
    else if (input_len == sizeof(uint32_t))
    {
        *value = *(uint32_t*)input;
    }
    else if (input_len == sizeof(uint64_t))
    {
        *value = *(uint64_t*)input;
    }
    else
    {
        *value = 0;
        result = CONVERT_ERROR_INPUT_SIZE;
    }

    return result;
}


ConvertError get_float_value(void *input, uint32_t input_len, double *value)
{
    ConvertError result = CONVERT_SUCCESS;

    if (input_len == sizeof(float))
    {
        *value = *(float*)input;
    }
    else if (input_len == sizeof(double))
    {
        *value = *(double*)input;
    }
    else
    {
        *value = 0.0;
        result = CONVERT_ERROR_INPUT_SIZE;
    }

    return result;
}


ConvertError put_int_value(void *output, uint32_t output_len, int64_t value)
{
    ConvertError result = CONVERT_SUCCESS;

    if (output_len == sizeof(int8_t))
    {
        *(int8_t*)output = (int8_t)value;
        if (value < INT8_MIN || value > INT8_MAX)
            result = CONVERT_ERROR_OVERFLOW;
    }
    else if (output_len == sizeof(int16_t))
    {
        *(int16_t*)output = (int16_t)value;
        if (value < INT16_MIN || value > INT16_MAX)
            result = CONVERT_ERROR_OVERFLOW;
    }
    else if (output_len == sizeof(int32_t))
    {
        *(int32_t*)output = (int32_t)value;
        if (value < INT32_MIN || value > INT32_MAX)
            result = CONVERT_ERROR_OVERFLOW;
    }
    else if (output_len == sizeof(int64_t))
    {
        *(int64_t*)output = (int64_t)value;
    }
    else
    {
        result = CONVERT_ERROR_OUTPUT_SIZE;
    }

    return result;
}

ConvertError put_uint_value(void *output, uint32_t output_len, uint64_t value)
{
    ConvertError result = CONVERT_SUCCESS;

    if (output_len == sizeof(uint8_t))
    {
        *(uint8_t*)output = (uint8_t)value;
        if (value > UINT8_MAX)
            result = CONVERT_ERROR_OVERFLOW;
    }
    else if (output_len == sizeof(uint16_t))
    {
        *(uint16_t*)output = (uint16_t)value;
        if (value > UINT16_MAX)
            result = CONVERT_ERROR_OVERFLOW;
    }
    else if (output_len == sizeof(uint32_t))
    {
        *(uint32_t*)output = (uint32_t)value;
        if (value > UINT32_MAX)
            result = CONVERT_ERROR_OVERFLOW;
    }
    else if (output_len == sizeof(uint64_t))
    {
        *(uint64_t*)output = (uint64_t)value;
    }
    else
    {
        result = CONVERT_ERROR_OUTPUT_SIZE;
    }

    return result;
}


ConvertError put_float_value(void *output, uint32_t output_len, double value)
{
    ConvertError result = CONVERT_SUCCESS;

    if (output_len == sizeof(float))
    {
        *(float*)output = (float)value;
        if (value < FLT_MIN || value > FLT_MAX)
            result = CONVERT_ERROR_OVERFLOW;
    }
    else if (output_len == sizeof(double))
    {
        *(double*)output = (double)value;
    }
    else
    {
        result = CONVERT_ERROR_OUTPUT_SIZE;
    }

    return result;
}


/* Error checking for strtoll/strtoi64 calls */
ConvertError check_strtox_result(const char *input, uint32_t input_len,
    char *invalid, int errno_value)
{
    ConvertError result = CONVERT_SUCCESS;

    assert(input != NULL);

    if (errno_value == ERANGE || invalid - input == 0)
    {
        result = CONVERT_ERROR_INPUT_STRING;
    }
    else if (invalid - input < input_len)
    {
        uint64_t offset = invalid - input;
        int found_nonspace = 0;

        // If conversion stopped at a decimal point, continuous zeros
        // area allowed before hitting whitespace
        if (input[offset] == '.')
        {
            offset++;

            while (offset < input_len && input[offset] == '0')
            {
                offset++;
            }
        }

        found_nonspace = 0;
        while (offset < input_len)
        {
            if (!isspace(*(input + offset)))
                found_nonspace = 1;
            offset++;
        }

        if (found_nonspace == 1)
        {
            result = CONVERT_ERROR_INPUT_STRING;
        }
        else
        {
            result = CONVERT_SUCCESS;
        }
    }

    return result;
}


ConvertError str2int_converter(const char *input, uint32_t input_len, int input_type,
    void *output, uint32_t output_len, void *arg)
{
    ConvertError result = CONVERT_SUCCESS;
    char *invalid;
    int64_t value;
   
    #ifdef DEBUG_ADAPTER
    {
        char *temp = calloc(input_len + 1, sizeof(char));
        memcpy(temp, input, input_len);
        printf("str2int_converter(): input=%s\n", temp);
        free(temp);
    }
    #endif

    invalid = NULL;
    errno = 0;
#if !defined(_WIN32)
    value = strtoll(input, &invalid, 10);
#else
    value = _strtoi64(input, &invalid, 10);
#endif
    result = check_strtox_result(input, input_len, invalid, errno);

    if (result == CONVERT_SUCCESS && output != NULL)
        result = put_int_value(output, output_len, value);

    #ifdef DEBUG_ADAPTER
    if (output != NULL)
    {
        if (output_len == sizeof(int8_t))
            printf("int_converter(): output=%d\n", *(int8_t*)output);
        if (output_len == sizeof(int16_t))
            printf("int_converter(): output=%d\n", *(int16_t*)output);
        if (output_len == sizeof(int32_t))
            printf("int_converter(): output=%d\n", *(int32_t*)output);
        if (output_len == sizeof(int64_t))
            printf("int_converter(): output=%lld\n", *(int64_t*)output);
    }
    #endif

    return result;
}


ConvertError str2uint_converter(const char *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg)
{
    ConvertError result = CONVERT_SUCCESS;
    char *invalid;
    uint64_t value;

    #ifdef DEBUG_ADAPTER
    {
        char *temp = calloc(input_len + 1, sizeof(char));
        memcpy(temp, input, input_len);
        printf("str2uint_converter(): input=%s\n", temp);
        free(temp);
    }
    #endif

    while (input_len > 0 && (*(char*)input == ' ' || *(char*)input == '\t')) {
        input_len--;
        input = ((char*)input) + 1;
    }

    if (input_len > 0 && *(char*)input == '-') {
        return CONVERT_ERROR_INPUT_TYPE;
    }

    invalid = NULL;
    errno = 0;
#if !defined(_WIN32)
    value = strtoull(input, &invalid, 10);
#else
    value = _strtoui64(input, &invalid, 10);
#endif
    result = check_strtox_result(input, input_len, invalid, errno);

    if (result == CONVERT_SUCCESS && output != NULL)
        result = put_uint_value(output, output_len, value);

    #ifdef DEBUG_ADAPTER
    if (output != NULL)
    {
        if (output_len == sizeof(uint8_t))
            printf("uint_converter(): output=%u\n", *(uint8_t*)output);
        if (output_len == sizeof(uint16_t))
            printf("uint_converter(): output=%u\n", *(uint16_t*)output);
        if (output_len == sizeof(uint32_t))
            printf("uint_converter(): output=%u\n", *(uint32_t*)output);
        if (output_len == sizeof(uint64_t))
            printf("uint_converter(): output=%llu\n", *(uint64_t*)output);
    }
    #endif

    return result;
}


ConvertError str2float_converter(const char *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg)
{
    ConvertError result = CONVERT_SUCCESS;

    #ifdef DEBUG_ADAPTER
    {
        char *temp = calloc(input_len + 1, sizeof(char));
        memcpy(temp, input, input_len);
        printf("float_converter(): converting token=%s\n", temp);
        free(temp);
    }
    #endif

    result = string_to_float(input, input_len, output, output_len);
    
    #ifdef DEBUG_ADAPTER
    if (output != NULL)
    {
        if (output_len == sizeof(float))
            printf("float_converter(): output=%f\n", *(float*)output);
        else if (output_len == sizeof(double))
            printf("float_converter(): output=%lf\n", *(double*)output);
    }
    #endif

    return result;
}


ConvertError str2str_converter(void *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg)
{
    ConvertError result = CONVERT_SUCCESS;
    char *start = (char*)input;

    #ifdef DEBUG_ADAPTER
    {
        char *temp = calloc(input_len + 1, sizeof(char));
        memcpy(temp, input, input_len);
        printf("string_converter(): converting token=%s\n", temp);
        free(temp);
    }
    #endif

    if (output != NULL)
        memset(output, '\0', (size_t)output_len);

    if (input_len > 0)
    {
        uint32_t len = input_len;
        if (output_len < len)
        {
            len = output_len;
            result = CONVERT_ERROR_OVERFLOW;
        }
        if (output != NULL)
        {
            memcpy(output, start, (size_t)len);
        }
    }

    return result;
}


ConvertError str2complex_converter(void *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg)
{
    ConvertError result = CONVERT_SUCCESS;

    double real = 0.0;
    double imag = 0.0;

    uint32_t real_offset;
    uint32_t imag_offset;
    char * token = (char*)input;

    int is_digit;


    #ifdef DEBUG_ADAPTER
    {    
        char *temp = calloc(input_len + 1, sizeof(char));
        memcpy(temp, input, input_len);
        printf("complex_converter(): converting token=%s\n", temp);
        free(temp);
    }
    #endif

    /* find start of real number */
    real_offset = 0;
    is_digit = isdigit(token[real_offset]);
    while (real_offset < input_len && !is_digit)
    {
        real_offset++;
    }

    /* find start of imaginary number */
    imag_offset = real_offset;
    while (imag_offset < input_len && token[imag_offset] != '+')
    {
        imag_offset++;
    }
    imag_offset++;

    if (real_offset >= input_len || imag_offset >= input_len)
    {
        result = CONVERT_ERROR_INPUT_STRING;
    }
    else
    {
        char *invalid;
        char *temp = calloc(1, (size_t)input_len + 1);
        memcpy(temp, input, (size_t)input_len);

        invalid = NULL;

        errno = 0;
        real = strtod(temp+real_offset, &invalid);
        if (invalid - (char*)temp < input_len || errno == ERANGE)
        {
            result = CONVERT_ERROR_INPUT_STRING;
        }
        else
        {
            invalid = NULL;

            errno = 0;
            imag = strtod(temp+imag_offset, &invalid);
            if (invalid - (char*)temp < input_len || errno == ERANGE)
            {
                result = CONVERT_ERROR_INPUT_STRING;
            }
            else
            {
                *(float*)output = (float)real;
                *((float*)output+1) = (float)imag;
            }
        }

        free(temp);
    }

    return result;
}


