import numpy
cimport numpy


cdef ConvertError str2str_object_converter(void *input_str, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg):
    """
    Wrapper function for calling string object converter function
    from low level C api. This is used to convert c strings to python
    string objects.

    Arguments:
    void *input - pointer to value to convert
    uint32_t input_len - length in bytes of input value
    void *output - pointer to memory chunk to store converted value
    uint32_t output_len - size of output memory chunk
    void *arg - pointer to python callable object which does the actual converting

    Returns:
    converted value as a python string object
    """
    cdef ConvertError result = CONVERT_ERROR_OBJECT_CONVERTER
    cdef PyObject **object_ptr
    object_ptr = <PyObject**>output
    cdef kh_string_t *kh_string_table = <kh_string_t*>arg
    cdef int ret
    cdef khiter_t it
    cdef object temp
    cdef char *input_str_copy

    try:
        # Convert c string to Python string object and store in output array
        if object_ptr != NULL:

            # string object hash table exists
            if kh_string_table != NULL:

                # Look for existing string object
                it = kh_get_string(kh_string_table, <char*>input_str)

                # String object doesn't exist, so create and store in output
                # array and hash table
                if it == kh_string_table.n_buckets:
                    temp = (<char*>input_str)[0:input_len].decode(config['encoding'])
                    object_ptr[0] = <PyObject*>temp
                    Py_INCREF(<object>object_ptr[0])
                    input_str_copy = <char*>malloc(input_len+1)
                    strncpy(input_str_copy, <char*>input_str, input_len+1)
                    it = kh_put_string(kh_string_table, <char*>input_str_copy, &ret)
                    kh_string_table.vals[it] = <PyObject*> object_ptr[0]

                # String object exists, so store existing object in array
                else:
                    object_ptr[0] = kh_string_table.vals[it]
                    Py_INCREF(<object>object_ptr[0])

            # No string object hash table exists; just convert and store
            else:
                temp = (<char*>input_str)[0:input_len].decode(config['encoding'])
                object_ptr[0] = <PyObject*>temp
                Py_INCREF(<object>object_ptr[0])

        # Try converting c string to Python string object (for type inference)
        else:
            temp = (<char*>input_str)[0:input_len].decode(config['encoding'])

        result = CONVERT_SUCCESS

    except Exception as e:
        result = CONVERT_ERROR_OBJECT_CONVERTER
 
    return result


cdef ConvertError str2datetime_object_converter(void *input_str, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg):
    """
    Wrapper function for calling string object converter function
    from low level C api. This is used to convert c strings to python
    string objects.

    Arguments:
    void *input - pointer to value to convert
    uint32_t input_len - length in bytes of input value
    void *output - pointer to memory chunk to store converted value
    uint32_t output_len - size of output memory chunk
    void *arg - pointer to python callable object which does the actual converting

    Returns:
    converted value as a python string object
    """
    cdef ConvertError result = CONVERT_ERROR_OBJECT_CONVERTER
    cdef PyObject **object_ptr
    object_ptr = <PyObject**>output
    cdef object temp

    try:
        if object_ptr != NULL:
            temp = str((<char*>input_str)[0:input_len].encode())
            object_ptr[0] = <PyObject*>temp
            Py_INCREF(<object>object_ptr[0])

        result = CONVERT_SUCCESS
    except Exception as e:
        result = CONVERT_ERROR_OBJECT_CONVERTER
 
    return result


cdef ConvertError str2datetime_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg):
    """
    Wrapper function for calling numpy datetime converter function
    from low level C api.

    Arguments:
    void *input - pointer to value to convert
    uint32_t input_len - length in bytes of input value
    void *output - pointer to memory chunk to store converted value
    uint32_t output_len - size of output memory chunk
    void *arg - pointer to python callable object which does the actual converting

    Returns:
    Convert result
    """
    cdef ConvertError result = CONVERT_ERROR_OBJECT_CONVERTER
    cdef numpy.npy_intp dims[1]
    cdef char *temp = <char*>input

    if arg == NULL:
        return CONVERT_ERROR_OBJECT_CONVERTER

    try:
        dtype = <object>arg
        value = dtype.type(<object>temp)
        if output != NULL:
            dims[0] = 1
            array = numpy.PyArray_SimpleNewFromData(1, dims, value.dtype.num, output)
            array.dtype = numpy.dtype(dtype)
            array[0] = value
        result = CONVERT_SUCCESS
    except Exception as e:
        result = CONVERT_ERROR_OBJECT_CONVERTER

    return result


cdef ConvertError python_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg):
    """
    Wrapper function for calling python converter function from low level C api.

    Arguments:
    void *input - pointer to value to convert
    uint32_t input_len - length in bytes of input value
    void *output - pointer to memory chunk to store converted value
    uint32_t output_len - size of output memory chunk
    void *arg - pointer to python callable object which does the actual converting

    Returns:
    Convert result
    """
    cdef numpy.npy_intp dims[1]
    cdef char *temp = <char*>calloc(1, input_len+1)
    cdef bytes py_string
    cdef ConvertError result = CONVERT_ERROR_USER_CONVERTER
    # "input" contains a long string (char*). We only copy "input_len" and make
    # sure that there is a null byte at the end (by using calloc with
    # input_len+1 above)
    memcpy(temp, input, input_len)

    try:
        # Convert "temp" to a Python string (bytes in fact)
        py_string = temp
        # Convert "arg" to Python callable:
        func = <object>arg
        # call python callable object to convert input value
        new_value = func(py_string)

        if isinstance(new_value, numpy.generic):
            data = bytes(new_value.data)
            if output != NULL:
                memcpy(output, <char *>data, output_len)
            result = CONVERT_SUCCESS
        # JNB: not sure if there is a better way to store objects in numpy object array
        elif isinstance(new_value, object):
            if output != NULL:
                dims[0] = 1
                array = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_OBJECT, output)
                array[0] = new_value
            result = CONVERT_SUCCESS
        else:
            result = CONVERT_ERROR_USER_CONVERTER

    except:
        result = CONVERT_ERROR_USER_CONVERTER
    finally:
        free(temp)

    return result


ctypedef uint64_t (*uint_numba_func_ptr)(char *)
ctypedef int64_t (*int_numba_func_ptr)(char *)
ctypedef double (*float_numba_func_ptr)(char *)
ctypedef PyObject* (*object_numba_func_ptr)(char *)
ctypedef int64_t (*datetime_numba_func_ptr)(char *)

cdef ConvertError str2uint_numba_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg):
    cdef uint_numba_func_ptr numba_func = <uint_numba_func_ptr><long>arg
    cdef uint64_t *output_ptr64 = <uint64_t*>output
    cdef uint32_t *output_ptr32 = <uint32_t*>output
    cdef uint16_t *output_ptr16 = <uint16_t*>output
    cdef uint8_t *output_ptr8 = <uint8_t*>output
    cdef uint64_t value

    try:
        if output_len == 8:
            value = <uint64_t>numba_func(<char*>input)
            if output != NULL:
                output_ptr64[0] = value
        elif output_len == 4:
            value = <uint32_t>numba_func(<char*>input)
            if value > 0xffffffff:
                return CONVERT_ERROR_NUMBA
            if output != NULL:
                output_ptr32[0] = value
        elif output_len == 2:
            value = <uint16_t>numba_func(<char*>input)
            if value > 0xffff:
                return CONVERT_ERROR_NUMBA
            if output != NULL:
                output_ptr16[0] = value
        elif output_len == 1:
            value = <uint8_t>numba_func(<char*>input)
            if value > 0xff:
                return CONVERT_ERROR_NUMBA
            if output != NULL:
                output_ptr8[0] = value
        else:
            return CONVERT_ERROR_NUMBA
    except:
        return CONVERT_ERROR_NUMBA
    return CONVERT_SUCCESS

cdef ConvertError str2float_numba_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg):
    cdef float_numba_func_ptr numba_func = <float_numba_func_ptr><long>arg
    cdef float *output_ptr32 = <float*>output
    cdef double *output_ptr64 = <double*>output
    cdef double value

    try:
        if output_len == 4:
            value = <float>numba_func(<char*>input)
            if output != NULL:
                output_ptr32[0] = value
        elif output_len == 8:
            value = <double>numba_func(<char*>input)
            if output != NULL:
                output_ptr64[0] = value
        else:
            return CONVERT_ERROR_NUMBA
    except:
        return CONVERT_ERROR_NUMBA
    return CONVERT_SUCCESS

cdef ConvertError str2datetime_numba_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg):
    cdef datetime_numba_func_ptr numba_func = <datetime_numba_func_ptr><long>arg
    cdef int64_t *output_ptr = <int64_t*>output
    cdef int64_t value

    try:
        if output_len == 8:
            value = <int64_t>numba_func(<char*>input)
            if output != NULL:
                output_ptr[0] = value
        else:
            return CONVERT_ERROR_NUMBA
    except:
        return CONVERT_ERROR_NUMBA
    return CONVERT_SUCCESS

cdef ConvertError str2object_numba_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg):
    cdef object_numba_func_ptr numba_func = <object_numba_func_ptr><long>arg
    cdef PyObject **output_ptr = <PyObject**>output
    cdef object value

    try:
        value = <object>numba_func(<char*>input)
        if output != NULL:
            output_ptr[0] = <PyObject*>value
    except:
        return CONVERT_ERROR_NUMBA

    return CONVERT_SUCCESS



