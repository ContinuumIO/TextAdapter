
cdef InputData* open_s3(object data):
    """
    Set up read/seek functions for S3 data source
    """
    cdef InputData *input_data = <InputData*>calloc(1, sizeof(InputData))
    input_data.seek = <seek_func_ptr>&seek_s3
    input_data.read = <read_func_ptr>&read_s3
    input_data.close = <close_func_ptr>&close_s3
    input_data.input = <void*>data
    return input_data

cdef void close_s3(InputData *input_data):
    """
    Clean up InputData for S3 data source
    """
    if input_data != NULL:
        free(input_data)

cdef AdapterError seek_s3(InputData *input, uint64_t offset):
    """
    Seek to offset in S3 data source

    Arguments:
    input - InputData struct
    offset - offset to seek to
    """
    if (input == NULL):
        return ADAPTER_ERROR_SEEK;

    s3_input = <object>input.input

    s3_key = s3_input['s3_key']
    if offset > (s3_key.size - input.header):
        return ADAPTER_ERROR_SEEK_S3

    s3_input['offset'] = offset + input.header
    return ADAPTER_SUCCESS


cdef AdapterError read_s3(InputData *input, char *buffer, uint64_t buffer_len, uint64_t *num_bytes_read):
    """
    Read bytes from S3 data source and store in buffer.

    Arguments:
    input - text adapter struct
    buffer - output buffer for data read from S3
    buffer_len - length of buffer
    num_bytes_read - pointer to variable for storing number of bytes read from S3
    """
    if (input == NULL):
        return ADAPTER_ERROR_SEEK;

    s3_input = <object>input.input
    offset = s3_input['offset']
    s3_key = s3_input['s3_key']

    if offset >= s3_key.size:
        num_bytes_read[0] = 0
        return ADAPTER_ERROR_READ_EOF

    if offset < 0:
        return ADAPTER_ERROR_READ

    try:
        data = s3_key.get_contents_as_string(headers={'Range' : 'bytes={0}-{1}'.format(offset, offset+buffer_len)})
    except:
        return ADAPTER_ERROR_READ_S3
    data_len = len(data)

    if data_len > buffer_len:
        data_len = buffer_len

    memcpy(buffer, <char*>data, data_len)
    num_bytes_read[0] = data_len

    s3_input['offset'] = s3_input['offset'] + data_len

    return ADAPTER_SUCCESS
