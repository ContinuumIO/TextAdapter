#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <sys/stat.h>
#include <assert.h>
#include "text_adapter.h"


/* Default buffer size for reading in chunks of text from data source */
#define TOKENIZE_BUFFER_SIZE 1024*1024*10


/* these come from json_tokenizer.c ... */
extern int push(JSON_checker jc, int mode);
extern int pop(JSON_checker jc, int mode);

converter_func_ptr default_converters[NUM_CONVERTER_FUNCS];

ConvertErrorInfo convert_error_info;


/* Clear error info.
   This should be called every time an api function is called. */
void clear_error_info(void)
{
    if (convert_error_info.token)
        free(convert_error_info.token);

    memset(&convert_error_info, 0, sizeof(ConvertErrorInfo));
}

ConvertErrorInfo get_error_info(void)
{
    return convert_error_info;
}


void reset_buffer(TextAdapterBuffer *buffer)
{
    buffer->size = 0;
    buffer->bytes_processed = 0;
    buffer->eof = 0;
}


/* Allocate and initialize new TextAdapter struct */
TextAdapter* open_text_adapter(InputData *input_data)
{
    TextAdapter *adapter;
    if (input_data == NULL)
    {
        return NULL;
    }

    adapter = (TextAdapter *)calloc(1, sizeof(TextAdapter));

    adapter->input_data = input_data;
    adapter->input_data->start_record = 0;
    adapter->input_data->start_offset = 0;
    adapter->input_data->header = 0;
    adapter->input_data->footer = 0;

    /* Extra byte is so buffer is always null terminated */
    adapter->buffer.data = calloc(1, TOKENIZE_BUFFER_SIZE + 1);

    /* buffer size is set when we fill it with data */
    /* buffer size of zero indicates it needs to be filled */
    reset_buffer(&adapter->buffer);

    adapter->reset_json_args = 0;

    clear_error_info();

    adapter->infer_types_mode = 0;

    default_converters[UINT_CONVERTER_FUNC] = &str2uint_converter;
    default_converters[INT_CONVERTER_FUNC] = &str2int_converter;
    default_converters[FLOAT_CONVERTER_FUNC] = &str2float_converter;
    default_converters[STRING_CONVERTER_FUNC] = &str2str_converter;

    adapter->fields = calloc(1, sizeof(FieldList));

    adapter->group_whitespace_delims = 0;
    adapter->any_whitespace_as_delim = 0;

    adapter->skipblanklines = 1;

    return adapter;
}


/* Deallocate TextAdapter struct */
void close_text_adapter(TextAdapter *adapter)
{
    if (adapter == NULL)
        return;

    if (adapter->buffer.data)
    {
        free(adapter->buffer.data);
    }

    clear_fields(adapter->fields);
    free(adapter->fields);
    
    clear_error_info();

    free(adapter);
}


/* Seek to specified record in input data stream.
   If index for data has been built then retrieve offset for record,
   otherwise just read from beginning of data until correct record
   has been found. */
AdapterError seek_record(TextAdapter *adapter, uint64_t rec_num)
{
    uint64_t num_records = 0;
    AdapterError result = ADAPTER_ERROR_SEEK;
    RecordOffset record_offset;

    if (adapter == NULL)
        return result;

    #ifdef DEBUG_ADAPTER
    printf("seek_record(): record_num=%llu header=%llu\n",
            rec_num, adapter->input_data->header);
    #endif

    if (adapter->num_records > 0 && rec_num >= adapter->num_records) {
        convert_error_info.record_num = rec_num;
        return ADAPTER_ERROR_SEEK_EOF;
    }

    reset_buffer(&adapter->buffer);

    if (adapter->reset_json_args) {
        JsonTokenizerArgs *json_args = (JsonTokenizerArgs *)adapter->tokenize_args;
        if (json_args->jc != NULL) {
            reject(json_args->jc);
        }
        json_args->jc = new_JSON_checker(20);
    }

    if (rec_num == 0)
    {
        /* Seek to beginning of input */
        if (adapter->input_data->seek_compressed)
        {
            result = adapter->input_data->seek_compressed(adapter->input_data, 0);
        }
        else
        {
            result = adapter->input_data->seek(adapter->input_data, 0);
        }
        
        adapter->input_data->start_record = 0;
    }
    else if (adapter->index)
    {
        record_offset = adapter->index_lookup(adapter->index, rec_num);

        /* Seek to offset in input */
        if (adapter->input_data->seek_compressed)
            result = adapter->input_data->seek_compressed(adapter->input_data,
                                                          record_offset.offset);
        else
            result = adapter->input_data->seek(adapter->input_data,
                                              record_offset.offset);
        adapter->input_data->start_record = record_offset.record_num;

        if (result == ADAPTER_SUCCESS)
        {
            /* We've seeked to last indexed record before the one we're looking
               for. Now we parse records unti we find it. */
            if (record_offset.record_num < rec_num)
            {
                result = read_records(adapter,
                                      rec_num - record_offset.record_num,
                                      1, 0, &num_records);
            }
        }
    }
    else
    {
        /* Seek to beginning of input */
        if (adapter->input_data->seek_compressed)
        {
            result = adapter->input_data->seek_compressed(adapter->input_data, 0);
        }
        else
        {
            result = adapter->input_data->seek(adapter->input_data, 0);
        }
        adapter->input_data->start_record = 0;

        if (result == ADAPTER_SUCCESS)
        {
            /* If not seeking first record,
               read until we get to correct record */
            if (rec_num > 0)
            {
                result = read_records(adapter, rec_num, 1, 0, &num_records);
            }
        }
    }


    return result;
}


AdapterError refresh_buffer(TextAdapterBuffer *buffer, InputData *input_data)
{
    AdapterError result = ADAPTER_SUCCESS;
    uint64_t num_bytes_read = 0;
    uint64_t bytes_left = buffer->size - buffer->bytes_processed;

    /* If we haven't processed all the bytes from the current buffer
       (because the end of the buffer falls in the middle of a token),
       then copy the leftover bytes to the beginning of buffer */
    if (bytes_left > 0)
    {
        memcpy(buffer->data,
            buffer->data + buffer->bytes_processed,
            (size_t)bytes_left);
    }

    /* Set offset of beginning of data stream */
    input_data->start_offset += buffer->bytes_processed;

    buffer->eof = 0;

    memset(buffer->data + bytes_left,
        '\0', (size_t)TOKENIZE_BUFFER_SIZE - bytes_left);

    if (input_data->read_compressed)
    {
        result = input_data->read_compressed(input_data,
            buffer->data + bytes_left,
            TOKENIZE_BUFFER_SIZE - bytes_left,
            &num_bytes_read);
    }
    else
    {
        result = input_data->read(input_data,
            buffer->data + bytes_left,
            TOKENIZE_BUFFER_SIZE - bytes_left,
            &num_bytes_read);
    }

    #ifdef DEBUG_ADAPTER
    printf("read_tokens(): read %u bytes into buffer %d\n",
            num_bytes_read, result);
    #endif

    if (num_bytes_read == 0)
    {
        buffer->eof = 1;
        return ADAPTER_ERROR_READ_EOF;
    }
    else if (result != ADAPTER_SUCCESS)
    {
        return result;
    }

    /* If we couldn't fill up the rest of the buffer, then we're out of
       data to read. Set the buffer eof flag so the process_token
       function will know that it needs to process the last piece of data
       in the buffer as a token even if it doesn't end in a delimiter. */
    if (num_bytes_read < TOKENIZE_BUFFER_SIZE - bytes_left)
    {
        buffer->eof = 1;
    }

    buffer->size = num_bytes_read + bytes_left;
    buffer->bytes_processed = 0;

    return ADAPTER_SUCCESS;
}


/* Parses specified number of tokens from data source and converts to data type.
   Function returns when all tokens have been found or end of data. */
AdapterError read_tokens(TextAdapter *adapter,
    uint64_t num_tokens,
    uint64_t step,
    char *output,
    uint64_t *num_tokens_found,
    int enable_index)
{
    AdapterError result = ADAPTER_ERROR_READ_TOKENS;

    uint64_t bytes_read = 0;
    uint64_t bytes_left = 0;

    assert(adapter != NULL);

    *num_tokens_found = 0;

    while (*num_tokens_found < num_tokens)
    {
        /* Check to see if we need to refresh the buffer */
        if (adapter->buffer.size == 0)
        {
            /* If we haven't processed all the bytes from the current buffer
               (because the end of the buffer falls in the middle of a token),
               then copy the leftover bytes to the beginning of buffer */
            if (bytes_left > 0)
            {
                memcpy(adapter->buffer.data,
                        adapter->buffer.data + adapter->buffer.bytes_processed,
                        (size_t)bytes_left);
            }

            /* Set offset of beginning of data stream */
            adapter->input_data->start_offset += adapter->buffer.bytes_processed;

            bytes_read = 0;
            adapter->buffer.eof = 0;

            memset(adapter->buffer.data + bytes_left,
                    '\0', (size_t)TOKENIZE_BUFFER_SIZE - bytes_left);

            if (adapter->input_data->read_compressed)
                result = adapter->input_data->read_compressed(adapter->input_data,
                        adapter->buffer.data + bytes_left,
                        TOKENIZE_BUFFER_SIZE - bytes_left,
                        &bytes_read);
            else
                result = adapter->input_data->read(adapter->input_data,
                        adapter->buffer.data + bytes_left,
                        TOKENIZE_BUFFER_SIZE - bytes_left,
                        &bytes_read);

            #ifdef DEBUG_ADAPTER
            printf("read_tokens(): read %u bytes into buffer %d\n",
                    bytes_read, result);
            #endif
      
            if (bytes_read == 0)
            {
                result = ADAPTER_ERROR_READ_EOF;
                break;
            }
            else if (result != ADAPTER_SUCCESS)
            {
                break;
            }

            /* If we couldn't fill up the rest of the buffer, then we're out of
               data to read. Set the buffer eof flag so the process_token
               function will know that it needs to process the last piece of data
               in the buffer as a token even if it doesn't end in a delimiter. */
            if (bytes_read < TOKENIZE_BUFFER_SIZE - bytes_left)
            {
                adapter->buffer.eof = 1;
            }
 
            adapter->buffer.size = bytes_read + bytes_left;
            adapter->buffer.bytes_processed = 0;
        }

        /* parse and process all the tokens in the buffer */
        result = adapter->tokenize(adapter, num_tokens, step, &output,
                num_tokens_found, enable_index, adapter->index_density);

        if (result != ADAPTER_SUCCESS)
            break;
        
        /* We haven't found all the tokens yet, so set the buffer size to zero.
           This will trigger a buffer refresh at top of loop. */
        if (*num_tokens_found < num_tokens)
        {
            bytes_left = adapter->buffer.size - adapter->buffer.bytes_processed;
            adapter->buffer.size = 0;

            #ifdef DEBUG_ADAPTER
            printf("read_tokens(): found %llu/%llu tokens, %llu bytes processed in buffer, %u bytes left in buffer\n",
                   *num_tokens_found, num_tokens,
                   adapter->buffer.bytes_processed,
                   bytes_left);
            #endif
        }
    }

    return result;
}


/* Read specified number of records and store in specified memory block.
   This is essentially a wrapper for read_tokens which does all the heavy
   lifting with managing the input buffer.
   If output memory block is NULL, we just want to parse input without
   converting tokens (for example if we're seeking to a specific record). */
AdapterError read_records(TextAdapter *adapter, uint64_t num_records,
    uint64_t step, char *output, uint64_t *num_records_found)
{
    AdapterError result = ADAPTER_ERROR_READ_RECORDS;
    uint64_t num_tokens;
    int enable_index = 0;
    uint64_t num_tokens_found = 0;
    uint64_t num_fields = adapter->fields->num_fields;

    #ifdef DEBUG_ADAPTER
    printf("read_records(): num_records=%llu step=%llu\n", num_records, step);
    #endif

    if (adapter == NULL)
        return result;

    if (adapter->num_records > 0 &&
            num_records > (adapter->num_records - adapter->input_data->start_record)) {
        return ADAPTER_ERROR_READ_EOF;
    }

    if (step == 0)
        return result;

    if (num_records_found != NULL)
        *num_records_found = 0;

    /* If no converter functions have been set for any fields, then bail */
    if (num_fields == 0)
        return ADAPTER_ERROR_NO_FIELDS;

    /* Calculate number of tokens we want tokenize function to find. */
    /* If number of records passed in is zero, set number of tokens */
    /* to max value to process whole file. */
    num_tokens = num_records * num_fields;

    if (num_tokens == 0)
    {
        num_tokens = UINT_MAX;
    }

    /* Clear output memory block */
    if (output)
    {
        uint32_t output_rec_size = get_output_record_size(adapter->fields);
        memset(output, 0, output_rec_size * ceil(num_records / abs(step)));
    }

    /* Read and process number of tokens and we calculated above */
    if (adapter->index != NULL)
        enable_index = 1;
    
    clear_error_info();

    result = adapter->tokenize(adapter, num_tokens, step, &output,
        &num_tokens_found, enable_index, adapter->index_density);

    /* Calculate number of records found */
    if (num_records_found != NULL)
    {
        *num_records_found = ceil(((float)num_tokens_found / num_fields) / abs(step));
    }

    #ifdef DEBUG_ADAPTER
    if (num_records_found != NULL)
    {
        printf("read_records(): %llu records found\n", *num_records_found);
    }
    #endif

    adapter->input_data->start_record += num_tokens_found / num_fields;

    return result;
}


AdapterError build_index(TextAdapter *adapter)
{
    AdapterError result = ADAPTER_ERROR_INDEX;
    tokenize_func_ptr temp_func;
    uint64_t num_tokens_found = 0;
    uint64_t num_fields;
    char *output;

    #ifdef DEBUG_ADAPTER
    printf("build_index() building index...\n");
    #endif

    if (adapter == NULL)
        return result;

    if (adapter->index == NULL)
        return result;

    if (adapter->input_data->seek_compressed)
        adapter->input_data->seek_compressed(adapter->input_data, 0);
    else
        adapter->input_data->seek(adapter->input_data, 0);

    temp_func = adapter->tokenize;
    adapter->tokenize = &record_tokenizer;
    num_fields = adapter->fields->num_fields;
    adapter->fields->num_fields = 1;

    clear_error_info();
    reset_buffer(&adapter->buffer);

    if (adapter->reset_json_args) {
        JsonTokenizerArgs *json_args = (JsonTokenizerArgs *)adapter->tokenize_args;
        if (json_args->jc != NULL) {
            reject(json_args->jc);
        }
        json_args->jc = new_JSON_checker(20);
    }

    output = NULL;
    result = adapter->tokenize(adapter, UINT_MAX, 1, &output,
        &num_tokens_found, 1, adapter->index_density);

    adapter->tokenize = temp_func;
    adapter->fields->num_fields = num_fields;

    adapter->num_records = num_tokens_found;

    return result;
}


AdapterError build_gzip_index(TextAdapter *adapter)
{
    GzipInput *gzip_input;
    unsigned char buffer[UNCOMPRESSED_WINDOW_SIZE];
    uint64_t bytes_read;
    int totin;
    int totout;
    int last;
    int z_result;
    unsigned char *window = malloc((UNCOMPRESSED_WINDOW_SIZE + 1) * sizeof(char));

    AdapterError result = ADAPTER_ERROR_INDEX;
    AdapterError read_result;

    #ifdef DEBUG_ADAPTER
    printf("build_gzip_index() building gzip index...\n");
    #endif

    if (adapter == NULL)
        return result;

    result = build_index(adapter);

    gzip_input = (GzipInput*)adapter->input_data->compressed_input;

    memset(buffer, '\0', UNCOMPRESSED_WINDOW_SIZE);
    gzip_input->z->next_out = (unsigned char *)buffer;
    gzip_input->z->avail_out = UNCOMPRESSED_WINDOW_SIZE;
    bytes_read = 0;
    totin = 0;
    totout = 0;
    last = 0;
    
    adapter->input_data->seek(adapter->input_data, 0 - adapter->input_data->header);
    inflateInit2(gzip_input->z, 47);

    do
    {
        char prebuffer[COMPRESSED_BUFFER_SIZE];
        memset(prebuffer, '\0', COMPRESSED_BUFFER_SIZE);

        read_result = adapter->input_data->read(adapter->input_data,
                prebuffer, COMPRESSED_BUFFER_SIZE, &bytes_read);

        if (read_result != ADAPTER_SUCCESS &&
            read_result != ADAPTER_ERROR_READ_EOF)
        {
            return result;
        }
    
        #ifdef DEBUG_ADAPTER
        printf("build_gzip_index() compressed bytes_read=%u\n", bytes_read);
        #endif

        gzip_input->z->avail_in = bytes_read;
        gzip_input->z->next_in = (unsigned char *)prebuffer;

        do
        {
            if (gzip_input->z->avail_out == 0)
            {
                gzip_input->z->avail_out = UNCOMPRESSED_WINDOW_SIZE;
                gzip_input->z->next_out = buffer;
            }

            totin += gzip_input->z->avail_in;
            totout += gzip_input->z->avail_out;
            z_result = inflate(gzip_input->z, Z_BLOCK);
            totin -= gzip_input->z->avail_in;
            totout -= gzip_input->z->avail_out;

            if ((totout == 0 || totout - last > GZIP_ACCESS_POINT_DISTANCE) &&
                ((gzip_input->z->data_type & 128) && !(gzip_input->z->data_type & 64)))
            {
                gzip_input->buffer_refreshed = 0;
                last = totout;

                if (gzip_input->z->avail_out > 0)
                    memcpy(window, buffer + UNCOMPRESSED_WINDOW_SIZE - gzip_input->z->avail_out,
                            gzip_input->z->avail_out);
                if (gzip_input->z->avail_out < UNCOMPRESSED_WINDOW_SIZE)
                    memcpy(window + gzip_input->z->avail_out,
                            buffer, UNCOMPRESSED_WINDOW_SIZE - gzip_input->z->avail_out);
                window[UNCOMPRESSED_WINDOW_SIZE] = '\0';

                adapter->add_gzip_access_point(adapter->index, window, totin, totout,
                    0, 0, gzip_input->z->data_type & 7);
            }
        }
        while (gzip_input->z->avail_in > 0);

    } while (z_result != Z_STREAM_END);
    
    free(window);

    return ADAPTER_SUCCESS;
}


ConvertError try_converter(char *input, uint64_t input_len, char *output,
                           FieldInfo *field_info,
                           int infer_type)
{
    ConvertError result;
    char temp;

    #ifdef DEBUG_ADAPTER
    printf("try_converter(): input=%s input_len = %llu\n", input, input_len);
    #endif

    if (input_len == 0) {
        return CONVERT_ERROR_INPUT_SIZE;
    }

    do
    {
        temp = input[input_len];

        /* Temporarily null terminate token in buffer before calling converter
           function. This is faster than copying the token into a new null
           terminated string. */
        input[input_len] = '\0';
        result = (*field_info->converter)(input, input_len, 0, output,
            field_info->output_field_size, field_info->converter_arg);
        input[input_len] = temp;

        if (infer_type
            && result != CONVERT_SUCCESS
            && result != CONVERT_ERROR_USER_CONVERTER)
        {
            if (field_info->converter ==
                default_converters[UINT_CONVERTER_FUNC])
            {
                field_info->converter =
                    default_converters[INT_CONVERTER_FUNC];
            }
            else if (field_info->converter ==
                default_converters[INT_CONVERTER_FUNC])
            {
                field_info->converter =
                    default_converters[FLOAT_CONVERTER_FUNC];
            }
            else if (field_info->converter ==
                default_converters[FLOAT_CONVERTER_FUNC])
            {
                field_info->converter =
                    default_converters[STRING_OBJECT_CONVERTER_FUNC];
            }
            else
            {
                /* We're out of converter functions to try */
                break;
            }
        }
    }
    while (infer_type &&
           result != CONVERT_SUCCESS &&
           result != CONVERT_ERROR_USER_CONVERTER);

    return result;
}


ConvertError try_fill_values(char *input, uint64_t input_len, char *output,
    FieldInfo *field_info)
{
    ConvertError result = CONVERT_ERROR;

    MissingValues *missing_values = &field_info->missing_values;
    FillValue *fill_value = &field_info->fill_value;

    if (missing_values == NULL || fill_value == NULL)
        return result;

    if (fill_value->fill_value == NULL)
        return result;
    
    if (input_len == 0)
    {
        if (output != NULL)
        {
            size_t field_size = (size_t)field_info->output_field_size;
            memcpy(output, fill_value->fill_value, field_size);
        }
        result = CONVERT_SUCCESS;
    }
    else
    {
        uint64_t i;
        for (i = 0; i < missing_values->num_missing_values; i++)
        {
            if (missing_values->missing_values[i] == NULL)
                continue;

            if (strncmp(missing_values->missing_values[i],
                        input, (size_t)input_len) == 0)
            {
                if (output != NULL)
                {
                    size_t field_size = (size_t)field_info->output_field_size;
                    memcpy(output, fill_value->fill_value, field_size);
                }
                result = CONVERT_SUCCESS;

                break;
            }
        }

        /* If a matching missing value was not found and loose flag is set,
           use fill value. */
        if (fill_value->loose && output != NULL)
        {
            size_t field_size = (size_t)field_info->output_field_size;
            memcpy(output, fill_value->fill_value, field_size);
            result = CONVERT_SUCCESS;
        }

    }

    return result;
}


AdapterError process_token(char *input, uint64_t input_len, char **output,
    FieldInfo *field_info,
    int infer_types_mode)
{
    AdapterError result = ADAPTER_ERROR_PROCESS_TOKEN;
    ConvertError convert_result;

    assert(field_info != NULL);

    #ifdef DEBUG_ADAPTER
    {
        char *temp = calloc(input_len + 1, sizeof(char));
        memcpy(temp, input, input_len);
        printf("process_token(): token=%s\n", temp);
        free(temp);
    }
    #endif

    convert_result = try_converter(input, input_len, *output,
            field_info, 0);

    if (convert_result != CONVERT_SUCCESS)
    {
        convert_result = try_fill_values(input, input_len, *output, field_info);
    }

    if (convert_result != CONVERT_SUCCESS
        && infer_types_mode
        && field_info->infer_type)
    {
        convert_result = try_converter(input, input_len, *output,
                field_info, 1);
    }

    if (convert_result != CONVERT_SUCCESS)
    {
        /* Set convert error info */
        convert_error_info.convert_result = convert_result;
        convert_error_info.token = calloc((size_t)input_len + 1, sizeof(char));
        memcpy(convert_error_info.token, input, (size_t)input_len);

        result = ADAPTER_ERROR_CONVERT;
    }
    else 
    {                   
        result = ADAPTER_SUCCESS;
    }

    if (*output != NULL)
        *output += field_info->output_field_size;

    return result;
}


#define PROCESS() \
do { \
    /* End of record has been hit, so process all the tokens now */ \
    result = ADAPTER_SUCCESS; \
    if (adapter->fields->field_info[field].converter != NULL \
        && record % step == 0 \
        && (*output != NULL \
            || (adapter->fields->field_info[field].infer_type == 1 \
                && adapter->infer_types_mode))) \
        result = process_token(adapter->buffer.data + token_start, \
            offset - token_start - skip_escapes, \
            output, \
            &adapter->fields->field_info[field], \
            adapter->infer_types_mode); \
\
    /* next token starts after delimiter/newline */ \
    token_start = offset + 1; \
\
    if (result != ADAPTER_SUCCESS) \
    { \
        convert_error_info.record_num = \
            adapter->input_data->start_record + record; \
        convert_error_info.field_num = field; \
        return result; \
    } \
\
    (*num_tokens_found)++; \
    skip_escapes = 0; \
\
    field++; \
    if (field >= adapter->fields->num_fields) \
    { \
        if (enable_index && \
            (adapter->input_data->start_record + record) % index_density == 0) \
        { \
            adapter->indexer(adapter->index, \
                adapter->input_data->start_record + record, \
                adapter->input_data->start_offset + record_offset); \
        } \
\
        field = 0; \
        record++; \
        if (c != '\n' && c != '\r') { \
            state = RECORD_END_STATE; \
        } \
        else { \
            state = DEFAULT_STATE; \
            record_offset = offset + 1; \
        } \
        /* bytes processed = end of token + delimiter/newline */ \
        adapter->buffer.bytes_processed = offset + 1; \
\
    } \
} while (0) \


int is_whitespace(char c)
{
    if (c == ' ' || c == '\t')
    {
        return 1;
    }

    return 0;
}


AdapterError delim_tokenizer(TextAdapter *adapter,
                             uint64_t num_tokens, uint64_t step,
                             char **output, uint64_t *num_tokens_found,
                             int enable_index, uint64_t index_density)
{
    TokenizerState state = DEFAULT_STATE;
    TokenizerState saved_state;

    char c = '\0';
    char prev_c = '\0';

    AdapterError result = ADAPTER_SUCCESS;
 
    uint64_t record;
    uint64_t field; 
    uint64_t offset;
    uint64_t record_offset = 0;
    uint64_t token_start = 0;
    uint32_t skip_escapes = 0;

    #ifdef DEBUG_ADAPTER
    printf("delim_tokenizer(): num_tokens=%llu" \
           " buffer.bytes_processed=%llu"       \
           " buffer.size=%llu"                  \
           " eof=%d\n",
           num_tokens,
           adapter->buffer.bytes_processed,
           adapter->buffer.size,
           adapter->buffer.eof);
    #endif

    assert(adapter != NULL);

    record = *num_tokens_found / adapter->fields->num_fields;
    field = *num_tokens_found % adapter->fields->num_fields;
    offset = adapter->buffer.bytes_processed;

    record_offset = offset;

    while (*num_tokens_found < num_tokens ||
           (*num_tokens_found == num_tokens && state == RECORD_END_STATE))
    {
        /* Check to see if we need to refresh the buffer */
        if (offset >= adapter->buffer.size)
        {
            if (adapter->buffer.eof)
            {
                result = ADAPTER_ERROR_READ_EOF;
                break;
            }

            offset = offset - adapter->buffer.bytes_processed;
            record_offset = record_offset - adapter->buffer.bytes_processed;
            token_start = token_start - adapter->buffer.bytes_processed;
            result = refresh_buffer(&adapter->buffer, adapter->input_data);
            if (result != ADAPTER_SUCCESS)
            {
                break;
            }
        }

        prev_c = c;
        c = adapter->buffer.data[offset];
        if (skip_escapes > 0)
        {
            adapter->buffer.data[offset - skip_escapes] =
                adapter->buffer.data[offset];
        }

        #ifdef DEBUG_ADAPTER
        printf("delim_tokenizer(): char=%c\n", c);
        printf("delim_tokenizer(): state=%d\n", state);
        printf("delim_tokenizer(): offset=%llu\n", offset);
        #endif

        switch (state)
        {
        case DEFAULT_STATE:
            if (c == adapter->escape_char)
            {
                skip_escapes++;
                saved_state = state;
                state = ESCAPE_STATE;
                token_start = offset;
            }
            else if (c == adapter->delim_char
                     || (adapter->any_whitespace_as_delim && is_whitespace(c)))
            {
                /* don't treat leading spaces at start of record as delimiter*/
                if (!is_whitespace(c) || adapter->group_whitespace_delims == 0)
                {
                    state = RECORD_STATE;
                    token_start = record_offset;
                    PROCESS();
                }
            }
            else if ((c == '\n' || c == '\r'))
            {
                if (adapter->skipblanklines == 0) {
                    /* Fill in missing fields with fill values if line is blank. */
                    state = RECORD_STATE;
                    while (field < adapter->fields->num_fields
                           && state == RECORD_STATE)
                    {
                        token_start = offset;
                        PROCESS();
                    }
                }
                record_offset = offset + 1; 
            }
            else if (c == adapter->comment_char)
            {
                state = COMMENT_STATE;
            }
            else if (c == adapter->quote_char)
            {
                state = QUOTE_STATE;
                /* actual token begins one character after quote */
                token_start = offset+1; 
            }
            else if (!isspace(c))
            {
                state = RECORD_STATE;
                token_start = record_offset;
            }
            break;
        case RECORD_STATE:
            if (c == adapter->escape_char)
            {
                skip_escapes++;
                saved_state = state;
                state = ESCAPE_STATE;
            }
            else if (c == adapter->delim_char
                     || (adapter->any_whitespace_as_delim && is_whitespace(c)))
            {
                /* only treat first consecutive space as delimiter;
                   ignore the rest */
                if (!is_whitespace(c)
                    || adapter->group_whitespace_delims == 0
                    || !is_whitespace(prev_c))
                {
                    PROCESS();
                }
            }
            else if ((c == '\n' || c == '\r'))
            {
                PROCESS();
                /* Fill in missing fields with fill values */
                while (field < adapter->fields->num_fields
                       && state == RECORD_STATE) {
                    token_start = offset;
                    PROCESS();
                }
                record_offset = offset + 1;
            }
            else if ((c == adapter->comment_char))
            {
                state = COMMENT_STATE;
                PROCESS();
            }
            else if (c == adapter->quote_char)
            {
                state = QUOTE_STATE;
                token_start = offset+1;
            }
            break;
        case RECORD_END_STATE:
            if (c == '\n' || c == '\r')
            {
                state = DEFAULT_STATE;
                record_offset = offset + 1;
            }
            break;
        case QUOTE_END_STATE:
            if (c == adapter->delim_char)
            {
                state = RECORD_STATE;
                token_start = offset + 1;
            }
            else if ((c == '\n' || c == '\r'))
            {
                state = DEFAULT_STATE;
                record_offset = offset + 1;
            }
            else if (c == adapter->comment_char)
            {
                state = COMMENT_STATE;
            }
            break;
        case ESCAPE_STATE:
            state = saved_state;
            if (state == DEFAULT_STATE) {
                state = RECORD_STATE;
            }
            break;
        default:
            if (state == COMMENT_STATE && (c == '\n' || c == '\r'))
            {
                state = DEFAULT_STATE;
                record_offset = offset + 1;
            }
            else if (state == QUOTE_STATE && c == adapter->quote_char)
            {
                state = QUOTE_END_STATE;
                /* actual token ends one character before quote */
                PROCESS();
            }
            else if (c == adapter->escape_char)
            {
                skip_escapes++;
                saved_state = state;
                state = ESCAPE_STATE;
            }
        }

        offset++;
    }

    #ifdef DEBUG_ADAPTER
    printf("delim_tokenizer() %d %d %d %d %d %d %d\n",
        state,
        token_start,
        offset,
        adapter->buffer.size,
        adapter->buffer.eof,
        field,
        adapter->fields->num_fields);
    #endif

    /* If we're last field of the last record of the file,
     * let's go ahead and try to process token */
    if ((state == RECORD_STATE || state == ESCAPE_STATE) &&
        adapter->buffer.size > 0 &&
        offset == adapter->buffer.size &&
        adapter->buffer.eof == 1 &&
        field < adapter->fields->num_fields)
    {
        PROCESS();

        /* Fill in last missing fields with fill values */
        while (field < adapter->fields->num_fields &&
               (state != RECORD_END_STATE && state != DEFAULT_STATE)) {
            token_start = offset;
            PROCESS();
        }
        adapter->buffer.bytes_processed = offset;
    }

    return result;
}


#define PROCESS_JSON() \
do { \
    if (token_start >= 0) \
    { \
        if (jc->state == ST \
                && adapter->infer_types_mode \
                && adapter->fields->field_info[field].infer_type \
                && adapter->fields->field_info[field].converter != default_converters[STRING_OBJECT_CONVERTER_FUNC]) { \
            adapter->fields->field_info[field].converter = default_converters[STRING_OBJECT_CONVERTER_FUNC]; \
        } \
        result = ADAPTER_SUCCESS; \
        if (adapter->fields->field_info[field].converter != NULL \
            && record % step == 0 \
            && (*output != NULL \
                || (adapter->fields->field_info[field].infer_type == 1 \
                    && adapter->infer_types_mode))) {\
            result = process_token(adapter->buffer.data + token_start, \
                offset - token_start, \
                output, \
                &adapter->fields->field_info[field], \
                adapter->infer_types_mode); \
        } \
\
        adapter->buffer.bytes_processed = offset + 1; \
\
        if (result != ADAPTER_SUCCESS) \
        { \
            convert_error_info.record_num = \
                adapter->input_data->start_record + record; \
            convert_error_info.field_num = field; \
            return result; \
        } \
\
        (*num_tokens_found)++; \
        token_start = -1; \
        field++; \
        if (field >= adapter->fields->num_fields) \
        { \
            field = 0; \
            record++; \
        } \
    } \
} while (0)

AdapterError json_tokenizer(TextAdapter *adapter,
                            uint64_t num_tokens, uint64_t step,
                            char **output, uint64_t *num_tokens_found,
                            int enable_index, uint64_t index_density)
{
    AdapterError result = ADAPTER_SUCCESS;
    JsonTokenizerArgs *json_args = NULL;
    struct JSON_checker_struct *jc = NULL;
    uint64_t offset = 0;
    char c = '\0';
    int64_t token_start = -1;
    uint64_t record = 0;
    uint64_t field = 0;
    int next_class, next_state;
    
    offset = adapter->buffer.bytes_processed;
    record = *num_tokens_found / adapter->fields->num_fields;
    field = *num_tokens_found % adapter->fields->num_fields;

    json_args = (JsonTokenizerArgs *)adapter->tokenize_args;
    jc = json_args->jc;

    while (*num_tokens_found < num_tokens)
    {
        /* Check to see if we need to refresh the buffer */
        if (offset >= adapter->buffer.size)
        {
            if (adapter->buffer.eof)
            {
                result = ADAPTER_ERROR_READ_EOF;
                break;
            }

            offset = offset - adapter->buffer.bytes_processed;
            token_start = token_start - adapter->buffer.bytes_processed;
            result = refresh_buffer(&adapter->buffer, adapter->input_data);
            if (result != ADAPTER_SUCCESS)
            {
                break;
            }
        }

        c = adapter->buffer.data[offset];
        
        /* character should be ascii */
        if ((unsigned int)c > 127)
        {
            result = ADAPTER_ERROR_INVALID_CHAR_CODE;
            convert_error_info.token = calloc((size_t)2, sizeof(char));
            convert_error_info.token[0] = c;
            break;
        }
        
        next_class = ascii_class[(unsigned int)c];
        if (next_class <= __)
        {
            result = ADAPTER_ERROR_JSON;
            break;
        }

        next_state = state_transition_table[jc->state][next_class];
        if (next_state >= 0)
        {
            jc->state = next_state;
        }
        else
        {
            switch (next_state)
            {
            case -16:
                token_start = offset;
                jc->state = ZE;
                break;
            case -14:
                token_start = offset;
                if (next_class == C_LOW_F)
                {
                    jc->state = F1;
                }
                else if (next_class == C_LOW_T)
                {
                    jc->state = T1;
                }
                else if (next_class == C_LOW_N)
                {
                    jc->state = N1;
                }
                break;
            case -15:
                offset++;
                PROCESS_JSON();
                offset--;
                jc->state = OK;
                break;
            case -10:
                token_start = offset;
                jc->state = IN;
                break;
            case -11:
                /* token starts after the quote character */
                token_start = offset + 1;
                jc->state = ST;
                break;
            case -9:
                if (!pop(jc, MODE_KEY)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = OK;
                break;

    /* } */ case -8:

                PROCESS_JSON();

                if (!pop(jc, MODE_OBJECT)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = OK;
                break;

    /* ] */ case -7:
                if (!pop(jc, MODE_ARRAY)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = OK;
                break;

    /* { */ case -6:
                if (!push(jc, MODE_KEY)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = OB;
                break;

    /* [ */ case -5:
                if (!push(jc, MODE_ARRAY)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = AR;
                break;

    /* " */ case -4:

                PROCESS_JSON();

                switch (jc->stack[jc->top]) {
                case MODE_KEY:
                    jc->state = CO;
                    break;
                case MODE_ARRAY:
                case MODE_OBJECT:
                    jc->state = OK;
                    break;
                default:
                    return ADAPTER_ERROR_JSON;
                }
                break;

    /* , */ case -3:

                PROCESS_JSON();

                switch (jc->stack[jc->top]) {
                case MODE_OBJECT:
    /*
        A comma causes a flip from object mode to key mode.
    */
                    if (!pop(jc, MODE_OBJECT) || !push(jc, MODE_KEY)) {
                        return ADAPTER_ERROR_JSON;
                    }
                    jc->state = KE;
                    break;
                case MODE_ARRAY:
                    jc->state = VA;
                    break;
                default:
                    return ADAPTER_ERROR_JSON;
                }
                break;

    /* : */ case -2:
                /* A colon causes a flip from key mode to object mode. */
                if (!pop(jc, MODE_KEY) || !push(jc, MODE_OBJECT)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = VA;
                break;
            default:
                return ADAPTER_ERROR_JSON;
            }
        }

        offset++;
    }

    if (adapter->buffer.eof &&
            offset >= adapter->buffer.size) {

        if (!JSON_checker_done(jc)) {
            result = ADAPTER_ERROR_JSON;
        }
        json_args->jc = NULL;
    }

    return result;
}


AdapterError json_record_tokenizer(TextAdapter *adapter,
                            uint64_t num_tokens, uint64_t step,
                            char **output, uint64_t *num_tokens_found,
                            int enable_index, uint64_t index_density)
{
    AdapterError result = ADAPTER_SUCCESS;
    JsonTokenizerArgs *json_args = NULL;
    struct JSON_checker_struct *jc = NULL;
    uint64_t offset = 0;
    char c = '\0';
    int64_t token_start = -1;
    uint64_t record = 0;
    uint64_t field = 0;
    int next_class, next_state;
    
    offset = adapter->buffer.bytes_processed;
    record = *num_tokens_found / adapter->fields->num_fields;
    field = *num_tokens_found % adapter->fields->num_fields;

    json_args = (JsonTokenizerArgs *)adapter->tokenize_args;
    jc = json_args->jc;

    while (*num_tokens_found < num_tokens)
    {
        /* Check to see if we need to refresh the buffer */
        if (offset >= adapter->buffer.size)
        {
            if (adapter->buffer.eof)
            {
                result = ADAPTER_ERROR_READ_EOF;
                break;
            }

            result = refresh_buffer(&adapter->buffer, adapter->input_data);
            if (result != ADAPTER_SUCCESS)
            {
                break;
            }
            offset = 0;
        }

        c = adapter->buffer.data[offset];
        
        /* character should be ascii */
        if ((unsigned int)c > 127)
        {
            result = ADAPTER_ERROR_INVALID_CHAR_CODE;
            convert_error_info.token = calloc((size_t)2, sizeof(char));
            convert_error_info.token[0] = c;
            break;
        }
        
        next_class = ascii_class[(unsigned int)c];
        if (next_class <= __)
        {
            result = ADAPTER_ERROR_JSON;
            break;
        }

        next_state = state_transition_table[jc->state][next_class];
        if (next_state >= 0)
        {
            jc->state = next_state;
        }
        else
        {
            switch (next_state)
            {
            case -16:
                jc->state = ZE;
                break;
            case -14:
                if (next_class == C_LOW_F)
                {
                    jc->state = F1;
                }
                else if (next_class == C_LOW_T)
                {
                    jc->state = T1;
                }
                else if (next_class == C_LOW_N)
                {
                    jc->state = N1;
                }
                break;
            case -15:
                jc->state = OK;
                break;
            case -10:
                jc->state = IN;
                break;
            case -11:
                jc->state = ST;
                break;
            case -9:
                if (!pop(jc, MODE_KEY)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = OK;
                break;

    /* } */ case -8:
                if ((jc->top == 1 && jc->stack[1] == MODE_OBJECT) ||
                    (jc->top == 2 && jc->stack[2] == MODE_OBJECT &&
                        jc->stack[1] == MODE_ARRAY)) {
                    offset++;
                    PROCESS_JSON();
                    offset--;
                }
                if (!pop(jc, MODE_OBJECT)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = OK;
                break;

    /* ] */ case -7:
                if (!pop(jc, MODE_ARRAY)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = OK;
                break;

    /* { */ case -6:
                if (jc->top == 0 || (jc->top == 1 && jc->stack[1] == MODE_ARRAY)) {
                    token_start = offset;
                }
                if (!push(jc, MODE_KEY)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = OB;
                break;

    /* [ */ case -5:
                if (!push(jc, MODE_ARRAY)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = AR;
                break;

    /* " */ case -4:
                switch (jc->stack[jc->top]) {
                case MODE_KEY:
                    jc->state = CO;
                    break;
                case MODE_ARRAY:
                case MODE_OBJECT:
                    jc->state = OK;
                    break;
                default:
                    return ADAPTER_ERROR_JSON;
                }
                break;

    /* , */ case -3:
                switch (jc->stack[jc->top]) {
                case MODE_OBJECT:
    /*
        A comma causes a flip from object mode to key mode.
    */
                    if (!pop(jc, MODE_OBJECT) || !push(jc, MODE_KEY)) {
                        return ADAPTER_ERROR_JSON;
                    }
                    jc->state = KE;
                    break;
                case MODE_ARRAY:
                    jc->state = VA;
                    break;
                default:
                    return ADAPTER_ERROR_JSON;
                }
                break;

    /* : */ case -2:
                /* A colon causes a flip from key mode to object mode. */
                if (!pop(jc, MODE_KEY) || !push(jc, MODE_OBJECT)) {
                    return ADAPTER_ERROR_JSON;
                }
                jc->state = VA;
                break;
            default:
                return ADAPTER_ERROR_JSON;
            }
        }

        offset++;
    }

    if (adapter->buffer.eof &&
            offset >= adapter->buffer.size) {

        if (!JSON_checker_done(jc)) {
            result = ADAPTER_ERROR_JSON;
        }
        json_args->jc = NULL;
    }

    return result;
}


AdapterError regex_tokenizer(TextAdapter *adapter,
    uint64_t num_tokens, uint64_t step, char **output,
    uint64_t *num_tokens_found, int enable_index, uint64_t index_density)
{
    AdapterError result = ADAPTER_SUCCESS;

    uint64_t record;
    uint64_t field;
 
    uint64_t i;
    int *outvector;
    int outvector_size;
    int match;
    int pcre_result;

    RegexTokenizerArgs *regex_args;

    assert(adapter != NULL);

    record = *num_tokens_found / adapter->fields->num_fields;
    field = *num_tokens_found % adapter->fields->num_fields;

    i = adapter->buffer.bytes_processed;

    /*
     * The outvector array will hold the result of pcre_exec().
     * The array must be three times the number of fields plus one.
     * The first two elements hold the byte offset of the first character
     * in the matched string, and the length of the matched string. Each 
     * pair of elements after that hold the byte offset and length of each
     * group in the matched string. That takes care of 2/3 of the array;
     * the last third of the outvector array is used internally by pcre_exec()
     * and doesn't concern us.
     */
    outvector = calloc(1, sizeof(int)*(adapter->fields->num_fields+1)*3);
    outvector_size = (adapter->fields->num_fields + 1) * 3;

    regex_args = (RegexTokenizerArgs *)adapter->tokenize_args;

    while (*num_tokens_found < num_tokens)
    {
        /* Check to see if we need to refresh the buffer */
        if (i >= adapter->buffer.size)
        {
            if (adapter->buffer.eof)
            {
                result = ADAPTER_ERROR_READ_EOF;
                break;
            }

            i = i - adapter->buffer.bytes_processed;
            result = refresh_buffer(&adapter->buffer, adapter->input_data);
            if (result != ADAPTER_SUCCESS)
            {
                break;
            }
        }

        pcre_result = pcre_exec(regex_args->pcre_regex,
            regex_args->extra_regex,
            adapter->buffer.data+i,
            adapter->buffer.size-i,
            0,
            0,
            outvector,
            outvector_size);

        if (pcre_result < 1)
        {
            #ifdef DEBUG_ADAPTER
            printf("regex_tokenizer(): no match found at offset %llu pcre_result=%d\n",
                i, pcre_result);
            #endif

            /* Advance to next line */
            while (i < adapter->buffer.size && adapter->buffer.data[i] != '\n')
            {
                i++;
            }
            i++;
            continue;
        }

        #ifdef DEBUG_ADAPTER
        {
            char *temp = calloc(outvector[1]-outvector[0], sizeof(char));
            memcpy(temp, adapter->buffer.data+i+outvector[0], outvector[1]-outvector[0]);
            printf("regex_tokenizer(): match found at offset %llu for string %s\n", i, temp);
            free(temp);
        }
        #endif

        /* outvector[1] stores length of entire matched record,
         * so outvector[1] + i (current pos in buffer) has to be less than
         * buffer size. */
        if (i + outvector[1] > adapter->buffer.size)
            break;

        for (match = 0; match < adapter->fields->num_fields; match++)
        {
            /* First group starts at element 2 */
            int index_start = (match+1)*2;
            int index_end = index_start + 1;

            char *token_ptr = adapter->buffer.data + i + outvector[index_start];
            result = process_token(token_ptr,
                outvector[index_end] - outvector[index_start],
                output,
                &adapter->fields->field_info[match],
                adapter->infer_types_mode);

            if (result != ADAPTER_SUCCESS)
            {
                convert_error_info.record_num =
                    adapter->input_data->start_record + record;
                convert_error_info.field_num = field;
                return result;
            }

            (*num_tokens_found)++;
        }

        record++;
        
        /* Add offset from i of start of matched record to length of
           matched record, and advance i by that amount */
        i += outvector[1]+outvector[0];
        adapter->buffer.bytes_processed = i;
    }

    return result;
}


AdapterError fixed_width_tokenizer(TextAdapter *adapter, uint64_t num_tokens,
    uint64_t step, char **output, uint64_t *num_tokens_found, int enable_index,
    uint64_t index_density)
{
    AdapterError result = ADAPTER_SUCCESS;

    uint64_t record;
    uint64_t field;
    uint64_t offset;
    uint64_t end_line_offset = 0;
    FieldInfo *field_info;

    assert(adapter != NULL);

    record = *num_tokens_found / adapter->fields->num_fields;
    field = *num_tokens_found % adapter->fields->num_fields;

    offset = adapter->buffer.bytes_processed;

    field_info = adapter->fields->field_info;
    while (*num_tokens_found < num_tokens)
    {
        /* Check to see if we need to refresh the buffer */
        if (offset + field_info[field].input_field_width > adapter->buffer.size)
        {
            if (adapter->buffer.eof)
            {
                result = ADAPTER_ERROR_READ_EOF;
                break;
            }

            offset = offset - adapter->buffer.bytes_processed;
            result = refresh_buffer(&adapter->buffer, adapter->input_data);
            if (result != ADAPTER_SUCCESS)
            {
                break;
            }
        }

         /* We don't want to process last field unless we can process to end
            of line. We always want to enter this function at the start of a
            record or field, not between last record and end of line. */
        if (field + 1 == adapter->fields->num_fields)
        {
            end_line_offset = offset + field_info[field].input_field_width;
            /* eat rest of line */
            while (end_line_offset < adapter->buffer.size &&
                   adapter->buffer.data[end_line_offset] != '\n' &&
                   adapter->buffer.data[end_line_offset] != '\r')
                end_line_offset++;

            /* eat newlines */
            while (end_line_offset < adapter->buffer.size &&
                   (adapter->buffer.data[end_line_offset] == '\n' ||
                    adapter->buffer.data[end_line_offset] == 'r'))
                end_line_offset++;

            /* Ran out of buffer before reaching the very end of line.
               If this is the last chunk of data, go ahead and process. */
            if (end_line_offset == adapter->buffer.size && !adapter->buffer.eof)
                return result;
        }

        /* Skip comment lines */
        if (adapter->buffer.data[offset] == adapter->comment_char)
        {
            while (offset < adapter->buffer.size
                   && adapter->buffer.data[offset] != '\n')
            {
                offset++;
            }
        }

        /* Skip blank lines */
        while (offset < adapter->buffer.size
               && adapter->buffer.data[offset] == '\n')
        {
            offset++;
        }

        if (offset == adapter->buffer.size)
        {
            return result;
        }

        if (field_info[field].converter && (record % step) == 0 && *output != 0)
        {
            uint64_t input_len = field_info[field].input_field_width;
            char *input = adapter->buffer.data + offset;

            #ifdef DEBUG_ADAPTER
            {
                char *temp = calloc(input_len + 1, sizeof(char));
                memcpy(temp, input, input_len);
                printf("fixed_width_tokenizer(): field=%llu token=%s\n", field, temp);
                free(temp);
            }
            #endif

            result = process_token(input, input_len, output,
                &adapter->fields->field_info[field],
                adapter->infer_types_mode);

            if (result != ADAPTER_SUCCESS)
            {
                convert_error_info.record_num =
                    adapter->input_data->start_record + record;
                convert_error_info.field_num = field;
                return result;
            }
        }

        (*num_tokens_found)++;
        offset += field_info[field].input_field_width;
        field++;

        /* end of record */
        if (field == adapter->fields->num_fields)
        {
            field = 0;
            record++;
            offset = end_line_offset;
        }

        adapter->buffer.bytes_processed = offset;
    }
    
    return result;
}


AdapterError record_tokenizer(TextAdapter *adapter, uint64_t num_tokens,
    uint64_t step, char **output, uint64_t *num_tokens_found, int enable_index,
    uint64_t index_density)
{
    AdapterError result = ADAPTER_SUCCESS;

    uint64_t offset = adapter->buffer.bytes_processed;
    TokenizerState state = DEFAULT_STATE;
    uint64_t record_offset = offset;
    uint64_t token_start = offset;
    uint64_t record = *num_tokens_found / adapter->fields->num_fields;
    uint64_t field = 0;
    char c;
    uint32_t skip_escapes = 0;

    while (*num_tokens_found < num_tokens)
    {
        /* Check to see if we need to refresh the buffer */
        if (offset >= adapter->buffer.size)
        {
            if (adapter->buffer.eof)
            {
                result = ADAPTER_ERROR_READ_EOF;
                break;
            }

            offset = offset - adapter->buffer.bytes_processed;
            token_start = token_start - adapter->buffer.bytes_processed;
            result = refresh_buffer(&adapter->buffer, adapter->input_data);
            if (result != ADAPTER_SUCCESS)
            {
                break;
            }
        }

        c = adapter->buffer.data[offset];

        #ifdef DEBUG_ADAPTER
        printf("record_tokenizer(): char=%c\n", c);
        printf("record_tokenizer(): state=%d\n", state);
        #endif

        switch (state)
        {
            case DEFAULT_STATE:
                if (c == '#')
                {
                    state = COMMENT_STATE;
                }
                else if (c == '\n' || c == '\r')
                {
                    record_offset = offset + 1;
                    token_start = offset + 1;
                }
                else if (c == '"')
                {
                    state = QUOTE_STATE;
                    record_offset = offset;
                    token_start = offset;
                }
                else if (!isspace(c))
                {
                    state = RECORD_STATE;
                    token_start = record_offset;
                }
                break;
            case RECORD_STATE:
                if (c == '\n' || c == '\r')
                {
                    state = DEFAULT_STATE;
                    PROCESS();
                    record_offset = offset + 1;
                }
                else if (c == '"')
                {
                    state = QUOTE_STATE;
                }
                break;
            case COMMENT_STATE:
                if (c == '\n' || c == '\r')
                {
                    state = DEFAULT_STATE;
                    record_offset = offset + 1;
                    token_start = offset + 1;
                }
                break;
            case QUOTE_STATE:
                if (c == '"')
                {
                    state = RECORD_STATE;
                }
                break;
            default:
                break;
        }

        offset++;
    }

    /* If we're on the last character of the last field of the last record
       of the file, let's go ahead and try to process token */
    if (state == RECORD_STATE &&
        adapter->buffer.size > 0 &&
        offset == adapter->buffer.size &&
        adapter->buffer.eof == 1 &&
        (field+1) == adapter->fields->num_fields)
    {
        PROCESS();
        adapter->buffer.bytes_processed = offset;
    }

    return result;
}


AdapterError line_tokenizer(TextAdapter *adapter, uint64_t num_tokens,
    uint64_t step, char **output, uint64_t *num_tokens_found, int enable_index,
    uint64_t index_density)
{
    AdapterError result = ADAPTER_SUCCESS;

    uint64_t offset = adapter->buffer.bytes_processed;
    TokenizerState state = DEFAULT_STATE;
    uint64_t token_start = offset;
    uint64_t record_offset = 0;
    uint64_t record = *num_tokens_found / adapter->fields->num_fields;
    uint64_t field = 0;
    char c = '\0';
    uint32_t skip_escapes = 0;

    // We should never need to index by line
    enable_index = 0;

    while (*num_tokens_found < num_tokens)
    {
        /* Check to see if we need to refresh the buffer */
        if (offset >= adapter->buffer.size)
        {
            if (adapter->buffer.eof)
            {
                result = ADAPTER_ERROR_READ_EOF;
                break;
            }

            offset = offset - adapter->buffer.bytes_processed;
            token_start = token_start - adapter->buffer.bytes_processed;
            result = refresh_buffer(&adapter->buffer, adapter->input_data);
            if (result != ADAPTER_SUCCESS)
            {
                break;
            }
        }

        c = adapter->buffer.data[offset];

        if (c == '\n' || c == '\r')
        {
            if (offset > token_start) {
                PROCESS();
            }
            token_start = offset + 1;
        }

        offset++;
    }

    /* If we're on the last character of the last field of the last record
       of the file, let's go ahead and try to process token */
    if (c != '\n' && c != '\r' &&
        adapter->buffer.size > 0 &&
        offset == adapter->buffer.size &&
        adapter->buffer.eof == 1 &&
        offset > adapter->buffer.bytes_processed)
    {
        PROCESS();
        adapter->buffer.bytes_processed = offset;
    }

    return result;
}



