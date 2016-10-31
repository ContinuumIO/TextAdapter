#include "mongo_adapter.h"
#include <string.h>
#include <assert.h>


MongoAdapter* open_mongo_adapter(char *host, int port, char *database_name,
    char *collection_name)
{
    MongoAdapter *adapter = calloc(1, sizeof(MongoAdapter));

    if (mongo_client(&adapter->conn, host, port) != MONGO_OK)
    {
        free(adapter);
        printf("JNB: open_mongo_adapter() %d\n", adapter->conn.err);
        return NULL;
    }

    adapter->database = (char *)calloc(1, strlen(database_name));
    strncpy(adapter->database, database_name, strlen(database_name));

    adapter->collection = (char *)calloc(1, strlen(collection_name));
    strncpy(adapter->collection, collection_name, strlen(collection_name));

    adapter->fields = calloc(1, sizeof(FieldList));

    adapter->default_converters[UINT_CONVERTER_FUNC] = &mongo2uint_converter;
    adapter->default_converters[INT_CONVERTER_FUNC] = &mongo2int_converter;
    adapter->default_converters[FLOAT_CONVERTER_FUNC] = &mongo2float_converter;

    return adapter;
}


void close_mongo_adapter(MongoAdapter *adapter)
{
    if (adapter == NULL)
        return;

    clear_fields(adapter->fields);
    free(adapter->fields);

    free(adapter->database);
    free(adapter->collection);
    free(adapter);
}


uint32_t get_num_records(MongoAdapter *adapter)
{
    double num_recs = mongo_count(&adapter->conn, adapter->database,
        adapter->collection, NULL);
    if (num_recs < MONGO_OK)
    {
        return MONGO_ADAPTER_ERROR;
    }

    return (uint32_t)num_recs;
}


ConvertError try_converter(void *input, uint32_t input_len, int type,
    void **output, FieldInfo *field_info,
    converter_func_ptr *converters)
{
    ConvertError result = CONVERT_ERROR;
    int type_changed = 0;
    do
    {
        result = (*field_info->converter)(input, input_len, type, *output,
            field_info->output_field_size, field_info->converter_arg);

        if (result != CONVERT_SUCCESS) 
        {
            if (result == CONVERT_ERROR_TRUNCATE && !(field_info->infer_type))
            {
                break; // user has set_field_types for this field. we're done 
            }

            // try to convert to other types
            *output = NULL;
            type_changed = 1;

            if (field_info->converter == converters[UINT_CONVERTER_FUNC])
            {
                field_info->converter = converters[INT_CONVERTER_FUNC];
                field_info->output_field_size = 8;
            }
            else if (field_info->converter == converters[INT_CONVERTER_FUNC])
            {
                field_info->converter = converters[FLOAT_CONVERTER_FUNC];
                field_info->output_field_size = 8;
            }
            else if (field_info->converter == converters[FLOAT_CONVERTER_FUNC])
            {
                field_info->converter = converters[STRING_OBJECT_CONVERTER_FUNC];
                field_info->output_field_size = 8;
            }
            else
            {
                /* We're out of converter functions to try */
                break;
            }
        }

    }
    while (result != CONVERT_SUCCESS);

    if (result == CONVERT_SUCCESS && type_changed)
    {
        result = CONVERT_SUCCESS_TYPE_CHANGED;
    }

    return result;
}


MongoAdapterError read_records(MongoAdapter *adapter, uint32_t start_record,
    uint32_t num_records, int32_t step, void *output, uint32_t *num_records_read)
{
    ConvertError convert_result;
    int type_changed = 0;
    uint32_t total_records = get_num_records(adapter);
    uint32_t current_record = 0;
    char *name;
    mongo_cursor cursor;
    bson_iterator it;
    size_t num_fields;
    size_t i;
    bson_type type;
    int64_t int64_temp;
    double double_temp;
    const char *char_temp;

    *num_records_read = 0;

    if (start_record > total_records)
        return MONGO_ADAPTER_ERROR_INVALID_START_REC;
    name = (char *)calloc(1,
        strlen(adapter->database) + strlen(adapter->collection) + 1);
    strncpy(name, adapter->database, strlen(adapter->database));
    name[strlen(adapter->database)] = '.';
    strncpy(name + strlen(adapter->database) + 1,
        adapter->collection,
        strlen(adapter->collection));

    mongo_cursor_init(&cursor, &adapter->conn, name);
    mongo_cursor_set_skip(&cursor, start_record);
    mongo_cursor_set_limit(&cursor, num_records);

    convert_result = CONVERT_SUCCESS;
    while (mongo_cursor_next(&cursor) == MONGO_OK
           && convert_result == CONVERT_SUCCESS)
    {
        if (current_record % step != 0)
        {
            current_record++;
            continue;
        }

        current_record++;

        num_fields = adapter->fields->num_fields;
        for (i = 0; i < num_fields; i++)
        {
            FieldInfo *field_info = &adapter->fields->field_info[i];
            if (field_info->converter == NULL)
            {
                continue;
            }

            type = bson_find(&it, &cursor.current, field_info->name);
            if (type == BSON_INT || type == BSON_LONG || type == BSON_BOOL)
            {
                int64_temp = bson_iterator_int(&it);
                convert_result = try_converter((void*)&int64_temp,
                    sizeof(int64_temp),
                    type,
                    &output,
                    field_info,
                    adapter->default_converters);
            }
            else if (type == BSON_DOUBLE)
            {
                double_temp = bson_iterator_double(&it);
                convert_result = try_converter((void*)&double_temp,
                    sizeof(double_temp),
                    type,
                    &output,
                    field_info,
                    adapter->default_converters);
            }
            else
            {
                /* Convert all other types to string objects for now */
                char_temp = bson_iterator_string(&it);
                convert_result = try_converter((void*)char_temp,
                    (int32_t)strlen(char_temp),
                    BSON_STRING,
                    &output,
                    field_info,
                    adapter->default_converters);
            }

            if (convert_result == CONVERT_SUCCESS && type_changed == 0)
            {
                output = (char*)output + field_info->output_field_size;
            }
            else if (convert_result == CONVERT_ERROR_TRUNCATE && !(field_info->infer_type))
            {
                /* acceptable case where user has set_field_types causing truncation, eg int(6.4) */
                convert_result = CONVERT_SUCCESS;
                if (type_changed == 0)
                {
                    output = (char*)output + field_info->output_field_size;
                }
            }
            else if (convert_result == CONVERT_SUCCESS_TYPE_CHANGED)
            {
                type_changed = 1;
            }
            else if (convert_result != CONVERT_SUCCESS)
            {
                break;
            }
        }    
        
        (*num_records_read)++;
    }

    mongo_cursor_destroy(&cursor);
    free(name);

    if (type_changed)
    {
        return MONGO_ADAPTER_ERROR_TYPE_CHANGED;
    }

    if (convert_result != CONVERT_SUCCESS)
    {
        return MONGO_ADAPTER_ERROR;
    }
    return MONGO_ADAPTER_SUCCESS;
}


ConvertError mongo2int_converter(void *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg)
{
    ConvertError result = CONVERT_ERROR;
    int64_t int_value = 0;
    double float_value = 0.0;
    int64_t temp_int;

    if (input_type == BSON_INT)
    {
        result = get_int_value(input, input_len, &int_value);
        if (result == CONVERT_SUCCESS && output != NULL)
        {
            result = put_int_value(output, output_len, int_value);
        }
    }
    else if (input_type == BSON_DOUBLE)
    {
        result = get_float_value(input, input_len, &float_value);
        
        if (result == CONVERT_SUCCESS)
        {
            temp_int = (int64_t)float_value;

            if (output != NULL)
            {
                result = put_int_value(output, output_len, temp_int);    
            }
            // To handle case where field type has been set by user
            if (result == CONVERT_SUCCESS &&
               (float_value - (double)temp_int > 0.001 || 
                float_value - (double)temp_int < -0.001))
            {
                result = CONVERT_ERROR_TRUNCATE; 
            }
        }
    }    
    return result;
}


ConvertError mongo2uint_converter(void *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg)
{
    ConvertError result = CONVERT_ERROR;
    int64_t int_value = 0;
    double float_value = 0.0;
    uint64_t uint_value;
    
    if (input_type == BSON_INT)
    {
        result = get_int_value(input, input_len, &int_value);
        if (result == CONVERT_SUCCESS)
        {
            if (output != NULL)
            {
                result = put_uint_value(output, output_len, (uint64_t)int_value);
            }
            if (result == CONVERT_SUCCESS && int_value < 0)
            {
                // As long as there wasn't an overflow or output size error,
                // place uint in the output, but mark result as truncated.
                // Handles case where field type has been set by user
                result = CONVERT_ERROR_TRUNCATE;
            }

        }
    }
    else if (input_type == BSON_DOUBLE)
    {
        result = get_float_value(input, input_len, &float_value);
        if (result == CONVERT_SUCCESS)
        {
            uint_value = (uint64_t)float_value;
            if (output != NULL)
            {
                result = put_uint_value(output, output_len, uint_value);    
            }
            if (result == CONVERT_SUCCESS && 
                ((int64_t)float_value < 0 ||
                float_value - (double)uint_value > 0.001))   
            {
                // As long as there wasn't an overflow or output size error,
                // place uint in the output, but mark result as truncated.
                // Handles case where field type has been set by user
                result = CONVERT_ERROR_TRUNCATE;
            }
        }
    }
    return result;
}


ConvertError mongo2float_converter(void *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg)
{
    ConvertError result = CONVERT_ERROR;
    double float_value = 0.0;
    int64_t int_value = 0;

    if (input_type == BSON_DOUBLE)
    {
        result = get_float_value(input, input_len, &float_value);
        if (result == CONVERT_SUCCESS && output != NULL)
        {
            result = put_float_value(output, output_len, float_value);
        }
    }
    else if (input_type == BSON_INT)
    {
        result = get_int_value(input, input_len, &int_value);
        if (result == CONVERT_SUCCESS && output != NULL)
        {
            result = put_float_value(output, output_len, (double)int_value);
        }
    }

    return result;
}



