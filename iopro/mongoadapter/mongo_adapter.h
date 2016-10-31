#ifndef MONGOADAPTER_H
#define MONGOADAPTER_H

#define MONGO_HAVE_STDINT
#include <_stdint.h>
#include <mongo.h>
#include <converter_functions.h>
#include <field_info.h>


ConvertError mongo2int_converter(void *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg);
ConvertError mongo2uint_converter(void *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg);
ConvertError mongo2float_converter(void *input, uint32_t input_len,
    int input_type, void *output, uint32_t output_len, void *arg);


typedef struct mongo_adapter_t
{
    mongo conn;
    mongo_cursor cursor;
    char *database;
    char *collection;

    FieldList *fields;

    converter_func_ptr default_converters[NUM_CONVERTER_FUNCS];

} MongoAdapter;


typedef enum
{
    MONGO_ADAPTER_SUCCESS,
    MONGO_ADAPTER_ERROR,
    MONGO_ADAPTER_ERROR_TYPE_CHANGED,
    MONGO_ADAPTER_ERROR_INVALID_START_REC
} MongoAdapterError;


MongoAdapter* open_mongo_adapter(char *host, int port, char *database_name,
    char *collection_name);
void close_mongo_adapter(MongoAdapter *adapter);

uint32_t get_num_records(MongoAdapter *adapter);

MongoAdapterError read_records(MongoAdapter *adapter, uint32_t start_record,
    uint32_t num_records, int32_t step, void *output, uint32_t *num_records_read);

ConvertError try_converter(void *input, uint32_t input_len, int type,
    void **output, FieldInfo *field_info, converter_func_ptr *converters);

#endif
