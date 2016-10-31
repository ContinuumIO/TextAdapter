#ifndef ACCUMULO_ADAPTER_H
#define ACCUMULO_ADAPTER_H

#include "AccumuloProxy.h"
#include "proxy_types.h"
#include <thrift/transport/TBufferTransports.h>
#include <string>
#include <vector>

using apache::thrift::transport::TFramedTransport;
using accumulo::AccumuloProxyClient;
using accumulo::ScanOptions;
using accumulo::ScanResult;
using accumulo::KeyValueAndPeek;
using accumulo::KeyValue;
using accumulo::Key;
using accumulo::Range;

typedef enum
{
    UINT_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    STR_FIELD
} FieldType;

typedef enum
{
    ADAPTER_SUCCESS,
    ADAPTER_SUCCESS_TRUNCATION,
    ADAPTER_ERROR_INVALID_SEEK,
    ADAPTER_ERROR_OUTPUT_TYPE,
    ADAPTER_ERROR_OUTPUT_TYPE_SIZE,
    ADAPTER_ERROR_EOF,
    ADAPTER_ERROR_INVALID_TABLE_NAME,
    ADAPTER_ERROR_INT_CONVERSION,
    ADAPTER_ERROR_FLOAT_CONVERSION,
    ADAPTER_ERROR_SOCKET,
    ADAPTER_ERROR_LOGIN,
    ADAPTER_ERROR_TABLE_NAME
} AdapterError;

typedef struct
{
    boost::shared_ptr<AccumuloProxyClient> client;
    boost::shared_ptr<TFramedTransport> transport;
    std::string login;
    std::string table;
    std::string scanner;
    FieldType field_type;
    int output_type_size;
    std::string start_key;
    bool start_key_inclusive;
    std::string stop_key;
    bool stop_key_inclusive;
    std::vector<std::string> missing_values;
    void *fill_value;
} accumulo_adapter_t;

// Allocates and initializes accumulo_adapter_t, and connects to accumulo
// instance
accumulo_adapter_t * open_accumulo_adapter(std::string server,
                                           int port,
                                           std::string username,
                                           std::string password,
                                           std::string table,
                                           AdapterError *error);

// Closes accumulo connection and cleans up accumulo_adapter_t
void close_accumulo_adapter(accumulo_adapter_t *adapter);

void add_missing_value(accumulo_adapter_t *adapter, std::string value);
void clear_missing_values(accumulo_adapter_t *adapter);
void set_fill_value(accumulo_adapter_t *adapter, void *value, int size);
void clear_fill_value(accumulo_adapter_t *adapter);

// Seek to specified record offset in accumulo table. Offset is from beginning
// of table.
AdapterError seek_record(accumulo_adapter_t *adapter, int record_offset);

// Read specified number of records in accumulo table
AdapterError read_records(accumulo_adapter_t *adapter, int num_records, int step,
    void *output, int *num_records_found);

#endif
