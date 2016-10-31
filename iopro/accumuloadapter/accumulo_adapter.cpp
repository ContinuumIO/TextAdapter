#include "accumulo_adapter.h"
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TCompactProtocol.h>
#include <string>
#include <map>
#include <utility>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <cmath>
#include <exception>

using apache::thrift::transport::TSocket;
using apache::thrift::transport::TFramedTransport;
using apache::thrift::protocol::TProtocol;
using apache::thrift::protocol::TCompactProtocol;
using apache::thrift::transport::TTransportException;
using apache::thrift::TException;
using accumulo::AccumuloSecurityException;
using accumulo::TableNotFoundException;
using std::string;
using std::map;
using std::make_pair;
using std::istringstream;
using std::vector;


#if _MSC_VER
namespace boost{
  void throw_exception(std::exception const &e){}
}
#endif

accumulo_adapter_t * open_accumulo_adapter(string server,
                                           int port,
                                           string username,
                                           string password,
                                           string table,
                                           AdapterError *error)
{
    boost::shared_ptr<TSocket> socket(new TSocket(server, port));
    boost::shared_ptr<TFramedTransport> transport(new TFramedTransport(socket));
    boost::shared_ptr<TProtocol> protocol(new TCompactProtocol(transport));
    
    // Open socket to Accumulo server
    try {
        transport->open();
    }
    catch (const TTransportException &) {
        *error = ADAPTER_ERROR_SOCKET;
        return NULL;
    }

    accumulo_adapter_t *adapter = new accumulo_adapter_t;
    adapter->transport = transport;

    // Create map of login properties
    map<string, string> login_properties;
    login_properties.insert(make_pair("password", password));
    
    // Create Accumulo proxy client. Variable 'login' stores reference to login
    // that is needed later for scanner. I have no idea why this isn't stored
    // in the client object.
    try {
        boost::shared_ptr<AccumuloProxyClient> client(new AccumuloProxyClient(protocol));
        string login;
        client->login(login, username, login_properties);
        adapter->client = client;
        adapter->login = login;
    }
    catch (const AccumuloSecurityException &) {
        *error = ADAPTER_ERROR_LOGIN;
        return NULL;
    }

    adapter->table = table;
    
    // Make sure table name is valid. Don't bother saving scanner reference since
    // later call to seek_record will create a new scanner object.
    try {
        ScanOptions options;
        string scanner;
        adapter->client->createScanner(scanner, adapter->login, adapter->table, options);
    }
    catch (const TableNotFoundException &) {
        *error = ADAPTER_ERROR_TABLE_NAME;
        return NULL;
    }
    catch (const TException &e) {
        *error = ADAPTER_ERROR_SOCKET;
        return NULL;
    }
    adapter->scanner = "";

    adapter->start_key = "";
    adapter->stop_key = "";
    adapter->start_key_inclusive = true;
    adapter->stop_key_inclusive = false;

    adapter->fill_value = NULL;

    return adapter;
}

void close_accumulo_adapter(accumulo_adapter_t *adapter)
{
    adapter->transport->close();
    clear_fill_value(adapter);
    delete adapter;
}

void add_missing_value(accumulo_adapter_t *adapter, std::string value)
{
    adapter->missing_values.push_back(value);
}

void clear_missing_values(accumulo_adapter_t *adapter)
{
    adapter->missing_values.clear();
}

void set_fill_value(accumulo_adapter_t *adapter, void *value, int size)
{
    if (adapter->fill_value != NULL) {
        delete adapter->fill_value;
    }

    adapter->fill_value = malloc(size);
    memcpy(adapter->fill_value, value, size);
}

void clear_fill_value(accumulo_adapter_t *adapter)
{
    if (adapter->fill_value != NULL) {
        delete adapter->fill_value;
        adapter->fill_value = NULL;
    }
}

AdapterError seek_record(accumulo_adapter_t *adapter, int record_offset)
{
    ScanOptions options;

    if (adapter->start_key != "") {
        options.range.start.row = adapter->start_key;
        // Accumulo C++ api doesn't seem to behave as expected when setting
        // start_key_inclusive, so if start_key_inclusive is true, set subkey
        // elements to empty strings so that everything with specified key
        // will be included. Otherwise, set subkey elements to ascii value 255
        // to exclude everything with specified key.
        if (adapter->start_key_inclusive) {
            options.range.start.colFamily = "";
            options.range.start.colQualifier = "";
            options.range.start.colVisibility = "";
        }
        else {
            options.range.start.colFamily = "\xff";
            options.range.start.colQualifier = "\xff";
            options.range.start.colVisibility = "\xff";
        }
        options.range.startInclusive = adapter->start_key_inclusive;
        options.range.__isset.start = true;
        options.range.__isset.startInclusive = true;
        options.__isset.range = true;
    }
    else {
        // If no start key is specified, use empty string which comes before
        // any other possible key
        options.range.start.row = "";
    }

    if (adapter->stop_key != "") {
        options.range.stop.row = adapter->stop_key;
        // Accumulo C++ api doesn't seem to behave as expected when setting
        // stop_key_inclusive, so if stop_key_inclusive is true, set subkey
        // elements to empty strings so that everything with specified key
        // will be included. Otherwise, set subkey elements to ascii value 255
        // to exclude everything with specified key.
        if (adapter->stop_key_inclusive) {
            options.range.stop.colFamily = "\xff";
            options.range.stop.colQualifier = "\xff";
            options.range.stop.colVisibility = "\xff";
        }
        else {
            options.range.stop.colFamily = "";
            options.range.stop.colQualifier = "";
            options.range.stop.colVisibility = "";
        }
        options.range.stopInclusive = adapter->stop_key_inclusive;
        options.range.__isset.stop = true;
        options.range.__isset.stopInclusive = true;
        options.__isset.range = true;
    }
    else {
        // If no stop key is specified, use invalid ascii char which comes after
        // any other possible key
        options.range.stop.row = "\xff";
    }

    // Create scanner object for reading Accumulo table. A reference to the
    // scanner object is stored in 'scanner' variable. The constructor should've
    // already verified that the table name is valid, but guard against thrown exception
    // just in case.
    string scanner;
    try {
        adapter->client->createScanner(scanner, adapter->login, adapter->table, options);
        adapter->scanner = scanner;
    }
    catch (const TableNotFoundException &) {
        return ADAPTER_ERROR_INVALID_TABLE_NAME;
    }
    
    // Seek to specified offset
    KeyValueAndPeek result;
    int records_found = 0;
    while (records_found < record_offset && adapter->client->hasNext(scanner)) {
        adapter->client->nextEntry(result, scanner);
        records_found++;
    }

    if (!adapter->client->hasNext(scanner)) {
        return ADAPTER_ERROR_INVALID_SEEK;
    }

    return ADAPTER_SUCCESS;
}

AdapterError read_records(accumulo_adapter_t *adapter, int num_records, int step,
    void *output, int *num_records_found)
{
    assert(adapter);
    assert(num_records_found);

    KeyValueAndPeek result;
    int64_t int64_value;
    uint64_t uint64_value;
    double float_value;
    *num_records_found = 0;
    bool truncation = false;
    bool missing_value = false;

    int i = 0;
    while (adapter->client->hasNext(adapter->scanner) && *num_records_found < num_records) {

        // Use scanner reference string to get result
        adapter->client->nextEntry(result, adapter->scanner);

        if (i++ % step != 0) {
            continue;
        }

        // If output is NULL, we're just using this function to seek from
        // current position without reading data into output array.
        if (output == NULL) {
            (*num_records_found)++;
            continue;
        }

        missing_value = false;
        if (result.keyValue.value == "") {
            missing_value = true;
        }
        else {
            vector<string>::iterator it = adapter->missing_values.begin();
            while (it != adapter->missing_values.end()) {
                if (result.keyValue.value == *it) {
                    missing_value = true;
                    break;
                }
                it++;
            }
        }

        const char *value_ptr = result.keyValue.value.c_str();
        int value_size = result.keyValue.value.size();
        char *end_ptr;
        int64_t result;
        uint64_t uresult;
        double double_result;
        if (missing_value) {
            if (adapter->fill_value != NULL) {
                memcpy(output, adapter->fill_value, adapter->output_type_size);
            }
            else {
                memset(output, 0, adapter->output_type_size);
            }
        }
        else if (adapter->field_type == INT_FIELD) {
            errno = 0;
#if defined(_WIN32)
            result = _strtoi64(value_ptr, &end_ptr, 10);
#else
            result = strtoll(value_ptr, &end_ptr, 10);
#endif
            if (*end_ptr != '\0') {
                return ADAPTER_ERROR_INT_CONVERSION;
            }
            else if (errno == ERANGE && (result == LLONG_MIN || result == LLONG_MAX)) {
                return ADAPTER_ERROR_INT_CONVERSION;
            }

            switch (adapter->output_type_size) {
            case 1:
                if (result > CHAR_MAX || result < CHAR_MIN) {
                    return ADAPTER_ERROR_INT_CONVERSION;
                }
                *(int8_t*)output = (int8_t)result;
                break;
            case 2:
                if (result > SHRT_MAX || result < SHRT_MIN) {
                    return ADAPTER_ERROR_INT_CONVERSION;
                }
                *(int16_t*)output = (int16_t)result;
                break;
            case 4:
                if (result > INT_MAX || result < INT_MIN) {
                    return ADAPTER_ERROR_INT_CONVERSION;
                }
                *(int32_t*)output = (int32_t)result;
                break;
            case 8:
                *(int64_t*)output = (int64_t)result;
                break;
            default:
                return ADAPTER_ERROR_INT_CONVERSION;
                break;
            }
        }
        else if (adapter->field_type == UINT_FIELD) {
            errno = 0;
#if defined(_WIN32)
            uresult = _strtoui64(value_ptr, &end_ptr, 10);
#else
            uresult = strtoull(value_ptr, &end_ptr, 10);
#endif
            if (*end_ptr != '\0') {
                return ADAPTER_ERROR_INT_CONVERSION;
            }
            else if (errno == ERANGE && uresult == ULLONG_MAX) {
                return ADAPTER_ERROR_INT_CONVERSION;
            }

            switch (adapter->output_type_size) {
            case 1:
                if (uresult > UCHAR_MAX) {
                    return ADAPTER_ERROR_INT_CONVERSION;
                }
                *(uint8_t*)output = (uint8_t)uresult;
                break;
            case 2:
                if (uresult > USHRT_MAX) {
                    return ADAPTER_ERROR_INT_CONVERSION;
                }
                *(uint16_t*)output = (uint16_t)uresult;
                break;
            case 4:
                if (uresult > UINT_MAX) {
                    return ADAPTER_ERROR_INT_CONVERSION;
                }
                *(uint32_t*)output = (uint32_t)uresult;
                break;
            case 8:
                *(uint64_t*)output = (uint64_t)uresult;
                break;
            default:
                return ADAPTER_ERROR_INT_CONVERSION;
            }
        }
        else if (adapter->field_type == FLOAT_FIELD) {
            errno = 0;
            double_result = strtod(value_ptr, &end_ptr);
            if (*end_ptr != '\0') {
                return ADAPTER_ERROR_FLOAT_CONVERSION;
            }
            else if (errno == ERANGE && (double_result == HUGE_VAL || double_result == -HUGE_VAL)) {
                return ADAPTER_ERROR_FLOAT_CONVERSION;
            }

            if (adapter->output_type_size == 4) {
                *(float*)output = (float)double_result;
            }
            else {
                *(double*)output = double_result;
            }
        }
        else if (adapter->field_type == STR_FIELD) {
            // This code assumes that string returned by Accumulo proxy
            // is ascii encoded.
            int output_size = value_size;
            if (value_size > adapter->output_type_size) {
                output_size = adapter->output_type_size;
                truncation = true;
            }
            strncpy(static_cast<char*>(output),
                    value_ptr,
                    output_size);
        }
        else {
            return ADAPTER_ERROR_OUTPUT_TYPE;
        }

        output = static_cast<char*>(output) + adapter->output_type_size;
        (*num_records_found)++;
    }

    if (*num_records_found < num_records) {
        return ADAPTER_ERROR_EOF;
    }

    if (truncation) {
        return ADAPTER_SUCCESS_TRUNCATION;
    }
    return ADAPTER_SUCCESS;
}
