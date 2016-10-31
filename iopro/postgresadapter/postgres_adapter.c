#include "postgres_adapter.h"
#include "postgis_fields.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#define NUM_POINT_PARAMS 2
#define NUM_LINE_PARAMS 3
#define NUM_LSEG_PARAMS 4
#define NUM_BOX_PARAMS 4
#define NUM_CIRCLE_PARAMS 3

#define PATH_HEADER 5
#define POLYGON_HEADER 4

#define FIELD_SHAPE_NDIM 4

void *create_list(void **);
void add_to_list(void *, double *, int);

// global variable to hold PostGIS OID, so we don't have
// to pass in adapter struct to functions just for this value
// that never changes
int postgis_geometry_oid = -1;

/*
 * Data coming from postgres is big endian, so swap bytes
 * assuming platform is little-endian (x86).
 */
void copy(uint8_t *dst, uint8_t *src, int num_bytes)
{
    int i, j;

    assert(dst != NULL);
    assert(src != NULL);

    j = num_bytes - 1;
    for (i = 0; i < num_bytes; i++) {
      dst[i] = src[j - i];
    }
}

void copy_items(uint8_t *dst, uint8_t *src, int item_size, int num_items)
{
    int i;

    assert(dst != NULL);
    assert(src != NULL);

    for (i = 0; i < num_items; i++) {
      copy(dst + i * item_size, src + i * item_size, item_size);
    }
}

void get_postgis_geometry_oid(postgres_adapter_t *adapter)
{
    AdapterError result;

    adapter->postgis_geometry_oid = -1;
    result = exec_query(adapter, "select oid from pg_catalog.pg_type where typname = 'geometry'");
    if (result != ADAPTER_SUCCESS) {
        return;
    }

    if (PQntuples(adapter->result) < 1) {
        return;
    }

    copy_items((uint8_t*)&adapter->postgis_geometry_oid, (uint8_t*)PQgetvalue(adapter->result, 0, 0), sizeof(int), 1);
    postgis_geometry_oid = adapter->postgis_geometry_oid;
    adapter->result = NULL;
}

postgres_adapter_t* open_postgres_adapter(const char *connection_uri)
{
    PGconn *conn;
    postgres_adapter_t *adapter;

    assert(connection_uri != NULL);

    conn = PQconnectdb(connection_uri);
    if (PQstatus(conn) != CONNECTION_OK) {
        PQfinish(conn);
        fprintf(stderr, "ERROR: could not connect to postgresql database\n");
        return NULL;
    }

    adapter = calloc(1, sizeof(postgres_adapter_t));
    if (adapter == NULL) {
        fprintf(stderr, "ERROR: could not allocate memory for postgres_adapter_t struct\n");
        return NULL;
    }

    adapter->conn = conn;
    adapter->result = NULL;
    adapter->result_error_msg = NULL;
    adapter->client_encoding = pg_encoding_to_char(PQclientEncoding(adapter->conn));
    adapter->field_shapes = NULL;

    get_postgis_geometry_oid(adapter);

    return adapter;
}

void close_postgres_adapter(postgres_adapter_t *adapter)
{
    assert(adapter != NULL);
    clear_query(adapter);
    PQfinish(adapter->conn);
    free(adapter);
}

AdapterError exec_query(postgres_adapter_t *adapter, const char *query)
{
    PGresult *result;
    char *msg;

    assert(adapter != NULL);
    assert(query != NULL);

    // Clean up previous query first
    if (adapter->result != NULL) {
        clear_query(adapter);
    }

    // Execute query and get binary results
	result = PQexecParams(adapter->conn, query, 0, NULL, NULL, NULL, NULL, 1);
	if (PQresultStatus(result) != PGRES_TUPLES_OK) {
        // Clear previous error message
        if (adapter->result_error_msg != NULL) {
            free(adapter->result_error_msg);
            adapter->result_error_msg = NULL;
        }
        msg = PQresultErrorMessage(result);
        if (msg != NULL) {
            adapter->result_error_msg = calloc(strlen(msg), 1);
            memcpy(adapter->result_error_msg, msg, strlen(msg));
        }
		PQclear(result);
        return ADAPTER_ERROR_INVALID_QUERY;
	}

    adapter->result = result;

    adapter->field_shapes = calloc(get_num_fields(adapter), sizeof(int*));

    return ADAPTER_SUCCESS;
}

void clear_query(postgres_adapter_t *adapter)
{
    int i;
    assert(adapter != NULL);
    if (adapter->field_shapes != NULL) {
        for (i = 0; i < get_num_fields(adapter); i++) {
            if (adapter->field_shapes[i] != NULL) {
                free(adapter->field_shapes[i]);
            }
        }
        free(adapter->field_shapes);
        adapter->field_shapes = NULL;
    }
    if (adapter->result_error_msg != NULL) {
        free(adapter->result_error_msg);
        adapter->result_error_msg = NULL;
    }
    if (adapter->result != NULL) {
        PQclear(adapter->result);
    }
    adapter->result = NULL;
}

int get_num_records(postgres_adapter_t *adapter)
{
    assert(adapter != NULL);
    assert(adapter->result != NULL);
    return PQntuples(adapter->result);
}

int get_num_fields(postgres_adapter_t *adapter)
{
    assert(adapter != NULL);
    assert(adapter->result != NULL);
    return PQnfields(adapter->result);
}

void set_field_dim_size(postgres_adapter_t *adapter, int field, int dim, int size)
{
    assert(adapter != NULL);
    assert(adapter->field_shapes != NULL);

    if (adapter->field_shapes[field] == NULL) {
        adapter->field_shapes[field] = calloc(FIELD_SHAPE_NDIM, sizeof(int));
    }
    adapter->field_shapes[field][dim] = size;
}

void clear_field_size(postgres_adapter_t *adapter, int field)
{
    if (adapter->field_shapes[field] != NULL) {
        free(adapter->field_shapes[field]);
        adapter->field_shapes[field] = NULL;
    }
}

int not_last_record(int row, int stop, int step)
{
    if (step > 0) {
        return row < stop;
    }
    else {
        return row > stop;
    }
}

int get_field_type(PGresult *result, int field)
{
    return PQftype(result, field);
}

int get_field_size(PGresult *result, int field)
{
    int type, size;
    
    type = PQftype(result, field);
    if (type == postgis_geometry_oid) {
        return sizeof(double);
    }
    else if (type == BPCHAROID || type == VARCHAROID || type == TEXTOID) {
        return -1;
    }
    else if (type == POINTOID ||
             type == LINEOID ||
             type == LSEGOID ||
             type == BOXOID ||
             type == CIRCLEOID ||
             type == PATHOID ||
             type == POLYGONOID) {
        return sizeof(double);
    }
    else {
        size = PQfsize(result, field);
        // If size is -1, we assume that this is a geometry field composed of
        // doubles, so return size of double
        if (size == -1) {
            return sizeof(double);
        }
        return size;
    }
}

int get_field_length(PGresult *result, int field)
{
    int type;
    char *data;
    int has_srid;
    int gis_type;

    type = PQftype(result, field);

    if (type == PATHOID || type == POLYGONOID || type == TEXTOID) {
        return -1;
    }
    else if (type == BPCHAROID || type == VARCHAROID) {
        // Get max number of chars in string.
        // Subtract 4 from PQfmod return value for header bytes.
        return PQfmod(result, field) - 4;
    }
    else if (type == postgis_geometry_oid) {
        data = PQgetvalue(result, 0, field);
        gis_type = get_gis_type(data, &has_srid);
        switch (gis_type) {
            case GIS_POINT2D:
                return 2;
            case GIS_POINT3D:
                return 3;
            case GIS_POINT4D:
                return 4;
            default:
                return -1;
        }
    }
    else {
        switch (type) {
            case POINTOID:
                return 2;
            case LINEOID:
                return 3;
            case LSEGOID:
            case BOXOID:
                return 4;
            case CIRCLEOID:
                return 3;
            default:
                // Assume rest of types are scalars
                return 1;
        }
    }
}

int get_value_length(PGresult *result, int row, int field)
{
    int has_srid;
    char *data;
    int type;
    int gis_type;

    type = PQftype(result, field);

    // special case for PostGIS Geometry type
    if (type == postgis_geometry_oid) {
        data = PQgetvalue(result, row, field);
        gis_type = get_gis_type(data, &has_srid);
        switch (gis_type) {
            case GIS_POINT2D:
                return 2;
            case GIS_POINT3D:
                return 3;
            case GIS_POINT4D:
                return 4;
            default:
                return -1;
        }

        return -1;
    }

    // normal postgres types
    switch (type) {
        case BOOLOID:
        case CHAROID:
        case INT8OID:
        case INT2OID:
        case INT4OID:
        case FLOAT4OID:
        case FLOAT8OID:
            return 1;
        case POINTOID:
            return 2;
        case LINEOID:
            return 3;
        case LSEGOID:
        case BOXOID:
            return 4;
        case CIRCLEOID:
            return 3;
        case BPCHAROID:
        case VARCHAROID:
            // For string fields, PQfmod returns number of characters + size
            // of 4 byte header
            return PQfmod(result, field) - 4;
        case TEXTOID:
            return 0;
        case PATHOID:
            return (PQgetlength(result, row, field) - PATH_HEADER) / get_field_size(result, field);
        case POLYGONOID:
            return (PQgetlength(result, row, field) - POLYGON_HEADER) / get_field_size(result, field);
    };
    
    return -1;
}

int get_value_size(PGresult *result, int row, int field)
{
    return PQgetlength(result, row, field);
}

char * get_field_data(PGresult *result, int row, int field)
{
    int type = get_field_type(result, field);
    if (type == PATHOID) {
        return PQgetvalue(result, row, field) + PATH_HEADER;
    }
    else if (type == POLYGONOID) {
        return PQgetvalue(result, row, field) + POLYGON_HEADER;
    }
    else {
        return PQgetvalue(result, row, field);
    }
}

void fill_nans(uint8_t *data, int float_size, int num_floats)
{
    int i;

    assert(float_size == 4 || float_size == 8);

    for (i = 0; i < num_floats; i++) {
        if (float_size == 4) {
            data[(i * float_size) + 2] = 0xc0;
            data[(i * float_size) + 3] = 0x7f;
        }
        else {
            data[(i * float_size) + 6] = 0xf8;
            data[(i * float_size) + 7] = 0x7f;
        }
    }
}

AdapterError read_records(postgres_adapter_t *adapter, int start, int stop, int step,
    char **output, int *num_records_found, int cast_types, void *src, void *dst,
    int dataframe)
{
    int num_fields;
    int row, field;
    int num_records_cast;
    int output_index;
    int field_type;
    char *data;
    int value_size;
    int num_values;
    int field_size;
    int field_length;
    int i;
    void *py_list;
    int max_num_values;
    char **output_start_ptr;
    char *temp;
    PGresult *result;

    assert(adapter != NULL);
    assert(output != NULL);
    assert(output[0] != NULL);
    assert(num_records_found != NULL);

    result = adapter->result;
    num_fields = PQnfields(result);
    output_start_ptr = calloc(num_fields, sizeof(char*));
    if (dataframe) {
        for (field = 0; field < num_fields; field++) {
            output_start_ptr[field] = output[field];
        }
    }
    else {
        output_start_ptr[0] = output[0];
    }
    num_records_cast = 0;
    output_index = 0;

    *num_records_found = 0;

    for (row = start; not_last_record(row, stop, step); row = row + step) {
        for (field = 0; field < num_fields; field++) {

            if (dataframe) {
                output_index = field;
            }

            field_type = get_field_type(adapter->result, field);
            data = get_field_data(adapter->result, row, field);
            value_size = get_value_size(adapter->result, row, field);
            num_values = get_value_length(adapter->result, row, field);
            field_size = get_field_size(adapter->result, field);
            field_length = get_field_length(adapter->result, field);

            // special case for PostGIS Geometry type
            if (field_type == postgis_geometry_oid) {
                if (!parse_gis_data(data, adapter->field_shapes[field], output + output_index, dataframe)) {
                    return ADAPTER_ERROR_PARSE_GIS;
                }
            }
            // normal postgres types
            else {
                switch (field_type) {
                    case INT2OID:
                    case INT4OID:
                    case INT8OID:
                    case FLOAT4OID:
                    case FLOAT8OID:
                    case POINTOID:
                    case LINEOID:
                    case LSEGOID:
                    case BOXOID:
                    case CIRCLEOID:
                        if (dataframe && (field_type == POINTOID ||
                                          field_type == LINEOID ||
                                          field_type == LSEGOID ||
                                          field_type == BOXOID ||
                                          field_type == CIRCLEOID)) {
                            temp = calloc(num_values, field_size);
                            copy_items((uint8_t*)temp, (uint8_t*)data, field_size, num_values);
                            py_list = (void*)create_list((void**)output + output_index);
                            for (i = 0; i < num_values; i++) {
                                add_to_list(py_list, ((double*)temp) + i, 1);
                            }
                            free(temp);
                        }
                        else if (value_size == 0) {
                            if (field_type == INT2OID || field_type == INT4OID || field_type == INT8OID) {
                                memset(output[output_index], 0, field_size * field_length);
                            }
                            else {
                                fill_nans((uint8_t*)output[output_index], field_size, field_length);
                            }
                        }
                        else {
                            copy_items((uint8_t*)output[output_index],
                                       (uint8_t*)data,
                                       field_size,
                                       num_values);
                        }
                        output[output_index] += field_size * field_length;
                        break;
                    case BPCHAROID:
                    case VARCHAROID:
                        if (dataframe) {
                            // Dataframe doesn't have a fixed width string type,
                            // so always output string object.
                            // convert_str2obj will handle incrementing output pointer
                            convert_str2object(data, adapter->client_encoding, output + output_index);
                        }
                        else {
                            // convert_str2str will handle incrementing output pointer
                            convert_str2str(data, adapter->client_encoding, output + output_index, field_length);
                        }
                        break;
                    case TEXTOID:
                        // convert_str2obj will handle incrementing output pointer
                        convert_str2object(data, adapter->client_encoding, output + output_index);
                        break;
                    case PATHOID:
                    case POLYGONOID:
                        if (!dataframe && adapter->field_shapes[field] != NULL) {
                            max_num_values = adapter->field_shapes[field][0] * 2;
                            if (num_values > max_num_values) {
                                num_values = max_num_values;
                            }

                            if (value_size == 0) {
                                fill_nans((uint8_t*)output[output_index], field_size, max_num_values);
                            }
                            else {
                                copy_items((uint8_t*)output[output_index], (uint8_t*)data, field_size, num_values);
                            }
                            output[output_index] += max_num_values * field_size;
                        }
                        else {
                            temp = calloc(num_values, field_size);
                            copy_items((uint8_t*)temp, (uint8_t*)data, field_size, num_values);
                            py_list = (void*)create_list((void*)(output + output_index));
                            for (i = 0; i < num_values; i = i + 2) {
                                add_to_list(py_list, ((double*)temp) + i, 2);
                            }
                            free(temp);
                        }
                        break;
                };
            }
        }

        (*num_records_found)++;

        // If we need to cast values to final dtype and the buffer array is
        // full, call cast_arrays function and pass in pointers to buffer array
        // and final array.
        if (cast_types && (*num_records_found) % CAST_BUFFER_SIZE == 0) {
            if (dataframe) {
                cast_dataframes(dst, src, num_records_cast, CAST_BUFFER_SIZE, num_fields);
                for (field = 0; field < num_fields; field++) {
                    output[field] = output_start_ptr[field];
                }
                num_records_cast = *num_records_found;
            }
            else {
                cast_arrays(dst, src, num_records_cast, CAST_BUFFER_SIZE);
                // Reset output pointer to start of buffer array
                output[0] = output_start_ptr[0];
                num_records_cast = *num_records_found;
            }
        }
    }

    // Cast remaining values in buffer array
    if (cast_types && num_records_cast < *num_records_found) {
        if (dataframe) {
            cast_dataframes(dst, src, num_records_cast, *num_records_found - num_records_cast, num_fields);
        }
        else {
            cast_arrays(dst, src, num_records_cast, *num_records_found - num_records_cast);
        }
    }

    return ADAPTER_SUCCESS;
}
