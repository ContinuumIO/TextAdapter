#ifndef POSTGRES_ADAPTER_H
#define POSTGRES_ADAPTER_H

#include <libpq-fe.h>

/* 
 * size of buffer used to hold intermediate numpy array values until
 * they can be casted to final type
 */
#define CAST_BUFFER_SIZE 10000

/*
 * PostgreSQL type defines copied from src/include/catalog/pg_type.h
 * It looks like the new conda package for the postgresql C lib contains this
 * header file, so we could probably include that header file instead of this
 # copy.
 */
#define BOOLOID			16
#define BYTEAOID		17
#define CHAROID			18
#define NAMEOID			19
#define INT8OID			20
#define INT2OID			21
#define INT2VECTOROID	22
#define INT4OID			23
#define REGPROCOID		24
#define TEXTOID			25
#define OIDOID			26
#define TIDOID		27
#define XIDOID 28
#define CIDOID 29
#define OIDVECTOROID	30
#define POINTOID		600
#define LSEGOID			601
#define PATHOID			602
#define BOXOID			603
#define POLYGONOID		604
#define LINEOID			628
#define FLOAT4OID 700
#define FLOAT8OID 701
#define ABSTIMEOID		702
#define RELTIMEOID		703
#define TINTERVALOID	704
#define UNKNOWNOID		705
#define CIRCLEOID		718
#define CASHOID 790
#define INETOID 869
#define CIDROID 650
#define BPCHAROID		1042
#define VARCHAROID		1043
#define DATEOID			1082
#define TIMEOID			1083
#define TIMESTAMPOID	1114
#define TIMESTAMPTZOID	1184
#define INTERVALOID		1186
#define TIMETZOID		1266
#define ZPBITOID	 1560
#define VARBITOID	  1562
#define NUMERICOID		1700

typedef struct
{
    PGconn *conn;
    PGresult *result;
    char *result_error_msg;
    const char *client_encoding;
    int **field_shapes;
    int postgis_geometry_oid;
} postgres_adapter_t;

typedef enum
{
    ADAPTER_SUCCESS,
    ADAPTER_ERROR_INVALID_QUERY,
    ADAPTER_ERROR_INVALID_QUERY_VALUE,
    ADAPTER_ERROR_INVALID_TYPE,
    ADAPTER_ERROR_INVALID_GIS_TYPE,
    ADAPTER_ERROR_PARSE_GIS
} AdapterError;

/*
 * Allocates and initializes postgres_adapter_t, and connects to database
 * using given connection URI.
 */
postgres_adapter_t* open_postgres_adapter(const char *connection_uri);

/*
 * Cleans up connection and query result objects, and postgres_adapter_t object
 */
void close_postgres_adapter(postgres_adapter_t *adapter);

/*
 * Executes given query. This function must be called before calling any functions
 * that return anything related to a query. Any query info or query results
 * returned later will be the state of the query at the time this function is
 * called. For example, if this function is called, then a new row is added to
 * a table that this query pulls from, that row won't be included in query
 * results unless this function is called again for query.
 */
AdapterError exec_query(postgres_adapter_t *adapter, const char *query);

/*
 * clean up query
 */
void clear_query(postgres_adapter_t *adapter);

/*
 * Get number of records in previously executed query.
 * exec_query must be called before this function.
 * The result of this function is the number of records at the time
 * exec_query was called.
 */
int get_num_records(postgres_adapter_t *adapter);

/*
 * Get number of fields in previously executed query.
 * exec_query must be called before this function.
 * The result of this function is the number of fields at the time
 * exec_query was called.
 */
int get_num_fields(postgres_adapter_t *adapter);

void set_field_dim_size(postgres_adapter_t *adapter, int field, int dim, int size);
void clear_field_size(postgres_adapter_t *adapter, int field);

/*
 * Read record data from previously executed query into preallocated memory
 * block pointed to by 'output'. Obviously, memory block must be big enough
 * to fit query data in; get_num_records and get_record_size functions can
 * be used to allocate enough space.
 * If values from db need to be casted to a custom dtype specified by user,
 * cast_type flag should be set to 1, and src and dst pointers should be set
 * to pointers to numpy buffer array and final array. As buffer array is filled
 * up with values from db, values in array will be copied to final array set to
 * user specified dtype, casting the values in the process.
 * exec_query must be called before this function.
 */
AdapterError read_records(postgres_adapter_t *adapter, int start, int stop, int step,
    char **output, int *num_records_found, int cast_types, void *src, void *dst,
    int dataframe);

int get_field_type(PGresult *result, int field);
int get_field_size(PGresult *result, int field);
int get_field_length(PGresult *result, int field);
int get_value_length(PGresult *result, int row, int field);
int get_value_size(PGresult *result, int row, int field);
char * get_field_data(PGresult *result, int row, int field);

#endif
