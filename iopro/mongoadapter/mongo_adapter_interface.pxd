cdef extern from 'stdlib.h':
    void* calloc(size_t, size_t)
    void* realloc(void *, size_t)
    void free(void *)

cdef extern from 'string.h':
    char *strncpy(char *, char *, size_t)
    void *memset(void *, int, size_t)
    void *memcpy(void *, void *, size_t)

cdef extern from 'mongo_adapter.h':
    int MONGO_HAVE_STDINT

cdef extern from 'bson.h':
    struct bson:
        pass

cdef extern from 'converters.h':
    ctypedef unsigned long long uint64_t

    ctypedef int (*converter_func_ptr)(void *, uint64_t, int, void *, uint64_t, void *)

cdef extern from 'mongo.h':
    struct mongo:
        pass
    struct mongo_cursor:
        pass
    double mongo_count(mongo*, char*, char*, bson*)

cdef extern from 'mongo_adapter.h':

    struct mongo_adapter_t:
        mongo conn
        mongo_cursor cursor
        char *database
        char *collection

        uint64_t num_fields
        char **field_names

        converter_func_ptr *converters
        void **converter_args
        uint64_t *field_sizes


    mongo_adapter_t* open_mongo_adapter(char *database_name, char *collection_name, char *host, int port)
    void close_mongo_adapter(mongo_adapter_t *adapter)

    void set_num_fields(mongo_adapter_t *adapter, uint64_t num_fields)

    void set_converter(mongo_adapter_t *adapter, uint64_t field_num, char * field_name, uint64_t field_size,
        converter_func_ptr converter, void *converter_arg)
    void clear_converters(mongo_adapter_t *adapter)

    uint64_t read_records(mongo_adapter_t *adapter, uint64_t start, uint64_t stop, int step, char *output)

    int mongo_int_converter(void *input, uint64_t input_len, int input_type, void *output, uint64_t output_len, void *arg)
    int mongo_uint_converter(void *input, uint64_t input_len, int input_type, void *output, uint64_t output_len, void *arg)
    int mongo_float_converter(void *input, uint64_t input_len, int input_type, void *output, uint64_t output_len, void *arg)

