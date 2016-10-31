cimport numpy

cdef extern from '_stdint.h':
    # Actual type lengths are defined in _stdint.h
    # Sizes here are just place holders
    ctypedef unsigned long long uint64_t
    ctypedef unsigned int uint32_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned char uint8_t
    ctypedef long long int64_t
    ctypedef int int32_t
    ctypedef short int16_t
    ctypedef char int8_t


cdef extern from 'converter_functions.h':

    ctypedef enum ConvertError:
        CONVERT_SUCCESS
        CONVERT_ERROR_UNKNOWN
        CONVERT_ERROR_OVERFLOW
        CONVERT_ERROR_INPUT_TYPE
        CONVERT_ERROR_INPUT_SIZE
        CONVERT_ERROR_OUTPUT_SIZE
        CONVERT_ERROR_INPUT_STRING
        CONVERT_ERROR_USER_CONVERTER
        CONVERT_ERROR_OBJECT_CONVERTER
        CONVERT_ERROR_LAST

    ctypedef ConvertError (*converter_func_ptr)(void *, uint32_t, int, void *, uint32_t, void *)

    #ConvertError str2int_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg)
    #ConvertError str2uint_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg)
    #ConvertError str2float_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg)
    #ConvertError str2str_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg)
    #ConvertError str2complex_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg)


cdef extern from 'stdlib.h':
    void* calloc(size_t, size_t)
    void* realloc(void *, size_t)
    void free(void *)

cdef extern from "Python.h":
    ctypedef struct PyObject

cdef extern from 'field_info.h':
    ctypedef struct MissingValues:
        char **missing_values
        uint32_t *missing_value_lens
        uint32_t num_missing_values

    ctypedef struct FillValue:
        void *fill_value
        int loose

    ctypedef struct FieldInfo:
        char *name
        converter_func_ptr converter
        void *converter_arg
        MissingValues missing_values
        FillValue fill_value
        uint32_t input_field_width
        uint32_t output_field_size
        int infer_type

    ctypedef struct FieldList:
        uint32_t num_fields
        FieldInfo *field_info

    ctypedef enum DefaultConverterFuncs:
        UINT_CONVERTER_FUNC
        INT_CONVERTER_FUNC
        FLOAT_CONVERTER_FUNC
        STRING_CONVERTER_FUNC
        STRING_OBJECT_CONVERTER_FUNC
        NUM_CONVERTER_FUNCS

    void clear_fields(FieldList *fields)
    void set_num_fields(FieldList *fields, uint32_t num_fields)
    FieldInfo *get_field_info(FieldList *fields, char *field_name, uint32_t field_num)

    void clear_missing_values(MissingValues *missing_values)
    void clear_fill_value(FillValue *fill_value)

    void init_infer_types(FieldList *fields)

    void init_missing_values(FieldList *fields, char *field_name,
        uint32_t field_num, uint32_t num_missing_values)

    void add_missing_value(FieldList *fields, char *field_name,
        uint32_t field_num, char *missing_value, uint32_t missing_value_len)

    void set_fill_value(FieldList *fields, char *field_name,
        uint32_t field_num, void *fill_value, uint32_t fill_value_len, int loose)

    uint32_t get_field_size(FieldList *fields, char *field_name, uint32_t field_num)
    uint32_t get_output_record_size(FieldList *fields)

    void set_field_width(FieldList *fields, uint32_t field, uint32_t width)

    void reset_converters(FieldList *fields)

    void set_converter(FieldList *fields, uint32_t field_num, char *field_name,
        uint32_t output_field_size, converter_func_ptr converter, void *converter_arg)


'''cdef extern from 'bson.h':

    ctypedef enum bson_type:
        BSON_INT
        BSON_LONG
        BSON_DOUBLE
        BSON_STRING'''


cdef extern from 'mongo_adapter.h':
    struct mongo_adapter_t:
        char *database
        char *collection
        FieldList *fields
        converter_func_ptr default_converters[<unsigned int>NUM_CONVERTER_FUNCS]


    ctypedef enum MongoAdapterError:
        MONGO_ADAPTER_SUCCESS
        MONGO_ADAPTER_ERROR
        MONGO_ADAPTER_ERROR_TYPE_CHANGED
        MONGO_ADAPTER_ERROR_INVALID_START_REC

    mongo_adapter_t* open_mongo_adapter(char *host, int port, char *database_name, char *collection_name)
    void close_mongo_adapter(mongo_adapter_t *adapter)

    uint32_t get_num_records(mongo_adapter_t *adapter)

    MongoAdapterError read_records(mongo_adapter_t *adapter, uint32_t start_record, uint32_t num_records,
        int32_t step, void *output, uint32_t *num_records_read)

    ConvertError mongo2int_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg)
    ConvertError mongo2uint_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg)
    ConvertError mongo2float_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg)
