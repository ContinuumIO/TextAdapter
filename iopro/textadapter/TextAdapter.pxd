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
    uint64_t UINT64_MAX

cdef extern from 'string.h':
    void *memcpy(void *, void *, size_t)
    char *strncpy(char *, char *, size_t)
    void *memset(void *, int, size_t)

cdef extern from "Python.h":
    ctypedef struct PyObject
    ctypedef struct FILE
    FILE* PyFile_AsFile(object)

cdef extern from 'stdlib.h':
    void* calloc(size_t, size_t)
    void* malloc(size_t)
    void* realloc(void *, size_t)
    void free(void *)


cdef extern from "../lib/khash.h":

    ctypedef uint32_t khint_t
    ctypedef khint_t khiter_t
    ctypedef char* kh_cstr_t

    ctypedef struct kh_string_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        khint_t *flags
        kh_cstr_t *keys
        PyObject **vals

    kh_string_t* kh_init_string()
    void kh_destroy_string(kh_string_t*)
    khint_t kh_get_string(kh_string_t*, kh_cstr_t)
    khint_t kh_put_string(kh_string_t*, kh_cstr_t, int*)
    khint_t kh_str_hash_func(const char *s)
    khint_t kh_exist(kh_string_t*, khint_t)

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, numpy.dtype descr,
                                int nd, numpy.npy_intp* dims,
                                numpy.npy_intp* strides, void* data,
                                int flags, object obj)
    struct PyArray_Descr:
        int type_num, elsize
        char type

    cdef enum:
        NPY_MAXDIMS


cdef extern from "zlib.h":
    int inflateEnd(void *)

cdef extern from "io_functions.h":
    InputData* open_file(const char *filename)
    void close_file(InputData *input)
    AdapterError read_file(InputData *input, char *buffer, uint64_t len,
        uint64_t *num_bytes_read)
    AdapterError seek_file(InputData *input, uint64_t offset)

    InputData* open_memmap(char *data, size_t size)
    void close_memmap(InputData *input)
    AdapterError read_memmap(InputData *input, char *buffer, uint64_t len,
        uint64_t *num_bytes_read)
    AdapterError seek_memmap(InputData *input, uint64_t offset)

    AdapterError read_gzip(InputData *input, char *buffer, uint64_t len,
        uint64_t *num_bytes_read)
    AdapterError seek_gzip(InputData *input, uint64_t offset)

    void init_gzip(InputData *input)
    void close_gzip(InputData *input)

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
        CONVERT_ERROR_NUMBA
        CONVERT_ERROR_LAST

    ctypedef ConvertError (*converter_func_ptr)(void *, uint32_t, int,
        void *, uint32_t, void *)

    ConvertError str2int_converter(void *input, uint32_t input_len,
        int input_type, void *output, uint32_t output_len, void *arg)
    ConvertError str2uint_converter(void *input, uint32_t input_len,
        int input_type, void *output, uint32_t output_len, void *arg)
    ConvertError str2float_converter(void *input, uint32_t input_len,
        int input_type, void *output, uint32_t output_len, void *arg)
    ConvertError str2str_converter(void *input, uint32_t input_len,
        int input_type, void *output, uint32_t output_len, void *arg)
    ConvertError str2complex_converter(void *input, uint32_t input_len,
        int input_type, void *output, uint32_t output_len, void *arg)

cdef extern from 'index.h':
    enum: UNCOMPRESSED_WINDOW_SIZE
    enum: DEFAULT_INDEX_DENSITY
    enum: GZIP_ACCESS_POINT_DISTANCE

    ctypedef struct RecordOffset:
        uint64_t record_num
        uint64_t offset

    ctypedef struct GzipIndexAccessPoint:
        uint8_t bits
        uint64_t compressed_offset
        uint64_t uncompressed_offset
        unsigned char window[UNCOMPRESSED_WINDOW_SIZE]

    ctypedef void (*indexer_func_ptr)(void *index, uint64_t record_num,
        uint64_t record_offset)
    ctypedef RecordOffset (*index_lookup_func_ptr)(void *index,
        uint64_t record_num)

    ctypedef void (*add_gzip_access_point_func_ptr)(void *index,
        unsigned char *buffer,
        uint32_t compressed_offset, uint64_t uncompressed_offset,
        int avail_in, int avail_out, uint8_t data_type)

    ctypedef void (*get_gzip_access_point_func_ptr)(void *index,
        uint64_t offset, GzipIndexAccessPoint *point)

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

    void set_num_fields(FieldList *fields, uint32_t num_fields)
    void init_missing_values(FieldList *fields, char *field_name,
        uint32_t field_num, uint32_t num_missing_values)
    void add_missing_value(FieldList *fields, char *field_name,
        uint32_t field_num, char *missing_value, uint32_t missing_value_len)
    void set_fill_value(FieldList *fields, char *field_name,
        uint32_t field_num, void *fill_value, uint32_t fill_value_len, int loose)
    uint32_t get_field_size(FieldList *fields, char *field_name, uint32_t field_num)

    ctypedef enum DefaultConverterFuncs:
        UINT_CONVERTER_FUNC
        INT_CONVERTER_FUNC
        FLOAT_CONVERTER_FUNC
        STRING_CONVERTER_FUNC
        STRING_OBJECT_CONVERTER_FUNC
        NUM_CONVERTER_FUNCS

    void set_field_width(FieldList *fields, uint32_t field, uint32_t width)
    void reset_converters(FieldList *fields)
    void set_converter(FieldList *fields, uint32_t field_num, char *field_name,
        uint32_t output_field_size, converter_func_ptr converter, void *converter_arg)
    int infer_types(FieldList *fields)


cdef extern from 'text_adapter.h':
    ctypedef enum AdapterError:
        ADAPTER_SUCCESS
        ADAPTER_ERROR_SEEK
        ADAPTER_ERROR_SEEK_EOF
        ADAPTER_ERROR_SEEK_S3
        ADAPTER_ERROR_READ
        ADAPTER_ERROR_READ_EOF
        ADAPTER_ERROR_READ_S3
        ADAPTER_ERROR_NO_FIELDS
        ADAPTER_ERROR_CONVERT
        ADAPTER_ERROR_INDEX
        ADAPTER_ERROR_PROCESS_TOKEN
        ADAPTER_ERROR_READ_TOKENS
        ADAPTER_ERROR_READ_RECORDS
        ADAPTER_ERROR_JSON
        ADAPTER_ERROR_INVALID_CHAR_CODE
        ADAPTER_ERROR_LAST


    ctypedef AdapterError (*read_func_ptr)(void *input, char *buffer,
        uint64_t len, uint64_t *num_bytes_read)
    ctypedef AdapterError (*seek_func_ptr)(void *input, uint64_t offset)
    ctypedef void (*close_func_ptr)(InputData *input)
    ctypedef AdapterError (*tokenize_func_ptr)(text_adapter_t *adapter,
        uint64_t num_tokens, uint64_t step, char **output,
        uint64_t *num_tokens_found, int enable_index, uint64_t index_density)

    ctypedef struct InputData:
        void *input
        read_func_ptr read
        seek_func_ptr seek
        close_func_ptr close
        void *compressed_input
        char *compressed_prebuffer
        read_func_ptr read_compressed
        seek_func_ptr seek_compressed
        get_gzip_access_point_func_ptr get_gzip_access_point
        uint64_t header
        uint64_t footer
        uint64_t start_record
        uint64_t start_offset
        void *index

    ctypedef struct MemMapInput:
        char *data
        uint64_t size
        uint64_t position

    ctypedef struct GzipInput:
        z_stream *z
        uint32_t compressed_bytes_processed
        uint64_t uncompressed_bytes_processed
        int buffer_refreshed
        void *uncompressed_input

    ctypedef struct JsonTokenizerArgs:
        JSON_checker_struct *jc

    ctypedef struct RegexTokenizerArgs:
        pcre *pcre_regex
        pcre_extra *extra_regex

    ctypedef struct ConvertErrorInfo:
        ConvertError convert_result
        char *token
        uint64_t record_num
        uint64_t field_num

    struct text_adapter_t:
        char delim_char
        char comment_char
        char quote_char
        char escape_char
        uint64_t num_records
        InputData *input_data
        tokenize_func_ptr tokenize
        void *tokenize_args
        uint64_t *field_widths
        void *index
        uint64_t index_density
        indexer_func_ptr indexer
        index_lookup_func_ptr index_lookup
        add_gzip_access_point_func_ptr add_gzip_access_point
        int infer_types_mode
        FieldList *fields
        int group_whitespace_delims
        int any_whitespace_as_delim
        int skipblanklines
        int reset_json_args

    AdapterError delim_tokenizer(text_adapter_t *adapter, uint64_t num_tokens,
        uint64_t step, char **output, uint64_t *num_tokens_found,
        int enable_index, uint64_t index_density)
    AdapterError json_tokenizer(text_adapter_t *adapter, uint64_t num_tokens,
        uint64_t step, char **output, uint64_t *num_tokens_found,
        int enable_index, uint64_t index_density)
    AdapterError json_record_tokenizer(text_adapter_t *adapter, uint64_t num_tokens,
        uint64_t step, char **output, uint64_t *num_tokens_found,
        int enable_index, uint64_t index_density)
    AdapterError regex_tokenizer(text_adapter_t *adapter, uint64_t num_tokens,
        uint64_t step, char **output, uint64_t *num_tokens_found,
        int enable_index, uint64_t index_density)
    AdapterError fixed_width_tokenizer(text_adapter_t *adapter,
        uint64_t num_tokens, uint64_t step, char **output,
        uint64_t *num_tokens_found, int enable_index, uint64_t index_density)
    AdapterError record_tokenizer(text_adapter_t *adapter, uint64_t num_tokens,
        uint64_t step, char **output, uint64_t *num_tokens_found,
        int enable_index, uint64_t index_density)
    AdapterError line_tokenizer(text_adapter_t *adapter, uint64_t num_tokens,
        uint64_t step, char **output, uint64_t *num_tokens_found,
        int enable_index, uint64_t index_density)

    AdapterError build_index(text_adapter_t *adapter)
    AdapterError build_gzip_index(text_adapter_t *adapter)

    text_adapter_t* open_text_adapter(InputData *input_data)

    void close_text_adapter(text_adapter_t *adapter)

    AdapterError seek_record(text_adapter_t *t, uint64_t rec_num)
    AdapterError seek_offset(text_adapter_t *t, uint64_t offset)
    AdapterError read_records(text_adapter_t *adapter, uint64_t num_records,
        uint64_t step, char *output, uint64_t *num_records_found)

    ConvertErrorInfo get_error_info()


# NOTE: This is after "text_adapter.h" so that
#       PCRE_STATIC gets defined before including pcre.h.
#       This is necessary for the Windows build.
cdef extern from "pcre.h":
    struct pcre
    struct pcre_extra
    pcre* pcre_compile(char *, int, char **, int *, unsigned char *)
    pcre_extra* pcre_study(pcre *, int, char **)

cdef extern from "zlib.h":
    ctypedef struct z_stream:
        pass

cdef extern from "json_tokenizer.h":
    struct JSON_checker_struct
    JSON_checker_struct* new_JSON_checker(int depth)

cdef extern converter_func_ptr default_converters[<unsigned int>NUM_CONVERTER_FUNCS]
