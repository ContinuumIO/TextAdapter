#ifndef TEXTADAPTER_H
#define TEXTADAPTER_H

#ifdef _WIN32
#define PCRE_STATIC
#endif

#include <stdio.h>
#include <zlib.h>
#include <pcre.h>
#include "converter_functions.h"
#include "index.h"
#include "field_info.h"
#include "json_tokenizer.h"


/* Buffer size for reading in compressed gzip data before uncompressing */
#define COMPRESSED_BUFFER_SIZE 1024*1024


/* TextAdapter error codes */
typedef enum
{
    ADAPTER_SUCCESS,
    ADAPTER_ERROR_SEEK,
    ADAPTER_ERROR_SEEK_EOF,
    ADAPTER_ERROR_SEEK_GZIP,
    ADAPTER_ERROR_SEEK_S3,
    ADAPTER_ERROR_READ,
    ADAPTER_ERROR_READ_EOF,
    ADAPTER_ERROR_READ_GZIP,
    ADAPTER_ERROR_READ_S3,
    ADAPTER_ERROR_NO_FIELDS,
    ADAPTER_ERROR_CONVERT,
    ADAPTER_ERROR_INDEX,
    ADAPTER_ERROR_PROCESS_TOKEN,
    ADAPTER_ERROR_READ_TOKENS,
    ADAPTER_ERROR_READ_RECORDS,
    ADAPTER_ERROR_JSON,
    ADAPTER_ERROR_INVALID_CHAR_CODE,
    ADAPTER_ERROR_LAST
} AdapterError;


typedef enum tokenizer_state
{
    DEFAULT_STATE,
    RECORD_STATE,
    RECORD_END_STATE,
    COMMENT_STATE,
    QUOTE_STATE,
    QUOTE_END_STATE,
    PROCESS_STATE,
    ESCAPE_STATE
} TokenizerState;


typedef struct text_adapter_t TextAdapter;
typedef struct input_data_t InputData;


/* read function type for reading blocks of text from data source */
typedef AdapterError (*read_func_ptr)(InputData *input,
    char *buffer, uint64_t len, uint64_t *num_bytes_read);

/* seek function type for seeking to position in data source */
typedef AdapterError (*seek_func_ptr)(InputData *input,
    uint64_t offset);

/* cleans up any handles or pointers involved in reading from data source */
typedef void (*close_func_ptr)(InputData *input);

/* tokenize function for parsing text buffer and finding fields appropriate
   converter function should be called for each field that is found */
typedef AdapterError (*tokenize_func_ptr)(TextAdapter *adapter,
    uint64_t num_tokens, uint64_t step, char **output,
    uint64_t *num_tokens_found, int enable_index, uint64_t index_density);


struct input_data_t
{
    void *input;

    /* retrieves data chunks from data source and stores in buffer */
    read_func_ptr read;

    /* seeks to new position in data source */
    seek_func_ptr seek;

    /* cleans up any handles or pointers involved in reading from data source */
    close_func_ptr close;

    void *compressed_input;
    
    char *compressed_prebuffer;

    /* retrieves and decompresses data chunks from compressed data source
       and stores in buffer */
    read_func_ptr read_compressed;

    /* seeks to new position in compressed data source */
    seek_func_ptr seek_compressed;

    /* Retreive gzip access point from index */
    get_gzip_access_point_func_ptr get_gzip_access_point;

    /* number of bytes to skip at beginning of data stream */
    uint64_t header; 

    /* number of bytes to skip at end of data stream */
    uint64_t footer;

    /* Record where reading is started from after seek */
    uint64_t start_record;

    /* Data offset where reading is started from after seek */
    uint64_t start_offset;

    /* index of record offsets */
    void *index;

};


typedef struct memmap_input_t
{
    char *data;
    uint64_t size;
    uint64_t position;
} MemMapInput;


typedef struct gzip_input_t
{
    /* data struct for reading gzipped compressed data */
    z_stream *z;

    uint32_t compressed_bytes_processed;
    uint64_t uncompressed_bytes_processed;
    int buffer_refreshed;

    /* data struct for reading uncompressed data */
    void *uncompressed_input;
} GzipInput;


typedef struct json_tokenizer_args_t
{
    struct JSON_checker_struct *jc;
} JsonTokenizerArgs;

typedef struct regex_tokenizer_args_t
{
    pcre *pcre_regex;
    struct pcre_extra *extra_regex;
} RegexTokenizerArgs;


typedef struct text_adapter_buffer_t
{
    char *data;
    uint64_t size;
    uint64_t bytes_processed;
    int eof;
} TextAdapterBuffer;


typedef struct convert_error_info_t
{
    ConvertError convert_result;
    char *token;
    uint64_t record_num;
    uint64_t field_num;
} ConvertErrorInfo;


typedef struct text_adapter_t
{
    uint64_t num_records;

    char delim_char;
    char comment_char;
    char quote_char;
    char escape_char;

    /* Setting this to true will treat a series of whitespace
       as a single delimiter. Otherwise, each whitespace char
       will delimim a single field. */
    int group_whitespace_delims;
    int any_whitespace_as_delim;

    int infer_types_mode;

    /* If 0, empty lines will be treated as missing fields. Defaults to 1. */
    int skipblanklines;

    InputData *input_data;

    /* array of field info for each field */
    FieldList *fields;

    /* buffer for storing chunks of data from data source to be parsed */
    TextAdapterBuffer buffer;

    /* parses tokens in buffer */
    tokenize_func_ptr tokenize;
    void *tokenize_args;

    /* index of record offsets */
    void *index;

    /* Density of record offsets index. Density value x means every
       x-th record is indexed. */
    uint64_t index_density;

    /* function for building additional index info for specific
       data stream type */
    indexer_func_ptr indexer;
    index_lookup_func_ptr index_lookup;
    add_gzip_access_point_func_ptr add_gzip_access_point;

    int reset_json_args;

} TextAdapter;


/* Allocate new Recfile struct and set functions */
TextAdapter* open_text_adapter(InputData *input_data);

/* Deallocate Recfile struct */
void close_text_adapter(TextAdapter *adapter);

/* Seek to specific record in data source */
AdapterError seek_record(TextAdapter *adapter, uint64_t rec_num);

/* Read specified number of records from data source, starting at current
   position. Fields in records will be converted to data type and stored in
   output buffer. Output buffer should be big enough to store
   requested records. */
AdapterError read_records(TextAdapter *adapter, uint64_t num_records,
    uint64_t step, char *output, uint64_t *num_records_found);

/* default build index function */
AdapterError build_index(TextAdapter *adapter);

/* initialize default index info */
void clear_gzip_index(TextAdapter *adapter);

/* build index function for gzip files */
AdapterError build_gzip_index(TextAdapter *adapter);

/* default tokenize function based on delimiter */
AdapterError delim_tokenizer(TextAdapter *adapter, uint64_t num_tokens,
    uint64_t step, char **output, uint64_t *num_tokens_found,
    int enable_index, uint64_t index_density);

AdapterError json_tokenizer(TextAdapter *adapter, uint64_t num_tokens,
    uint64_t step, char **output, uint64_t *num_tokens_found,
    int enable_index, uint64_t index_density);

AdapterError json_record_tokenizer(TextAdapter *adapter, uint64_t num_tokens,
    uint64_t step, char **output, uint64_t *num_tokens_found,
    int enable_index, uint64_t index_density);

/* regular expression tokenize function */
AdapterError regex_tokenizer(TextAdapter *adapter, uint64_t num_tokens,
    uint64_t step, char **output, uint64_t *num_tokens_found,
    int enable_index, uint64_t index_density);

/* tokenize function based on predefined field widths */
AdapterError fixed_width_tokenizer(TextAdapter *adapter, uint64_t num_tokens,
    uint64_t step, char **output, uint64_t *num_tokens_found,
    int enable_index, uint64_t index_density);

AdapterError record_tokenizer(TextAdapter *adapter, uint64_t num_tokens,
    uint64_t step, char **output, uint64_t *num_tokens_found,
    int enable_index, uint64_t index_density);

AdapterError line_tokenizer(TextAdapter *adapter, uint64_t num_tokens,
    uint64_t step, char **output, uint64_t *num_tokens_found,
    int enable_index, uint64_t index_density);

ConvertErrorInfo get_error_info(void);


#endif
