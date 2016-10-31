#ifndef INDEX_H
#define INDEX_H

#include <_stdint.h>


/* buffer size of uncompressed gzip data */
#define UNCOMPRESSED_WINDOW_SIZE 32768

/* Default index density value. Density value determines how many records
   to skip between each indexed record */
#define DEFAULT_INDEX_DENSITY 1000

/* Default distance in bytes between gzip access points */
#define GZIP_ACCESS_POINT_DISTANCE 1024 * 1024


typedef struct record_offset_t
{
    uint64_t record_num;
    uint64_t offset;
} RecordOffset;


typedef struct gzip_index_access_point_t
{
    uint8_t bits;
    uint64_t compressed_offset;
    uint64_t uncompressed_offset;
    unsigned char window[UNCOMPRESSED_WINDOW_SIZE];
} GzipIndexAccessPoint;


/* indexer function pointer type */
typedef void (*indexer_func_ptr)(void *index, uint64_t record_num,
    uint64_t record_offset);

typedef RecordOffset (*index_lookup_func_ptr)(void *index, uint64_t record_num);

/* add gzip access point function pointer type */
typedef void (*add_gzip_access_point_func_ptr)(void *index,
    unsigned char *buffer,
    uint32_t compressed_offset,
    uint64_t uncompressed_offset,
    int avail_in,
    int avail_out,
    uint8_t data_type);

typedef void (*get_gzip_access_point_func_ptr)(void *index,
    uint64_t offset,
    GzipIndexAccessPoint *point);

void indexer_callback(void *index, uint64_t record_num, uint64_t record_offset);
RecordOffset index_lookup_callback(void *index, uint64_t record_num);

void add_gzip_access_point_callback(void *index,
    unsigned char *window,
    uint32_t compressed_offset,
    uint64_t uncompressed_offset,
    int avail_in,
    int avail_out,
    uint8_t bits);

void get_gzip_access_point_callback(void *index,
    uint64_t offset,
    GzipIndexAccessPoint *point);

#endif
