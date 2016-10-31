#ifndef IO_FUNCTIONS_H
#define IO_FUNCTIONS_H

#include "text_adapter.h"

/* default file read/seek functions */
InputData* open_file(const char *filename);
void close_file(InputData *input);
AdapterError seek_file(InputData *input, uint64_t offset);
AdapterError read_file(InputData *input,
    char *buffer, uint64_t len, uint64_t *num_bytes_read);

/* memmap read/seek functions */
InputData* open_memmap(char *data, size_t size);
void close_memmap(InputData *input);
AdapterError seek_memmap(InputData *input, uint64_t offset);
AdapterError read_memmap(InputData *input,
    char *buffer, uint64_t len, uint64_t *num_bytes_read);

/* gzip read/seek functions */
AdapterError seek_gzip(InputData *input, uint64_t offset);
AdapterError read_gzip(InputData *input,
    char *buffer, uint64_t len, uint64_t *num_bytes_read);

/* setup/teardown functions for gzip decompression data structures */
void init_gzip(InputData *input);
void close_gzip(InputData *input);

#endif
