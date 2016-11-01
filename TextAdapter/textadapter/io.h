#include "text_adapter.h"

AdapterError seek_s3(InputData *input, uint64_t offset);
AdapterError read_s3(InputData *input, char *buffer, uint64_t buffer_len,
    uint64_t *num_bytes_read);
