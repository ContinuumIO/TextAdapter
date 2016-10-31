#include "io_functions.h"
#include "stdlib.h"
#include "string.h"
#include <assert.h>


InputData* open_file(const char *filename)
{
    InputData *input;
    FILE *fh;

    if (filename == NULL) {
        return NULL;
    }

    input = calloc(1, sizeof(InputData));
    input->read = &read_file;
    input->seek = &seek_file;
    input->close = &close_file;

    fh = fopen(filename, "r");
    if (fh == NULL) {
        return NULL;
    }

    setvbuf(fh, NULL, _IONBF, 0);
    input->input = (void*)fh;
    return input;
}

void close_file(InputData *input)
{
    if (input == NULL) {
        return;
    }
    fclose(input->input);
    free(input);
}


AdapterError seek_file(InputData *input, uint64_t offset)
{
    int result;

    if (input == NULL)
        return ADAPTER_ERROR_SEEK;

    if (input->footer > 0)
    {
        uint64_t current, end, end_offset;
        current = ftell(input->input);
        fseek(input->input, 0, SEEK_END);
        end = ftell(input->input);
        fseek(input->input, current, SEEK_SET);

        end_offset = end - input->footer;

        if ((offset + input->header) > end_offset)
        {
            return ADAPTER_ERROR_SEEK;
        }
    }

    result = fseek((FILE*)input->input, offset + input->header, SEEK_SET);
    if (result != 0)
        return ADAPTER_ERROR_SEEK;

    input->start_offset = offset;
    return ADAPTER_SUCCESS;
}


AdapterError read_file(InputData *input, char *buffer, uint64_t len,
    uint64_t *num_bytes_read)
{
    uint64_t current;
    size_t bytes_read;

    if (input == NULL)
        return ADAPTER_ERROR_READ;

    if (num_bytes_read != NULL)
        *num_bytes_read = 0;

    current = ftell(input->input);

    if (input->footer > 0)
    {
        uint64_t end;
        fseek(input->input, 0, SEEK_END);
        end = ftell(input->input);
        fseek(input->input, current, SEEK_SET);

        if (current > end - input->footer)
        {
            return ADAPTER_ERROR_READ;
        }

        if (current + len > end - input->footer)
        {
            len = end - input->footer - current;
        }
    }

    bytes_read = fread(buffer, 1, len, input->input);

    if (bytes_read < len && ferror(input->input))
        return ADAPTER_ERROR_READ;

    if (num_bytes_read != NULL)
        *num_bytes_read = bytes_read;

    return ADAPTER_SUCCESS;
}


InputData* open_memmap(char *data, size_t size)
{
    InputData *input;
    MemMapInput *memmap_input;

    input = calloc(1, sizeof(InputData));
    if (input == NULL) {
        return NULL;
    }

    input->read = &read_memmap;
    input->seek = &seek_memmap;
    input->close = &close_memmap;

    memmap_input = calloc(1, sizeof(MemMapInput));
    if (memmap_input == NULL) {
        return NULL;
    }
    memmap_input->data = data;
    memmap_input->size = size;
    memmap_input->position = 0;

    input->input = memmap_input;
    return input;
}


void close_memmap(InputData *input)
{
    if (input != NULL) {
        if (input->input != NULL) {
            free(input->input);
        }
        free(input);
    }
}


AdapterError seek_memmap(InputData *input, uint64_t offset)
{
    MemMapInput *memmap_input;
    AdapterError result = ADAPTER_SUCCESS;

    if (input == NULL)
        return ADAPTER_ERROR_SEEK;

    memmap_input = (MemMapInput*)input->input;

    if ((offset + input->header) > memmap_input->size - input->footer)
    {
        memmap_input->position = memmap_input->size - input->footer;
    }
    else
    {
        memmap_input->position = offset + input->header;
    }

    input->start_offset = offset;

    return result;
}


AdapterError read_memmap(InputData *input, char *buffer, uint64_t len,
    uint64_t *num_bytes_read)
{
    MemMapInput *memmap_input;
    uint64_t bytes_left;

    if (input == NULL)
        return ADAPTER_ERROR_READ;

    memmap_input = (MemMapInput*)input->input;

    if (num_bytes_read != NULL)
        *num_bytes_read = 0;

    if (input->footer > 0
        && memmap_input->position >= memmap_input->size - input->footer)
    {
        return ADAPTER_ERROR_READ;
    }
   
    if (memmap_input->position >= memmap_input->size)
    {
        return ADAPTER_ERROR_READ_EOF;
    }

    bytes_left = memmap_input->size - memmap_input->position - input->footer;
    if (len > bytes_left)
        len = bytes_left;

    memcpy(buffer, memmap_input->data + memmap_input->position, len);
    memmap_input->position += len;

    if (num_bytes_read != NULL)
        *num_bytes_read = len;
        
    return ADAPTER_SUCCESS;
}


/* Seek offset in gzipped files. If index hasn't been built yet, seek will
   simply read from beginning of file until offset is reached. */
AdapterError seek_gzip(InputData *input, uint64_t offset)
{
    GzipIndexAccessPoint access_point;
    int ret;
    GzipInput *gzip_input;
    char *temp;
    char c;
    AdapterError result = ADAPTER_ERROR_SEEK_GZIP;
    uint64_t header;
    
    if (input == NULL)
        return ADAPTER_ERROR_SEEK;

    #ifdef DEBUG_ADAPTER 
    printf("seek_gzip(): offset=%llu\n", offset);
    #endif
   
    input->start_offset = 0;

    gzip_input = (GzipInput*)input->compressed_input; 

    if (gzip_input->z)
    {
        free(gzip_input->z);
        gzip_input->z = calloc(1, sizeof(z_stream));
    }

    gzip_input->compressed_bytes_processed = 0;
    gzip_input->uncompressed_bytes_processed = 0;
    gzip_input->buffer_refreshed = 0;
    gzip_input->z->zalloc = Z_NULL;
    gzip_input->z->zfree = Z_NULL;
    gzip_input->z->opaque = Z_NULL;
    gzip_input->z->avail_in = 0;
    gzip_input->z->next_in = Z_NULL;

    if (offset == 0)
    {
        inflateInit2(gzip_input->z, 47);
        /* Seek() skips header bytes, but we don't want to do this for gzip
         * file. Temporarily set header bytes to 0 so seek() will seek to
         * actual first byte in file. */
        header = input->header;
        input->header = 0;
        input->seek(input, 0);
        input->header = header;
        input->start_offset = offset;
        temp = calloc(input->header, sizeof(char));
        // Read header bytes
        input->read_compressed(input, temp, input->header, NULL);
        free(temp);
        return ADAPTER_SUCCESS;
    }
    else if (input->index == NULL)
    {
        inflateInit2(gzip_input->z, 47);
        /* Seek() skips header bytes, but we don't want to do this for gzip
         * file. Temporarily set header bytes to 0 so seek() will seek to
         * actual first byte in file. */
        header = input->header;
        input->header = 0;
        input->seek(input, 0);
        input->header = header;
        input->start_offset = offset;

        temp = calloc(input->header + offset, sizeof(char));
        // Read header bytes
        input->read_compressed(input, temp, offset + input->header, NULL);
        free(temp);

        return ADAPTER_SUCCESS;
    }
    
    inflateInit2(gzip_input->z, -15);

    input->get_gzip_access_point(input->index, offset, &access_point);

    /* Seek() skips header bytes, but we don't want to do this for gzip
     * file. Temporarily set header bytes to 0 so seek() will seek to
     * actual first byte in file. */
    header = input->header;
    input->header = 0;
    ret = input->seek(input, (access_point.compressed_offset -
        (access_point.bits ? 1 : 0)));
    input->header = header;
    if (ret == -1)
        return ADAPTER_ERROR_SEEK_GZIP;

    if (access_point.bits)
    {
        result = input->read(input, &c, 1, NULL);
        if (result != ADAPTER_SUCCESS)
        {
            return result;
        }
        
        ret = inflatePrime(gzip_input->z,
                           access_point.bits,
                           (int)c >> (8 - access_point.bits));
        if (ret < 0)
        {
            return ADAPTER_ERROR_SEEK_GZIP;
        }
    }

    ret = inflateSetDictionary(gzip_input->z,
                               access_point.window,
                               UNCOMPRESSED_WINDOW_SIZE);
    if (ret < 0)
    {
        return ADAPTER_ERROR_SEEK_GZIP;
    }

    temp = calloc(offset - access_point.uncompressed_offset, sizeof(char));
    // Read header bytes + rest of offset
    input->read_compressed(input, temp,
            offset - access_point.uncompressed_offset + input->header, NULL);
    free(temp);
   
    input->start_offset = offset;

    return ADAPTER_SUCCESS;
}


/* Read bytes from gzipped files into buffer */
AdapterError read_gzip(InputData *input, char *buffer, uint64_t len,
    uint64_t *num_bytes_read)
{
    GzipInput *gzip_input;
    uint64_t total_bytes_processed;
    uint64_t total_bytes_read;
    int eof_flag;
    uint64_t bytes_read;

    if (input == NULL)
        return ADAPTER_ERROR_READ;

    if (num_bytes_read != NULL)
        *num_bytes_read = 0;

    gzip_input = (GzipInput*)input->compressed_input;

    total_bytes_processed = 0;
    total_bytes_read = 0;
    gzip_input->z->next_out = (unsigned char *)buffer;
    gzip_input->z->avail_out = len;
    eof_flag = 0;
    bytes_read = 0;

    #ifdef DEBUG_ADAPTER
    printf("read_gzip(): buffer size: %u\n", (unsigned int)gzip_input->z->avail_out);
    #endif

    while (total_bytes_processed < len && eof_flag == 0)
    {
        AdapterError read_result;
        int result;

        if (gzip_input->z->avail_in == 0)
        {
            memset(input->compressed_prebuffer, '\0', COMPRESSED_BUFFER_SIZE);

            read_result = input->read(input, input->compressed_prebuffer,
                    COMPRESSED_BUFFER_SIZE, &bytes_read);

            if (read_result != ADAPTER_SUCCESS)
            {
                return read_result;
            }

            if (bytes_read < COMPRESSED_BUFFER_SIZE)
            {
                #ifdef DEBUG_ADAPTER
                printf("read_gzip(): %u bytes read; end of file reached\n", bytes_read);
                #endif

                eof_flag = 1;
            }

            gzip_input->z->avail_in = bytes_read;
            gzip_input->z->next_in = (unsigned char *)input->compressed_prebuffer;
        }

        do
        {
            result = inflate(gzip_input->z, Z_BLOCK);
        }
        while (result == Z_OK);

        if (result == Z_STREAM_END)
        {
            eof_flag = 1;
        }
        else if (result == Z_NEED_DICT
                 || result == Z_MEM_ERROR
                 || result == Z_DATA_ERROR)
        {
            #ifdef DEBUG_ADAPTER
            printf("read_gzip(): gzip inflate result=%d avail_in=%u avail_out=%u\n",
                    result, gzip_input->z->avail_in, gzip_input->z->avail_out);
            #endif
            return ADAPTER_ERROR_READ_GZIP;
        }

        total_bytes_processed = len - gzip_input->z->avail_out;
        total_bytes_read += bytes_read;
    }

    total_bytes_read -= gzip_input->z->avail_in;

    /* We have to seek from the beginning of the file because for some reason
       seeking backward caused eof flag to be raised on next read */
    gzip_input->compressed_bytes_processed += total_bytes_read;
    gzip_input->uncompressed_bytes_processed += total_bytes_processed;

    gzip_input->buffer_refreshed = 1;

    if (total_bytes_processed < len && !eof_flag)
    {
        return ADAPTER_ERROR_READ_GZIP;
    }

    if (num_bytes_read != NULL)
        *num_bytes_read = total_bytes_processed;

    return ADAPTER_SUCCESS;
}


void init_gzip(InputData *input)
{
    GzipInput *gzip_input;

    if (input == NULL) {
        return;
    }

    input->read_compressed = &read_gzip;
    input->seek_compressed = &seek_gzip;

    gzip_input = calloc(1, sizeof(GzipInput));
    gzip_input->compressed_bytes_processed = 0;
    gzip_input->uncompressed_bytes_processed = 0;
    gzip_input->buffer_refreshed = 0;
    gzip_input->z = calloc(1, sizeof(z_stream));
    gzip_input->uncompressed_input = input;

    input->compressed_input = gzip_input;
    input->compressed_prebuffer = calloc(COMPRESSED_BUFFER_SIZE, sizeof(char));
}


void close_gzip(InputData *input)
{
    if (input == NULL) {
        return;
    }

    if (input->compressed_input != NULL) {
        if (((GzipInput*)input->compressed_input)->z != NULL) {
            free(((GzipInput*)input->compressed_input)->z);
        }
        free(input->compressed_input);
    }

    if (input->compressed_prebuffer != NULL) {
        free(input->compressed_prebuffer);
    }
}
