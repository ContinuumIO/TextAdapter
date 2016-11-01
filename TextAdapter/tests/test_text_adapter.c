#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <assert.h>
#include "../textadapter/text_adapter.h"


#define PRINT_RECORDS(data, num_rows, num_fields, format) \
do { \
    int j; \
    int k; \
    for (j = 0; j < num_rows; j++) \
    { \
        for (k = 0; k < num_fields; k++) \
        { \
            printf(format, *(data+(j*num_fields)+k)); \
            if (k < num_fields - 1) \
                printf(" "); \
        } \
        printf("\n"); \
    } \
} while (0) 


double get_time()
{
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec + (time.tv_usec / 1000000.0);
}


void test_ints(char *filename, uint64_t num_recs)
{
    uint64_t num_fields = 5;

    FILE *input = fopen(filename, "r");
    setvbuf(input, NULL, _IONBF, 0);
    
    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL, NULL);
    adapter->tokenize = &tokenizer;
    set_num_fields(adapter, num_fields);
    adapter->delim_char = ',';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(uint32_t), &uint_converter, NULL);
    }

    uint32_t *data = calloc(num_recs, sizeof(uint32_t)*num_fields);

    fseek(input, 0, SEEK_SET);

    double t1 = get_time();
    int result = read_records(adapter, num_recs, 1, (char *)data, NULL);
    double t2 = get_time();

    assert(result == ADAPTER_SUCCESS);

    uint64_t i;
    for (i = 0; i < num_recs * num_fields; i++)
    {
        if (*(data+i) != i)
            printf("JNB: %u %llu\n", *(uint32_t*)(data+i), i);
        assert(*(data + i) == i);
    }

    printf("PASSED: read %s in %.2lf seconds\n", filename, t2-t1);

    free(data);
    close_text_adapter(adapter);
}


void test_signed_ints(char *filename, uint64_t num_recs, uint64_t num_fields)
{
    FILE *input = fopen(filename, "r");
    setvbuf(input, NULL, _IONBF, 0);
    
    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL, NULL);
    adapter->tokenize = &tokenizer;
    set_num_fields(adapter, num_fields);
    adapter->delim_char = ',';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(int32_t), &int_converter, NULL);
    }

    int *data = calloc(num_recs, sizeof(int32_t)*num_fields);

    double t1 = get_time();
    int result = read_records(adapter, num_recs, 1, (char *)data, NULL);
    /*PRINT_RECORDS(data, 10, 5, "%u"); */
    double t2 = get_time();
    
    assert(result == ADAPTER_SUCCESS);

    uint64_t i;
    int64_t num = -1;
    for (i = 0; i < num_recs * num_fields; i++)
    {
        assert(*(data + i) == num);
        if (num < 0)
            num -= 1;
        else
            num += 1;
        num *= -1;
    }

    printf("PASSED: read %s in %.2lf seconds\n", filename, t2-t1);

    free(data);
    close_text_adapter(adapter);
}


void test_floats(char *filename, uint64_t num_recs)
{
    uint64_t num_fields = 5;

    FILE *input = fopen(filename, "r");
    setvbuf(input, NULL, _IONBF, 0);
    
    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL, NULL);
    adapter->tokenize = &tokenizer;
    set_num_fields(adapter, num_fields);
    adapter->delim_char = ',';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(double), &float_converter, NULL);
    }

    double *data = calloc(num_recs, sizeof(double)*num_fields);

    double t1 = get_time();
    int result = read_records(adapter, num_recs, 1, (char *)data, NULL);
    /*PRINT_RECORDS(data, 10, 5, "%u"); */
    double t2 = get_time();

    assert(result == ADAPTER_SUCCESS);

    uint64_t i;
    double value = 0.0;
    for (i = 0; i < num_recs * num_fields; i++)
    {
        if (*(data+i) > value && (*(data+i) - value) > 0.01f)
            printf("JNB: %f %f %f\n", *(data+i), value, *(data+i) - value);
        else if (*(data+i) <= value && (*(data+i) - value) < -0.01f)
            printf("JNB: %f %f %f\n", *(data+i), value, *(data+i) - value);
        if (*(data+i) > value)
            assert(abs(*(data+i) - value) < 0.01f);
        else if (*(data+i) < value)
            assert(abs(*(data+i) - value) > -0.01f);
        value += 0.1;
    }

    printf("PASSED: read %s in %.2lf seconds\n", filename, t2-t1);

    free(data);
    close_text_adapter(adapter);
}


void test_index(char *filename, uint64_t num_recs)
{
    uint64_t num_fields = 5;

    FILE *input = fopen(filename, "r");
    
    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL, &indexer);
    adapter->tokenize = &tokenizer;
    set_num_fields(adapter, num_fields);
    adapter->delim_char = ',';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(uint32_t), &uint_converter, NULL);
    }

    int *data = calloc(1, sizeof(uint32_t)*num_fields);

    double t1 = get_time();
    build_index(adapter, 1);
    double t2 = get_time();

    seek_record(adapter, (num_recs / 2));
    int result = read_records(adapter, 1, 1, (char *)data, NULL);
    assert(*data == (num_recs / 2) * 5);

    assert(result == ADAPTER_SUCCESS);

    seek_record(adapter, num_recs - 1);
    result = read_records(adapter, 1, 1, (char *)data, NULL);
    assert(*data == (num_recs - 1) * 5);

    assert(result == ADAPTER_SUCCESS);

    seek_record(adapter, 0);
    result = read_records(adapter, 1, 1, (char *)data, NULL);
    assert(*data == 0);

    assert(result == ADAPTER_SUCCESS);

    printf("PASSED: indexed ./data/ints in %.2lf seconds\n", t2-t1);

    free(data);
    close_text_adapter(adapter);
}


void test_step()
{
    uint64_t num_fields = 5;

    FILE *input = fopen("./data/ints", "r");
    
    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL, NULL);
    adapter->tokenize = &tokenizer;
    assert(adapter != NULL);
    setvbuf(input, NULL, _IONBF, 0);

    set_num_fields(adapter, num_fields);
    adapter->delim_char = ',';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(uint32_t), &uint_converter, NULL);
    }

    int *data = calloc(5, sizeof(uint32_t)*num_fields);

    double t1 = get_time();
    int result = read_records(adapter, 10, 2, (char *)data, NULL);
    double t2 = get_time();

    assert(result == ADAPTER_SUCCESS);

    uint64_t offset = 0;
    uint64_t i;
    for (i = 0; i < 10; i += 2)
    {
        uint64_t j;
        for (j = 0; j < 5; j++)
        {
            assert(*(data+offset) == 5*i+j);
            offset++;
        }
    }

    printf("PASSED: stepped over every other record in ./data/ints in %.2lf seconds\n", t2-t1);

    free(data);
    close_text_adapter(adapter);
}


void test_mmap()
{
    uint64_t num_recs = 2500000;
    uint64_t num_fields = 5;

    FILE *input = fopen("./data/ints", "r");
    
    fseek(input, 0, SEEK_END);
    uint64_t len = ftell(input);
    fseek(input, 0, SEEK_SET);
    void *memmap = mmap(0, len, PROT_READ, MAP_SHARED, (int)fileno(input), 0);
    MemMapInput memmap_input;
    memmap_input.data = memmap;
    memmap_input.size = len;
    memmap_input.position = 0;
    TextAdapter *adapter = open_text_adapter((void *)&memmap_input, NULL, &read_memmap, NULL, &seek_memmap, NULL, NULL);
    adapter->tokenize = &tokenizer;
    setvbuf(input, NULL, _IONBF, 0);
    
    set_num_fields(adapter, num_fields);
    adapter->delim_char = ',';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(uint32_t), &uint_converter, NULL);
    }

    int *data = calloc(num_recs, sizeof(int)*num_fields);

    double t1 = get_time();
    int result = read_records(adapter, num_recs, 1, (char *)data, NULL);
    double t2 = get_time();

    assert(result == ADAPTER_SUCCESS);

    uint64_t i;
    for (i = 0; i < num_recs * num_fields; i++)
    {
        if (*(data+i) != i)
            printf("JNB: %u %llu\n", *(data+i), i);
        assert(*(data + i) == i);
    }

    printf("PASSED: read memmapped ./data/ints in %.2lf seconds\n", t2-t1);

    free(data);
    close_text_adapter(adapter);
}


void test_gzip()
{
    uint64_t num_recs = 2500000;
    uint64_t num_fields = 5;

    FILE *input = fopen("./data/ints.gz", "r");

    GzipState gzip_state;
    gzip_state.compressed_bytes_processed = 0;
    gzip_state.uncompressed_bytes_processed = 0;
    gzip_state.buffer_refreshed = 0;

    gzip_state.z = calloc(1, sizeof(z_stream));
    gzip_state.z->zalloc = Z_NULL;
    gzip_state.z->zfree = Z_NULL;
    gzip_state.z->opaque = Z_NULL;
    gzip_state.z->avail_in = 0;
    gzip_state.z->next_in = Z_NULL;
    inflateInit2(gzip_state.z, 47);

    TextAdapter *adapter = open_text_adapter((void *)input, (void*)&gzip_state, &read_file, &read_gzip, &seek_file, &seek_gzip, NULL);
    adapter->tokenize = &tokenizer;

    set_num_fields(adapter, num_fields);
    adapter->delim_char = ',';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(uint32_t), &uint_converter, NULL);
    }

    int *data = calloc(num_recs, sizeof(int)*num_fields);

    int result = read_records(adapter, num_recs, 1, (char *)data, NULL);
    seek_record(adapter, 0);

    assert(result == ADAPTER_SUCCESS);

    double t1 = get_time();
    result = read_records(adapter, num_recs, 1, (char *)data, NULL);
    double t2 = get_time();

    assert(result == ADAPTER_SUCCESS);

    uint64_t i;
    for (i = 0; i < num_recs * num_fields; i++)
    {
        if (*(data+i) != i)
        {
            printf("JNB: %u %llu\n", *(data+i), i);
        }
        assert(*(data + i) == i);
    }

    printf("PASSED: read gzipped ./data/ints.gz in %.2lf seconds\n", t2-t1);

    free(data);
    close_text_adapter(adapter);
}


void test_gzip_index()
{
    uint64_t num_recs = 2500000;
    uint64_t num_fields = 5;

    FILE *input = fopen("./data/ints.gz", "r");

    GzipState gzip_state;
    gzip_state.compressed_bytes_processed = 0;
    gzip_state.uncompressed_bytes_processed = 0;
    gzip_state.z = calloc(1, sizeof(z_stream));
    gzip_state.z->zalloc = Z_NULL;
    gzip_state.z->zfree = Z_NULL;
    gzip_state.z->opaque = Z_NULL;
    gzip_state.z->avail_in = 0;
    gzip_state.z->next_in = Z_NULL;
    inflateInit2(gzip_state.z, 47);

    TextAdapter *adapter = open_text_adapter((void *)input, (void*)&gzip_state, &read_file, &read_gzip, &seek_file, &seek_gzip, &indexer);
    adapter->tokenize = &tokenizer;

    set_num_fields(adapter, num_fields);
    adapter->delim_char = ',';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(uint32_t), &uint_converter, NULL);
    }

    int *data = calloc(num_recs, sizeof(int)*num_fields);

    double t1_index = get_time();
    build_gzip_index(adapter, 1);
    double t2_index = get_time();

    GzipIndex *gzip_index = (GzipIndex*)adapter->index->compression_index;
    uint64_t num_access_points = gzip_index->num_access_points;
    printf("index built using %llu bytes of memory\n", (num_access_points * sizeof(GzipIndexAccessPoint)));
    printf("testing record seeks...\n");

    seek_record(adapter, 0);
    int result = read_records(adapter, 1, 1, (char *)data, NULL);
    
    assert(result == ADAPTER_SUCCESS);

    double t1_seek = get_time();
    seek_record(adapter, 100000);
    double t2_seek = get_time();
    result = read_records(adapter, 1, 1, (char *)data, NULL);
    assert(result == ADAPTER_SUCCESS);
    assert(*data == 500000);

    seek_record(adapter, 2000000);
    result = read_records(adapter, 1, 1, (char *)data, NULL);
    assert(result == ADAPTER_SUCCESS);
    assert(*data == 10000000);

    seek_record(adapter, 0);
    result = read_records(adapter, 1, 1, (char *)data, NULL);
    assert(result == ADAPTER_SUCCESS);
    assert(*data == 0);

    seek_record(adapter, 2499999);
    read_records(adapter, 1, 1, (char *)data, NULL);
    assert(result == ADAPTER_SUCCESS);
    assert(*data == 12499995);

    printf("PASSED: indexed ./data/ints.gz in %.2lf seconds\n", t2_index-t1_index);
    printf("        seeked to record 100000 in %.2lf seconds\n", t2_seek-t1_seek);

    free(data);
    close_text_adapter(adapter);

}


void test_regex(char *filename, uint64_t num_recs)
{
    uint64_t num_fields = 5;

    FILE *input = fopen(filename, "r");
    setvbuf(input, NULL, _IONBF, 0);
   
    /*regcomp(&recfile->re, "([0-9]*),([0-9]*),([0-9]*),([0-9]*),([0-9]*)\n", REG_EXTENDED); */
    /*const char *error; */
    /*int erroroffset; */
    /*recfile->tokenize_args = pcre_compile("([0-9]*),([0-9]*),([0-9]*),([0-9]*),([0-9]*)\n", 0, &error, &erroroffset, NULL); */
    /*adapter->delim_char = ','; */
    /*adapter->quote_char = '"'; */
    /*adapter->comment_char = '\0'; */
    const char *error;
    int error_offset;
    struct regex_tokenizer_args_t regex_args;
    regex_args.pcre_regex = pcre_compile("^([0-9]*),([0-9]*),([0-9]*),([0-9]*),([0-9]*)", PCRE_EXTENDED, &error, &error_offset, 0);
    regex_args.extra_regex = pcre_study(regex_args.pcre_regex, 0, &error);

    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL, NULL);
    adapter->tokenize = &regex_tokenizer;
    set_num_fields(adapter, num_fields);
    adapter->tokenize_args = (void*)&regex_args;

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(uint32_t), &uint_converter, NULL);
    }

    uint32_t *data = calloc(num_recs, sizeof(uint32_t)*num_fields);

    fseek(input, 0, SEEK_SET);

    double t1 = get_time();
    int result = read_records(adapter, num_recs, 1, (char *)data, NULL);
    double t2 = get_time();
    
    assert(result == ADAPTER_SUCCESS);

    uint64_t i;
    for (i = 0; i < num_recs * num_fields; i++)
    {
        if (*(data+i) != i)
            printf("JNB: %u %llu\n", *(uint32_t*)(data+i), i);
        assert(*(data + i) == i);
    }

    printf("PASSED: read %s in %.2lf seconds\n", filename, t2-t1);

    free(data);
    close_text_adapter(adapter);
}


void test_fixed_width()
{
    uint64_t num_recs = 1000000;
    uint64_t num_fields = 5;

    FILE *input = fopen("./data/fixedwidths", "r");
    setvbuf(input, NULL, _IONBF, 0);
    
    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL, NULL);
    adapter->tokenize = &fixed_width_tokenizer;
    set_num_fields(adapter, num_fields);
    adapter->delim_char = '\0';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    uint32_t i;
    for (i = 0; i < num_fields; i++)
    {
        set_converter(adapter, i, sizeof(uint32_t), &uint_converter, NULL);
        adapter->field_widths[i] = i + 2;
    }

    uint32_t *data = calloc(num_recs, sizeof(uint32_t)*num_fields);

    fseek(input, 0, SEEK_SET);

    double t1 = get_time();
    int result = read_records(adapter, num_recs, 1, (char *)data, NULL);
    double t2 = get_time();
    
    assert(result == ADAPTER_SUCCESS);

    uint32_t values[5];
    values[0] = 0;
    values[1] = 0;
    values[2] = 0;
    values[3] = 0;
    values[4] = 0;

    for (i = 0; i < num_recs * num_fields; i++)
    {
        uint32_t index = i % num_fields;
        if (*(data+i) != values[index])
            printf("JNB: %u %u\n", *(uint32_t*)(data+i), values[index]);
        assert(*(data + i) == values[index]);
        if ((i+1) % num_fields == 0)
        {
            values[0] = (values[0] + 1) % 100;
            values[1] = (values[1] + 1) % 1000;
            values[2] = (values[2] + 1) % 10000;
            values[3] = (values[3] + 1) % 100000;
            values[4] = (values[4] + 1) % 1000000;
        }
    }

    printf("PASSED: read ./data/fixedwidth in %.2lf seconds\n", t2-t1);

    free(data);
    close_text_adapter(adapter);
}


void test_type_inference()
{
    FILE *input = fopen("./data/test.csv", "r");
    setvbuf(input, NULL, _IONBF, 0);
    
    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL, NULL);
    adapter->tokenize = &tokenizer;
    set_num_fields(adapter, 37);
    adapter->delim_char = ',';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    fseek(input, 0, SEEK_SET);

    int i;
    for (i = 0; i < adapter->num_fields; i++)
    {
        set_converter(adapter, i, sizeof(uint64_t), &uint_converter, NULL);
    }

    double t1 = get_time();
    adapter->infer_types = 1;
    read_records(adapter, 9, 1, NULL, NULL);
    /*adapter->infer_types = 0; */
    /*int result = read_records(adapter, 9, 1, NULL, NULL); */
    double t2 = get_time();

    /*assert(result == ADAPTER_SUCCESS); */

    printf("PASSED: infered types in %.2lf seconds\n", t2-t1);

    close_text_adapter(adapter);
}


void test_missing_values()
{
    uint64_t num_recs = 2500000;
    uint64_t num_fields = 5;

    FILE *input = fopen("./data/ints", "r");
    setvbuf(input, NULL, _IONBF, 0);
    
    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL, NULL);
    adapter->tokenize = &tokenizer;
    set_num_fields(adapter, num_fields);
    adapter->delim_char = ',';
    adapter->quote_char = '"';
    adapter->comment_char = '\0';

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(uint32_t), &uint_converter, NULL);
    }
   
    int i;
    for (i = 0; i < 5; i++)
        init_missing_values(adapter, i, 5);

    char temp1[2] = "NA";
    char temp2[3] = "NaN";
    char temp3[3] = "inf";
    char temp4[4] = "-inf";
    char temp5[4] = "None";
    
    for (i = 0; i < 5; i++)
    {
        add_missing_value(adapter, i, temp1, 2);
        add_missing_value(adapter, i, temp2, 3);
        add_missing_value(adapter, i, temp3, 3);
        add_missing_value(adapter, i, temp4, 4);
        add_missing_value(adapter, i, temp5, 4);
    }

    uint32_t *data = calloc(num_recs, sizeof(uint32_t)*num_fields);

    fseek(input, 0, SEEK_SET);

    /*double t1 = get_time(); */
    int result = read_records(adapter, num_recs, 1, (char *)data, NULL);
    /*double t2 = get_time(); */

    assert(result == ADAPTER_SUCCESS);
}


void test_line_tokenizer(uint64_t num_recs)
{
    uint64_t num_fields = 5;

    FILE *input = fopen("./data/ints", "r");
    setvbuf(input, NULL, _IONBF, 0);
    
    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL, NULL);
    adapter->tokenize = &record_tokenizer;
    set_num_fields(adapter, num_fields);

    fseek(input, 0, SEEK_SET);

    read_records(adapter, num_recs, 1, NULL, NULL);
}


int main()
{
    /* num recs should be the same number that was passed to generate.py */
    /* to generate datasets */
    uint64_t num_recs = 1000000;

    /*test_ints("./data/ints", num_recs);*/
    /*test_signed_ints("./data/signedints", 2500000, 5); */
    /*test_floats("./data/floats", num_recs);
    test_index("./data/ints", num_recs);
    test_gzip();
    test_gzip_index();
    test_step();
    test_mmap();
    test_regex("./data/ints", num_recs);
    test_fixed_width();*/
    /*test_type_inference(); */
    /*test_missing_values();*/
    test_line_tokenizer(num_recs);

    return 0;
}

