#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "../textadapter/text_adapter.h"
#include "../textadapter/io_functions.h"


int main()
{
    uint64_t num_fields = 5;

    FILE *input = fopen("./data/ints", "r");
    setvbuf(input, NULL, _IONBF, 0);
    
    TextAdapter *adapter = open_text_adapter((void *)input, NULL, &read_file, NULL, &seek_file, NULL);
    adapter->tokenize = &delim_tokenizer;
    set_num_fields(adapter, num_fields);
    adapter->delim_char = ',';
    adapter->quote_char = '\0';
    adapter->comment_char = '\0';

    int c;
    for (c = 0; c < num_fields; c++)
    {
        set_converter(adapter, c, sizeof(uint32_t), &uint_converter, NULL);
    }

    uint32_t *data = calloc(10000000, sizeof(uint32_t)*num_fields);

    fseek(input, 0, SEEK_SET);

    clock_t t0 = clock();
    uint64_t recs_read = 0;
    int result = read_records(adapter, 10000000, 1, (char *)data, &recs_read);
    clock_t t1 = clock();

    assert(result == ADAPTER_SUCCESS);

    printf("PASSED: read %llu records in %.2lf seconds\n", recs_read, (double)(t1-t0) / (double)CLOCKS_PER_SEC);

    free(data);
    close_text_adapter(adapter);
}
