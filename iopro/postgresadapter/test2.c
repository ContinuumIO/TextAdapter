#include "postgres_adapter.h"
#include <stdio.h>
#include <stdint.h>

int main(int argc, char *argv[])
{
    const char *uri = "postgresql://localhost:5432/foo";
    const char *query = "select value1, value2 from foo1";
    PostgresAdapter *adapter = open_postgres_adapter(uri);
    if (adapter == NULL) {
        fprintf(stderr, "ERROR: could not open postgresql adapter\n");
        exit(1);
    }
    exec_query(adapter, query);

    int num_fields = get_num_fields(adapter);
    printf("num_fields: %d\n", num_fields);

    char *output = calloc(get_num_records(adapter), get_record_size(adapter));
    int num_records_found = 0;
    read_records(adapter, output, &num_records_found);
    printf("num_records_found: %d\n", num_records_found);

    char **field_names = calloc(num_fields, sizeof(char*));
    get_field_names(adapter, field_names);
    printf("field names: %s %s\n", field_names[0], field_names[1]);

    Oid *field_types = calloc(num_fields, sizeof(Oid));
    get_field_types(adapter, field_types);
    printf("field types: %d %d\n", field_types[0], field_types[1]);

    char *output_ptr = output;
    for (int i = 0; i < num_records_found; i++) {
        uint32_t value1;
        double value2;
        memcpy((char *)&value1, output_ptr, sizeof(value1));
        memcpy((char *)&value2, output_ptr + sizeof(value1), sizeof(value2));
        printf("field values: %u %f\n", value1, value2);
        output_ptr += get_record_size(adapter);
    }
    
    free(field_names);
    free(field_types);
    free(output);
    clear_query(adapter);
    close_postgres_adapter(adapter);

    return 0;
}
