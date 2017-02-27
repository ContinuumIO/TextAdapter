#include "mongo_adapter.h"


int main()
{
    MongoAdapter *adapter = open_mongo_adapter("127.0.0.1", 27017, "MongoAdapter_tests", "test_ints");
    set_converter(adapter->fields, 0, "field0", sizeof(uint64_t), &mongo2int_converter, NULL);
    set_converter(adapter->fields, 0, "field1", sizeof(uint64_t), &mongo2int_converter, NULL);
    set_converter(adapter->fields, 0, "field2", sizeof(uint64_t), &mongo2int_converter, NULL);
    set_converter(adapter->fields, 0, "field3", sizeof(uint64_t), &mongo2int_converter, NULL);
    set_converter(adapter->fields, 0, "field4", sizeof(uint64_t), &mongo2int_converter, NULL);

    uint64_t *output = calloc(10000000, sizeof(uint64_t) * 5);
    uint64_t num_recs_read = 0;
    MongoAdapterError error = read_records(adapter, 0, 10000000, 1, (void *)output, &num_recs_read);
    printf("JNB: num_recs_read=%llu error=%d\n", num_recs_read, error);
    close_mongo_adapter(adapter);

    return 0;
}
