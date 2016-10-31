#include <libpq/libpq-fe.h>
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

typedef struct
{
    uint32_t value1;
    double value2;
} Data;

int main(int argc, char *argv[])
{
    PGconn *conn = PQconnectdb("postgresql://localhost:5432/foo");
    if (PQstatus(conn) != CONNECTION_OK) {
        fprintf(stderr, "Connection failed\n");
        PQfinish(conn);
        exit(1);
    }

    const char *select_cmd = "SELECT value1, value2 FROM foo1 limit 10000000";
	PGresult *res = PQexecParams(conn, select_cmd, 0, NULL, NULL, NULL, NULL, 1);

	if (PQresultStatus(res) != PGRES_TUPLES_OK) {
		fprintf(stderr, "Select failed: %s\n", PQerrorMessage(conn));
		PQclear(res);
        PQfinish(conn);
        exit(1);
	}

    Data *data = calloc(10000000, sizeof(Data));
    int field1 = PQfnumber(res, "value1");
    int field2 = PQfnumber(res, "value2");
	for (int i = 0; i < PQntuples(res); i++) {
        double value2;
        char *ptr = PQgetvalue(res, i, field2);
        char *ptr2 = (char *)&value2;
        ptr2[0] = ptr[7];
        ptr2[1] = ptr[6];
        ptr2[2] = ptr[5];
        ptr2[3] = ptr[4];
        ptr2[4] = ptr[3];
        ptr2[5] = ptr[2];
        ptr2[6] = ptr[1];
        ptr2[7] = ptr[0];
        //printf("%u %f\n", ntohl((uint32_t*)PQgetvalue(res, i, field1)), value2);
        data[i].value1 = ntohl((uint32_t*)PQgetvalue(res, i, field1));
        data[i].value2 = value2;
    }
	PQclear(res);

    return 0;
}
