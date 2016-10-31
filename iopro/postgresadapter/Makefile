CC = gcc
CFLAGS = -I$(PGDEV)/src/interfaces/ -I$(PGDEV)/src/include/
LDFLAGS = -L$(PGDEV)/src/interfaces/libpq -lpq

all: test1 test2

test1: test1.o
	$(CC) -o test1 $(LDFLAGS) test1.o

test1.o: test1.c
	$(CC) -c $(CFLAGS) test1.c

test2: test2.o postgres_adapter.o
	$(CC) -o test2 $(LDFLAGS) test2.o postgres_adapter.o

test2.o: test2.c
	$(CC) -c $(CFLAGS) test2.c

postgres_adapter.o: postgres_adapter.c
	$(CC) -c $(CFLAGS) postgres_adapter.c

clean:
	rm -rf *.o
	rm -rf test1
	rm -rf test2
