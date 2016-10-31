import sys
import pyodbc
import numpy as np
from time import time

# The number of rows in the table
N = 1000*1000

# To do a memory profile, run python interpreter with '-m memory_profiler'
mprofile = False
# Automagically detect if the memory_profiler library is loaded
for arg in sys.argv:
    if 'memory_profiler' in arg:
        mprofile = True

if mprofile:
    # This exposes another interesting way to profile (time based)
    from memory_profiler import memory_usage
else:
    # Assign the decorator to the identity
    profile = lambda x: x


def write_tables():
    import tables

    dtype = np.dtype("S7,f4,f4,f4,f4,i4")
    t0 = time()
    sarray = np.fromiter(((str(i), float(i), float(2*i), None, float(4*i), i)
                          for i in xrange(N)), dtype, count=N)
    t1 = time() - t0
    print "Created sarray with %d rows in %.3fs" % (N, t1)

    t0 = time()
    h5f = tables.openFile("market.h5", "w")
    table = h5f.createTable(h5f.root, "market", dtype)
    table.append(sarray)
    h5f.close()
    t1 = time() - t0
    print "[PyTables] Stored %d rows in %.3fs" % (N, t1)

def write_tables2():
    import tables

    dtype = np.dtype("S7,f4,f4,f4,f4,i4")
    # t0 = time()
    # sarray = np.fromiter(((str(i), float(i), float(2*i), None, float(4*i), i)
    #                       for i in xrange(N)), dtype, count=N)
    # t1 = time() - t0
    # print "Created sarray with %d rows in %.3fs" % (N, t1)

    t0 = time()
    h5f = tables.openFile("market.h5", "w")
    table = h5f.createTable(h5f.root, "market", dtype)
    count = 10000
    for j in xrange(count, N, count):
        sarray = np.fromiter(((str(i), float(i), float(2*i), None, float(4*i), i)
                              for i in xrange(j)), dtype)
        table.append(sarray)
    h5f.close()
    t1 = time() - t0
    print "[PyTables] Stored %d rows in %.3fs" % (N, t1)

def write_native():

    print "    *** sqlite3 driver (write native) ***"
    import sqlite3
    connection = sqlite3.connect('market.sqlite')
    cursor = connection.cursor()
    connection.isolation_level='DEFERRED'
    print "isolation_level:", connection.isolation_level
    cursor.execute("pragma synchronous = 0")
    # Query the synchronous state
    cursor.execute("pragma synchronous")
    print "pragma synchronous:", cursor.fetchone()

    try:
        cursor.execute('drop table market')
    except:
        pass

    cursor.execute('create table market '
                   '(symbol varchar(7),'
                   ' open float, low float, high float, close float,'
                   ' volume int)')
    # Add content to 'column' row by row
    t0 = time()
    for i in xrange(N):
        cursor.execute(
            "insert into market(symbol, open, low, high, close, volume)"
            " values (?, ?, ?, ?, ?, ?)",
            (str(i), float(i), float(2*i), None, float(4*i), i))
    #cursor.execute("commit")             # not supported by SQLite
    connection.commit()
    t1 = time() - t0
    print "Stored %d rows in %.3fs" % (N, t1)

def write(cursor):

    try:
        cursor.execute('drop table market')
    except:
        pass

    cursor.execute('create table market '
                   '(symbol varchar(7),'
                   ' open float, low float, high float, close float,'
                   ' volume int)')
    # Add content to 'column' row by row
    t0 = time()
    for i in xrange(N):
        cursor.execute(
            "insert into market(symbol, open, low, high, close, volume)"
            " values (?, ?, ?, ?, ?, ?)",
            (str(i), float(i), float(2*i), None, float(4*i), i))
    cursor.execute("commit")             # not supported by SQLite
    t1 = time() - t0
    print "Stored %d rows in %.3fs" % (N, t1)


@profile
def execute_query(cursor):
    if mprofile:
        mem_val = memory_usage((cursor.execute, ("select * from market",)))
        mem_max = max(mem_val)

    t0 = time()
    cursor.execute("select * from market")
    t1 = time() - t0
    print "[execute query] Query executed in %.3fs" % (t1,)
    if mprofile:
        print "Max memory consumption for executing query: %.1f MB" % mem_max

@profile
def get_native(cursor):
    print "\n    *** native driver (list) ***"
    execute_query(cursor)

    t0 = time()
    slist = cursor.fetchall()
    t1 = time() - t0
    print "[fetch as list] Fetch executed in %.3fs" % (t1,)

@profile
def get_list(cursor):
    print "\n    *** pyodbc driver (list) ***"
    execute_query(cursor)

    t0 = time()
    slist = cursor.fetchall()
    t1 = time() - t0
    print "[fetch as list] Fetch executed in %.3fs" % (t1,)

@profile
def get_darray(cursor):
    print "\n    *** pyodbc driver (dictarray) ***"
    execute_query(cursor)

    t0 = time()
    darray = cursor.fetchdictarray()
    t1 = time() - t0
    print "[fetch as dictarray] Fetch executed in %.3fs" % (t1,)


if __name__ == "__main__":

    # set up a cursor for native and odbc connections
    if "mysql" in sys.argv:
        connection = pyodbc.connect(
            'DSN=myodbc;UID=faltet;PWD=continuum;DATABASE=test')
        cursor = connection.cursor()
        print "    *** mysql native driver ***"
        import MySQLdb
        connection = MySQLdb.connect("localhost", "faltet", "continuum", "test")
        native_cursor = connection.cursor()
    elif "postgres" in sys.argv:
        connection = pyodbc.connect(
            'DSN=odbcpostgresql;UID=faltet;PWD=continuum;DATABASE=test')
        cursor = connection.cursor()
        print "    *** postgres native driver ***"
        import psycopg2
        connection = psycopg2.connect(user="faltet", password="continuum", database="test")
        native_cursor = connection.cursor()
    else:
        connection = pyodbc.connect(
            'DSN=odbcsqlite;DATABASE=market.sqlite')
        cursor = connection.cursor()
        print "    *** sqlite3 native driver ***"
        import sqlite3
        connection = sqlite3.connect('market.sqlite')
        native_cursor = connection.cursor()


    if "write" in sys.argv:
        print "writing"
        if mprofile:
            #mem_val = memory_usage((write_tables, ()))
            #mem_max = max(mem_val)
            #print "Max memory consumption for pytables: %.1f MB" % mem_max
            mem_val = memory_usage((write, (cursor,)))
            mem_max = max(mem_val)
            print "Max memory consumption for write: %.1f MB" % mem_max
        else:
            #write_tables()
            #write_sqlite_native()
            #write_native(native_cursor)
            write(cursor)

    if mprofile:
        mem_val = memory_usage((get_native, (native_cursor,)))
        mem_max = max(mem_val)
        print "Max memory consumption for list (native): %.1f MB" % mem_max
        mem_val = memory_usage((get_list, (cursor,)))
        mem_max = max(mem_val)
        print "Max memory consumption for list: %.1f MB" % mem_max
        mem_val = memory_usage((get_darray, (cursor,)))
        mem_max = max(mem_val)
        print "Max memory consumption for dictarray: %.1f MB" % mem_max
    else:
        get_native(native_cursor)
        get_list(cursor)
        get_darray(cursor)

    connection.close()

