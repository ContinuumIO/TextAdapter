from __future__ import print_function

# Script written to help reproduce a crash
# it can be run in a way that allows checking large databases that also contain nulls.




import iopro.pyodbc as pyodbc
import sys, getopt
import random
import string
import datetime
import numpy as np

_connect_string = "DRIVER=/usr/local/lib/psqlodbcw.so;SERVER=localhost;DATABASE=iopro_pyodbc_test"


_test_table_name = 'test_table'

_insert_command = """
INSERT INTO %s VALUES (?, ?)
""" % _test_table_name

_select_command = """
SELECT * FROM %s LIMIT ?
""" % _test_table_name

def random_string(chars=string.letters + string.digits + ' ', maxlen=42):
    return "".join(random.choice(chars) for x in range(random.randint(0,maxlen)))

tables = (
    {
        "name": "numeric_tests",
        "type": "decimal(8,4)",
        "descr": "Table for test on numeric types",
        "generator": lambda x: [[random.uniform(-500,500)] for i in xrange(x)]
    },
    {
        "name": "char_tests",
        "type": "varchar(16)",
        "descr": "Table for test on character types",
        "generator": lambda x: [[random_string(maxlen=16)] for i in xrange(x)]
    },
    {
        "name": "datetime_tests",
        "type": "timestamp",
        "descr": "Table for test on datetimes",
        "generator": lambda x: [[datetime.datetime.now() + datetime.timedelta(seconds=random.randint(-3600, 3600))] for i in xrange(x)]
    }
)
    

def verbose_exec(cursor, command, *args, **kwargs):
    print(command)
    cursor.execute(command, *args, **kwargs)

def generate_tables(count):
    import os, binascii
    import random
    import datetime

    print("Generating tables for tests (%s elements)" % repr(count))

    conn = pyodbc.connect(_connect_string)
    cur = conn.cursor()

    for tab in tables:
        print("Table %s: %s" % (tab["name"], tab["descr"]))
        verbose_exec(cur, "drop table if exists %s" % (tab["name"]))
        verbose_exec(cur, "create table %s (val %s)" %(tab["name"], tab["type"]))
        values = tab["generator"](count/2)
        values.extend([(None,)] * (count - len(values))) # add nulls
        random.shuffle(values) #and shuffle
        cur.executemany("insert into %s values(?)" % (tab["name"],) , values)

    cur.commit()
    conn.close()

def exit_help(val):
    print('''
%s <options> <tests...>:

\tAvailable Options:

\t-h \t print this help
\t-c --create <rows>\tgenerate the table with <rows> rows

\t%s
''' % (sys.argv[0], '\n\t'.join(_experiments.keys())))
    sys.exit(val)



def read_only_connect():
    return pyodbc.connect(_connect_string,
                          ansi=True,
                          unicode_results=False,
                          readonly=True)

def run_query_fetchsarray_size(count):
    conn = read_only_connect()
    cur = conn.cursor()
    cur.execute(_select_command, count)
    result = cur.fetchsarray(int(count))
    del(cur)
    del(conn)
    return result

def run_query(table, sqltype, func, count):
    conn = read_only_connect()
    cur = conn.cursor()
    cur.execute("select cast(val as %s) from %s limit %s" % 
                (sqltype, table, 'all' if count is None or int(count) < 0 else repr(count)))
    res = func(cur, count)
    del(cur)
    del(conn)
    return res


# supported sqltypes and the table they will use
_numeric_types = set(["smallint", "int", "integer", "bigint", "float(24)", 
                   "float(53)", "real", "double precision", "decimal(8,4)"])
_character_types = set() #{ "char(16)" }
_datetime_types = set(["timestamp", "time", "date"])
_all_types = _numeric_types | _character_types | _datetime_types

_experiments = {
    "fetchall": lambda x, c: x.fetchall(),
    "fetchdictarray": lambda x,c: x.fetchdictarray() if c is None or int(c) < 0 else x.fetchdictarray(int(c)),
    "fetchsarray": lambda x,c: x.fetchsarray() if c is None or int(c) < 0 else x.fetchsarray(int(c))
}

def test(types, experiments, n):
    for t in types:
        table = None
        if t in _numeric_types:
            table = 'numeric_tests'
        elif t in _character_types:
            table = 'char_tests'
        elif t in _datetime_types:
            table = 'datetime_tests'
        
        if table is not None:
            for e in experiments:
                try:
                    print("Running experiment %s for type %s; count is %s" % (e, t, repr(n)))
                    func = _experiments[e]
                    res = run_query(table, t, func, n)

                    if isinstance(res, dict):
                        print("dtype returned is %s" % repr(res["val"].dtype))
                    elif isinstance(res, np.ndarray):
                        print("dtype returned is %s" % repr(res.dtype))
                except KeyError:
                    print("don't know how to run '%s' for type '%s'" % (e, t))
        else:
            print("unknown type '%s'", t)
            

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hcgn:t:v", ["create", "count=", "type=", "verbose", "use_guards"])
    except getopt.GetoptError:
        exit_help(2)

    # these are the supported experiments

    type_ = None
    n = None
    command = None
    create = False
    trace = False
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            exit_help(0)
        elif opt in("-c", "--create"):
            create = True
        elif opt in("-n", "--count"):
            n = arg
        elif opt in("-t", "--type"):
            type_ = arg
        elif opt in("-v", "--verbose"):
            trace = True
        elif opt in("-g", "--use_guards"):
            guards = True

    try:
        pyodbc.enable_tracing(trace)
    except:
        if (trace):
            print("it seems your IOPro does not support tracing in pyodbc")

    try:
        pyodbc.enable_mem_guards(guards)
    except:
        if (trace):
            print("it seems your IOPro does not mem_guards in pyodbc")

    # defaults
    types = _all_types if type_ is None else (type_, )

    if (create):
        count = int(n) if n is not None else 100000
        generate_tables(count)

    if len(args) == 0 and not create:
        exit_help(0)

    test(types, args, n)


if __name__ == '__main__':
    main(sys.argv[1:])
    
    
