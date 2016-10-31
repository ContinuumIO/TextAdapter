"""This script serves as a test to see if sqlcancel works as expected.
TODO: Make this into a proper test"""

import iopro.pyodbc as odbc
from time import time, sleep
import threading

_connection_string = 'DSN=SQLServerTest'
_table_name = 'TEST_TABLE'

def ensure_table(conn):
    cursor = conn.cursor()
    if cursor.tables(table=_table_name).fetchone():
        print 'skipping creation, table exists'
        return 

    cursor.execute('CREATE TABLE %s (field1 TEXT, field2 TEXT)' % 
                   _table_name)

    for i in range(10000):
        cursor.execute('INSERT INTO %s(field1, field2) values (?, ?)' 
                       % _table_name, str(i), str(i*2))

    cursor.commit()

def drop_table(conn):
    cursor = conn.cursor()
    cursor.execute('DROP TABLE %s' % _table_name)
    cursor.commit()

def query(conn):
    cursor = conn.cursor()
    select_str = """
SELECT a.field1, b.field2
FROM 
    %s AS a, %s AS b
WHERE 
    a.field2 LIKE b.field1""" % (_table_name, _table_name)

    print select_str
    cursor.execute(select_str)

    result = cursor.fetchall()
    if len(result) > 40:
        print ('%s ... %s' % (str(result[:20]), str(result[-20:])))
    else:
        print (result)



def query_with_time_out(conn, to):
    def watchdog(cursor, time_out):
        print ('started thread')
        while time_out > 0.0:
            print '.'
            wait_time = min(time_out, 1.0)
            sleep(wait_time)
            time_out -= wait_time

        print ('issuing cancel')
        cursor.cancel()

    cursor = conn.cursor()

    select_str = """
SELECT a.field1, b.field2
FROM 
    %s AS a, %s AS b
WHERE 
    a.field2 LIKE b.field1""" % (_table_name, _table_name)

    print select_str

    t = threading.Thread(target=watchdog, args=(cursor, to))
    t.start()
    try:
        cursor.execute(select_str)

        result = cursor.fetchall()
    except odbc.Error:
        result = 'timed out'

    if len(result) > 40:
        print ('%s ... %s' % (str(result[:20]), str(result[-20:])))
    else:
        print (result)


def main(conn_str):
    print ('Connecting to data source...')
    conn = odbc.connect(conn_str)

    print ('Building the table...')
    ensure_table(conn)

    print ('Trying queries...')
    t1 = time()
    query_with_time_out(conn, 5.0)
    t2 = time()
    query(conn)
    t3 = time()
    print ('query ellapsed %d s, query_with_timeout ellapsed %d s' % 
           (t3-t2, t2-t1))


def clean(conn_str): 
    print ('Connecting to data source...')
    conn = odbc.connect(conn_str)
    print ('Dropping the table')
    drop_table(conn)


if __name__ == '__main__':
    main(_connection_string)
