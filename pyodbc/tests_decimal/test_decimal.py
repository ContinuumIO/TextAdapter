from __future__ import print_function

"""This script serves as a test to see if the decimal and numeric
types work as expected"""

import iopro.pyodbc as odbc

_connection_string = 'DSN=SQLServerTest'
_table_name = 'TEST_NUMERIC'


def _report_exec(cursor, sqlcommand, *args, **kwargs):
    _str = 'executing sql:'
    print('\n\n%s\n%s\n%s' % (_str, '-'*len(_str), sqlcommand))
    cursor.execute(sqlcommand, *args, **kwargs)


def ensure_table(conn):
    cursor = conn.cursor()
    if cursor.tables(table=_table_name).fetchone():
        print('skipping creation, table exists')
        return
    
    create_str = ('CREATE TABLE %s (\n'
                  '    field_num NUMERIC(16,9),\n'
                  '    field_dec DECIMAL(16,9)\n'
                  ')\n' % _table_name)
    insert_str = ('INSERT INTO %s (field_num, field_dec)\n'
                  '    VALUES (?, ?)' % _table_name) 

    _report_exec(cursor, create_str)
    _report_exec(cursor, insert_str, '42.00', '32.42456')

    cursor.commit()


def drop_table(conn):
    cursor = conn.cursor()
    _report_exec(cursor, 'DROP TABLE %s' % _table_name)
    cursor.commit()

def query(conn):
    cursor = conn.cursor()
    select_str = 'SELECT field_num, field_dec FROM %s' % _table_name
    _report_exec(cursor, select_str)

    result = cursor.fetchone()
    print(result)

def connect(conn_str):
    print('Connecting to data source...')
    return odbc.connect(conn_str)
    

def main(conn_str):
    conn = connect(conn_str)
    ensure_table(conn)
    query(conn)
    drop_table(conn)

def clean(conn_str):
    conn = connect(conn_str)
    drop_table(conn)

if __name__ == '__main__':
    main(_connection_string)
