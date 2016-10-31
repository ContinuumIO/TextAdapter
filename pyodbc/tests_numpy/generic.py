#!/usr/bin/python
# -*- coding: latin-1 -*-

usage = """\
usage: %prog [options] backend

Unit tests for ODBC access to different databases backends (SQLite,
MySQL and PostgreSQL are supported).

To use, pass a backend string, like 'sqlite' (default), 'mysql' and
'postgresql' as the parameter. The tests will create and drop tables
t1 and t2 as necessary.

You must also configure the backend into a setup.cfg file in the root
of the project (the same one setup.py would use) like so:

  [sqlite]
  connection-string=DSN=odbcsqlite;Database=test-sqlite.db
  [mysql]
  connection-string=DSN=myodbc;Database=test
"""

import sys, os, re
import unittest
from decimal import Decimal
from datetime import datetime, date, time
from os.path import join, getsize, dirname, abspath
from testutils import *
import numpy as np
from datetime import datetime

_TESTSTR = '0123456789-abcdefghijklmnopqrstuvwxyz-'

# A global variable with the DB backend
backend = 'sqlite'  # Default backend


class GenericTestCase(unittest.TestCase):

    def __init__(self, method_name, connection_string):
        unittest.TestCase.__init__(self, method_name)
        self.connection_string = connection_string

    def setUp(self):
        self.cnxn   = pyodbc.connect(self.connection_string)
        self.cursor = self.cnxn.cursor()

        for i in range(3):
            try:
                self.cursor.execute("drop table t%d" % i)
                self.cnxn.commit()
            except:
                pass

        self.cnxn.rollback()

    def tearDown(self):
        try:
            self.cursor.close()
            self.cnxn.close()
        except:
            # If we've already closed the cursor or connection,
            # exceptions are thrown.
            pass

    #
    # ints and floats
    #

    def test_tinyint(self):
        if backend == 'postgresql':
            # PostgreSQL does not support the tiny integer
            return
        value = -46
        self.cursor.execute("create table t1(n tinyint)")
        self.cursor.execute("insert into t1 values (?)", value)
        result = self.cursor.execute("select n from t1").fetchone()[0]
        self.assertEqual(result, value)
        result = self.cursor.execute("select n from t1").fetchsarray()[0][0]
        self.assertEqual(result.dtype, np.int8)
        self.assertEqual(result, value)
        result = self.cursor.execute("select n from t1").fetchdictarray()['n'][0]
        self.assertEqual(result, value)
        self.assertEqual(result.dtype, np.int8)

    def test_smallint(self):
        value = 1234
        self.cursor.execute("create table t1(n smallint)")
        self.cursor.execute("insert into t1 values (?)", value)
        result = self.cursor.execute("select n from t1").fetchone()[0]
        self.assertEqual(result, value)
        result = self.cursor.execute("select n from t1").fetchsarray()[0][0]
        self.assertEqual(result, value)
        self.assertEqual(result.dtype, np.int16)
        result = self.cursor.execute("select n from t1").fetchdictarray()['n'][0]
        self.assertEqual(result, value)
        self.assertEqual(result.dtype, np.int16)

    def test_int(self):
        value = 123456
        self.cursor.execute("create table t1(n int)")
        self.cursor.execute("insert into t1 values (?)", value)
        result = self.cursor.execute("select n from t1").fetchone()[0]
        self.assertEqual(result, value)
        result = self.cursor.execute("select n from t1").fetchsarray()[0][0]
        self.assertEqual(result, value)
        self.assertEqual(result.dtype, np.int32)
        result = self.cursor.execute("select n from t1").fetchdictarray()['n'][0]
        self.assertEqual(result, value)
        self.assertEqual(result.dtype, np.int32)

    def test_negative_int(self):
        value = -123456
        self.cursor.execute("create table t1(n int)")
        self.cursor.execute("insert into t1 values (?)", value)
        result = self.cursor.execute("select n from t1").fetchone()[0]
        self.assertEqual(result, value)
        result = self.cursor.execute("select n from t1").fetchsarray()[0][0]
        self.assertEqual(result, value)
        self.assertEqual(result.dtype, np.int32)
        result = self.cursor.execute("select n from t1").fetchdictarray()['n'][0]
        self.assertEqual(result, value)
        self.assertEqual(result.dtype, np.int32)

    def test_bigint(self):
        input = 3000000000
        self.cursor.execute("create table t1(d bigint)")
        self.cursor.execute("insert into t1 values (?)", input)
        result = self.cursor.execute("select d from t1").fetchone()[0]
        self.assertEqual(result, input)
        result = self.cursor.execute("select d from t1").fetchsarray()[0][0]
        self.assertEqual(result, input)
        self.assertEqual(result.dtype, np.int64)
        result = self.cursor.execute("select d from t1").fetchdictarray()['d'][0]
        self.assertEqual(result, input)
        self.assertEqual(result.dtype, np.int64)

    def test_negative_bigint(self):
        input = -430000000
        self.cursor.execute("create table t1(d bigint)")
        self.cursor.execute("insert into t1 values (?)", input)
        result = self.cursor.execute("select d from t1").fetchone()[0]
        self.assertEqual(result, input)
        result = self.cursor.execute("select d from t1").fetchsarray()[0][0]
        self.assertEqual(result, input)
        self.assertEqual(result.dtype, np.int64)
        result = self.cursor.execute("select d from t1").fetchdictarray()['d'][0]
        self.assertEqual(result, input)
        self.assertEqual(result.dtype, np.int64)

    def test_float(self):
        value = 1234.567
        if backend == 'postgresql':
            self.cursor.execute("create table t1(n real)")
        else:
            self.cursor.execute("create table t1(n float)")
        self.cursor.execute("insert into t1 values (?)", value)
        result = self.cursor.execute("select n from t1").fetchone()[0]
        self.assertAlmostEqual(result, value, places=2)
        result = self.cursor.execute("select n from t1").fetchsarray()[0][0]
        if backend == 'sqlite':
            # SQLite only supports float64
            self.assertEqual(result.dtype, np.float64)
        else:
            self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(result, value, places=2)
        result = self.cursor.execute("select n from t1").fetchdictarray()['n'][0]
        self.assertAlmostEqual(result, value, places=2)
        if backend == 'sqlite':
            self.assertEqual(result.dtype, np.float64)
        else:
            self.assertEqual(result.dtype, np.float32)

    def test_negative_float(self):
        value = -200
        self.cursor.execute("create table t1(n float)")
        self.cursor.execute("insert into t1 values (?)", value)
        result  = self.cursor.execute("select n from t1").fetchone()[0]
        self.assertEqual(value, result)
        result = self.cursor.execute("select n from t1").fetchsarray()[0][0]
        self.assertEqual(result, value)
        result = self.cursor.execute("select n from t1").fetchdictarray()['n'][0]
        self.assertEqual(result, value)

    def test_double(self):
        value = 1234.5678901
        self.cursor.execute("create table t1(n double precision)")
        self.cursor.execute("insert into t1 values (?)", value)
        result = self.cursor.execute("select n from t1").fetchone()[0]
        self.assertAlmostEqual(result, value, places=2)
        result = self.cursor.execute("select n from t1").fetchsarray()[0][0]
        self.assertEqual(result.dtype, np.float64)
        self.assertAlmostEqual(result, value, places=2)
        result = self.cursor.execute("select n from t1").fetchdictarray()['n'][0]
        self.assertAlmostEqual(result, value, places=2)
        self.assertEqual(result.dtype, np.float64)

    #
    # rowcount
    #

    def test_rowcount_select(self):
        """
        Ensure Cursor.rowcount is set properly after a select statement.
        """
        self.cursor.execute("create table t1(i int)")
        count = 4
        for i in range(count):
            self.cursor.execute("insert into t1 values (?)", i)

        self.cursor.execute("select * from t1")
        rows = self.cursor.fetchall()
        self.assertEqual(len(rows), count)

        self.cursor.execute("select * from t1")
        rows_sarray = self.cursor.fetchsarray()
        self.assertEqual(len(rows_sarray), count)

        self.cursor.execute("select * from t1")
        rows_dictarray = self.cursor.fetchdictarray()
        self.assertEqual(len(rows_dictarray['i']), count)

    #
    # date/time
    #

    def test_timestamp_dictarray(self):
        if np.__version__ < "1.7": return
        self.cursor.execute("create table t1(a int, b timestamp, c int)")

        dates = [datetime.strptime("2008-04-%02d 00:01:02"%i, "%Y-%m-%d %H:%M:%S")
                 for i in range(1,11)]

        params = [ (i, dates[i-1], i) for i in range(1,11) ]
        npparams = [ (i, np.datetime64(dates[i-1]), i) for i in range(1,11) ]

        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)", params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()
        for param, row in zip(npparams, zip(rows['a'], rows['b'], rows['c'])):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1], row[1])
            self.assertEqual(param[2], row[2])

    def test_timestamp_sarray(self):
        if np.__version__ < "1.7": return
        self.cursor.execute("create table t1(a int, b timestamp, c int)")

        dates = [datetime.strptime("2008-04-%02d 00:01:02"%i, "%Y-%m-%d %H:%M:%S")
                 for i in range(1,11)]

        params = [ (i, dates[i-1], i) for i in range(1,11) ]
        npparams = [ (i, np.datetime64(dates[i-1]), i) for i in range(1,11) ]

        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)", params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchsarray()
        for param, row in zip(npparams, rows):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1], row[1])
            self.assertEqual(param[2], row[2])

    def test_date_dictarray(self):
        if np.__version__ < "1.7": return
        self.cursor.execute("create table t1(a int, b date, c int)")

        dates = [date(2008, 4, i) for i in range(1,11)]
        npdates = np.array(dates, dtype='datetime64[D]')

        params = [ (i, dates[i-1], i) for i in range(1,11) ]
        npparams = [ (i, npdates[i-1], i) for i in range(1,11) ]

        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)", params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()
        for param, row in zip(npparams, zip(rows['a'], rows['b'], rows['c'])):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1], row[1])
            self.assertEqual(param[2], row[2])

    def test_date_sarray(self):
        if np.__version__ < "1.7": return
        self.cursor.execute("create table t1(a int, b date, c int)")

        dates = [date(2008, 4, i) for i in range(1,11)]
        npdates = np.array(dates, dtype='datetime64[D]')

        params = [ (i, dates[i-1], i) for i in range(1,11) ]
        npparams = [ (i, npdates[i-1], i) for i in range(1,11) ]

        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)", params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchsarray()
        for param, row in zip(npparams, rows):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1], row[1])
            self.assertEqual(param[2], row[2])

    def test_time_dictarray(self):
        if np.__version__ < "1.7": return
        self.cursor.execute("create table t1(a int, b time, c int)")
        N = 60
        times = range(N)
        nptimes = np.array(times, dtype='timedelta64[s]')

        params = [ (i, time(0, 0, times[i]), i) for i in range(N) ]
        npparams = [ (i, nptimes[i], i) for i in range(N) ]

        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)", params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()
        for param, row in zip(npparams, zip(rows['a'], rows['b'], rows['c'])):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1], row[1])
            self.assertEqual(param[2], row[2])

    def test_time_sarray(self):
        if np.__version__ < "1.7": return
        self.cursor.execute("create table t1(a int, b time, c int)")
        N = 60
        times = range(N)
        nptimes = np.array(times, dtype='timedelta64[s]')

        params = [ (i, time(0, 0, times[i]), i) for i in range(N) ]
        npparams = [ (i, nptimes[i], i) for i in range(N) ]

        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)", params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchsarray()
        for param, row in zip(npparams, rows):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1], row[1])
            self.assertEqual(param[2], row[2])

    #
    # NULL values (particular values)
    #

    def test_null_tinyints(self):
        if backend == 'postgresql':
            # PostgreSQL does not support the tiny integer
            return
        ints = [i for i in range(1,3)]

        self.cursor.execute("create table t1(a int, b tinyint, c tinyint)")
        params = [(0, ints[0], 0),
                  (1, ints[1], None),
                  (2, None, 2),
                  (3, None, None),
                  ]
        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)",
                                params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()
        
        # row 0
        self.assertEqual(rows['b'][0], ints[0])
        self.assertEqual(rows['c'][0], 0)
        # row 1
        self.assertEqual(rows['b'][1], ints[1])
        self.assertEqual(rows['c'][1], -2**7)
        # row 2
        self.assertEqual(rows['b'][2], -2**7)
        self.assertEqual(rows['c'][2], 2)
        # row 3
        self.assertEqual(rows['b'][3], -2**7)
        self.assertEqual(rows['c'][3], -2**7)

    def test_null_smallints(self):
        ints = [i for i in range(1,3)]

        self.cursor.execute("create table t1(a int, b smallint, c smallint)")
        params = [(0, ints[0], 0),
                  (1, ints[1], None),
                  (2, None, 2),
                  (3, None, None),
                  ]
        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)",
                                params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()
        
        # row 0
        self.assertEqual(rows['b'][0], ints[0])
        self.assertEqual(rows['c'][0], 0)
        # row 1
        self.assertEqual(rows['b'][1], ints[1])
        self.assertEqual(rows['c'][1], -2**15)
        # row 2
        self.assertEqual(rows['b'][2], -2**15)
        self.assertEqual(rows['c'][2], 2)
        # row 3
        self.assertEqual(rows['b'][3], -2**15)
        self.assertEqual(rows['c'][3], -2**15)

    def test_null_ints(self):
        ints = [i for i in range(1,3)]

        self.cursor.execute("create table t1(a int, b int, c int)")
        params = [(0, ints[0], 0),
                  (1, ints[1], None),
                  (2, None, 2),
                  (3, None, None),
                  ]
        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)",
                                params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()

        # row 0
        self.assertEqual(rows['b'][0], ints[0])
        self.assertEqual(rows['c'][0], 0)
        # row 1
        self.assertEqual(rows['b'][1], ints[1])
        self.assertEqual(rows['c'][1], -2**31)
        # row 2
        self.assertEqual(rows['b'][2], -2**31)
        self.assertEqual(rows['c'][2], 2)
        # row 3
        self.assertEqual(rows['b'][3], -2**31)
        self.assertEqual(rows['c'][3], -2**31)

    def test_null_bigints(self):
        ints = [i for i in range(1,3)]

        self.cursor.execute("create table t1(a int, b bigint, c bigint)")
        params = [(0, ints[0], 0),
                  (1, ints[1], None),
                  (2, None, 2),
                  (3, None, None),
                  ]
        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)",
                                params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()

        # row 0
        self.assertEqual(rows['b'][0], ints[0])
        self.assertEqual(rows['c'][0], 0)
        # row 1
        self.assertEqual(rows['b'][1], ints[1])
        self.assertEqual(rows['c'][1], -2**63)
        # row 2
        self.assertEqual(rows['b'][2], -2**63)
        self.assertEqual(rows['c'][2], 2)
        # row 3
        self.assertEqual(rows['b'][3], -2**63)
        self.assertEqual(rows['c'][3], -2**63)

    def test_null_floats(self):
        floats = [float(i) for i in range(1,3)]

        self.cursor.execute("create table t1(a int, b float, c float)")
        params = [(0, floats[0], 0),
                  (1, floats[1], None),
                  (2, None, 2),
                  (3, None, None),
                  ]
        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)",
                                params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()

        # row 0
        self.assertEqual(rows['b'][0], floats[0])
        self.assertEqual(rows['c'][0], 0)
        # row 1
        self.assertEqual(rows['b'][1], floats[1])
        # The line below does not work without the str() representation
        self.assertEqual(str(rows['c'][1]), str(np.nan))
        # row 2
        # The line below does not work without the str() representation
        self.assertEqual(str(rows['b'][2]), str(np.nan))
        self.assertEqual(rows['c'][2], 2)
        # row 3
        # The line below does not work without the str() representation
        self.assertEqual(str(rows['b'][3]), str(np.nan))
        self.assertEqual(str(rows['c'][3]), str(np.nan))

    def test_null_doubles(self):
        floats = [float(i) for i in range(1,3)]

        self.cursor.execute("create table t1(a int, b float8, c float8)")
        params = [(0, floats[0], 0),
                  (1, floats[1], None),
                  (2, None, 2),
                  (3, None, None),
                  ]
        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)",
                                params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()

        # row 0
        self.assertEqual(rows['b'][0], floats[0])
        self.assertEqual(rows['c'][0], 0)
        # row 1
        self.assertEqual(rows['b'][1], floats[1])
        # The line below does not work without the str() representation
        self.assertEqual(str(rows['c'][1]), str(np.nan))
        # row 2
        # The line below does not work without the str() representation
        self.assertEqual(str(rows['b'][2]), str(np.nan))
        self.assertEqual(rows['c'][2], 2)
        # row 3
        # The line below does not work without the str() representation
        self.assertEqual(str(rows['b'][3]), str(np.nan))
        self.assertEqual(str(rows['c'][3]), str(np.nan))

    def test_null_strings(self):
        strings = [str(i) for i in range(1,3)]

        self.cursor.execute("create table t1(a int, b varchar(2), c varchar(2))")
        params = [(0, strings[0], ''),
                  (1, strings[1], None),
                  (2, None, '2'),
                  (3, None, None),
                  ]
        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)",
                                params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()

        # row 0
        self.assertEqual(rows['b'][0], strings[0])
        self.assertEqual(rows['c'][0], '')
        # row 1
        self.assertEqual(rows['b'][1], strings[1])
        self.assertEqual(rows['c'][1], 'NA')
        # row 2
        self.assertEqual(rows['b'][2], 'NA')
        self.assertEqual(rows['c'][2], '2')
        # row 3
        # The line below does not work without the str() representation
        self.assertEqual(rows['b'][3], 'NA')
        self.assertEqual(rows['c'][3], 'NA')

    def test_null_timestamp(self):
        if np.__version__ < "1.7": return
        dates = [datetime.strptime("2008-04-%02d 00:01:02"%i,
                                   "%Y-%m-%d %H:%M:%S")
                 for i in range(1,3)]

        self.cursor.execute("create table t1(a int, b timestamp, c int)")
        params = [(0, dates[0], 0),
                  (1, dates[1], None),
                  (2, None, 2),
                  (3, None, None),
                  ]
        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)",
                                params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()

        # row 0
        self.assertEqual(rows['b'][0], np.datetime64(dates[0]))
        self.assertEqual(rows['c'][0], 0)
        # row 1
        self.assertEqual(rows['b'][1], np.datetime64(dates[1]))
        self.assertEqual(rows['c'][1], -2147483648)
        # row 2
        # The line below does not work without the str() representation
        self.assertEqual(str(rows['b'][2]), str(np.datetime64('NaT')))
        self.assertEqual(rows['c'][2], 2)
        # row 3
        self.assertEqual(str(rows['b'][3]), str(np.datetime64('NaT')))
        self.assertEqual(rows['c'][3], -2147483648)

    def test_null_date(self):
        if np.__version__ < "1.7": return
        dates = [date(2008, 4, i) for i in range(1,3)]
        npdates = np.array(dates, dtype="datetime64[D]")

        self.cursor.execute("create table t1(a int, b date, c int)")
        params = [(0, dates[0], 0),
                  (1, dates[1], None),
                  (2, None, 2),
                  (3, None, None),
                  ]
        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)",
                                params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()

        # row 0
        self.assertEqual(rows['b'][0], npdates[0])
        self.assertEqual(rows['c'][0], 0)
        # row 1
        self.assertEqual(rows['b'][1], npdates[1])
        self.assertEqual(rows['c'][1], -2147483648)
        # row 2
        # The line below does not work without the str() representation
        self.assertEqual(str(rows['b'][2]), str(np.datetime64('NaT')))
        self.assertEqual(rows['c'][2], 2)
        # row 3
        self.assertEqual(str(rows['b'][3]), str(np.datetime64('NaT')))
        self.assertEqual(rows['c'][3], -2147483648)

    def test_null_time(self):
        if np.__version__ < "1.7": return
        dates = [time(0, 0, i) for i in range(1,3)]
        npdates = np.array(range(1,3), dtype="timedelta64[s]")

        self.cursor.execute("create table t1(a int, b time, c int)")
        params = [(0, dates[0], 0),
                  (1, dates[1], None),
                  (2, None, 2),
                  (3, None, None),
                  ]
        self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)",
                                params)

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()

        # row 0
        self.assertEqual(rows['b'][0], npdates[0])
        self.assertEqual(rows['c'][0], 0)
        # row 1
        self.assertEqual(rows['b'][1], npdates[1])
        self.assertEqual(rows['c'][1], -2147483648)
        # row 2
        # The line below does not work without the str() representation
        self.assertEqual(str(rows['b'][2]), str(np.timedelta64('NaT')))
        self.assertEqual(rows['c'][2], 2)
        # row 3
        self.assertEqual(str(rows['b'][3]), str(np.timedelta64('NaT')))
        self.assertEqual(rows['c'][3], -2147483648)

    #
    # partial fetchs
    #

    def test_partial_fetch_dict(self):
        self.cursor.execute("create table t1(a real, b int)")

        params = [ (i, i) for i in range(6) ]

        self.cursor.executemany("insert into t1(a, b) values (?,?)", params)
        self.cursor.execute("select * from t1 order by a")

        # Row 0
        rows = self.cursor.fetchdictarray(1)
        self.assertEqual(len(rows['a']), 1)
        self.assertEqual(params[0][0], rows['a'][0])
        self.assertEqual(params[0][1], rows['b'][0])

        # Rows 1,2
        rows = self.cursor.fetchdictarray(2)
        self.assertEqual(len(rows['a']), 2)
        self.assertEqual(params[1][0], rows['a'][0])
        self.assertEqual(params[1][1], rows['b'][0])
        self.assertEqual(params[2][0], rows['a'][1])
        self.assertEqual(params[2][1], rows['b'][1])

        # Rows 3,4,5
        rows = self.cursor.fetchdictarray(3)
        self.assertEqual(len(rows['a']), 3)
        self.assertEqual(params[3][0], rows['a'][0])
        self.assertEqual(params[3][1], rows['b'][0])
        self.assertEqual(params[4][0], rows['a'][1])
        self.assertEqual(params[4][1], rows['b'][1])
        self.assertEqual(params[5][0], rows['a'][2])
        self.assertEqual(params[5][1], rows['b'][2])

        # A new fetch should return a length 0 container
        rows = self.cursor.fetchdictarray(1)
        self.assertEqual(len(rows['a']), 0)

    def test_partial_fetch_sarray(self):
        self.cursor.execute("create table t1(a real, b int)")

        params = [ (i, i) for i in range(6) ]

        self.cursor.executemany("insert into t1(a, b) values (?,?)", params)
        self.cursor.execute("select * from t1 order by a")

        # Row 0
        rows = self.cursor.fetchsarray(1)
        self.assertEqual(len(rows['a']), 1)
        self.assertEqual(params[0][0], rows['a'][0])
        self.assertEqual(params[0][1], rows['b'][0])

        # Rows 1,2
        rows = self.cursor.fetchsarray(2)
        self.assertEqual(len(rows['a']), 2)
        self.assertEqual(params[1][0], rows['a'][0])
        self.assertEqual(params[1][1], rows['b'][0])
        self.assertEqual(params[2][0], rows['a'][1])
        self.assertEqual(params[2][1], rows['b'][1])

        # Rows 3,4,5
        rows = self.cursor.fetchsarray(3)
        self.assertEqual(len(rows['a']), 3)
        self.assertEqual(params[3][0], rows['a'][0])
        self.assertEqual(params[3][1], rows['b'][0])
        self.assertEqual(params[4][0], rows['a'][1])
        self.assertEqual(params[4][1], rows['b'][1])
        self.assertEqual(params[5][0], rows['a'][2])
        self.assertEqual(params[5][1], rows['b'][2])

        # A new fetch should return a length 0 container
        rows = self.cursor.fetchsarray(1)
        self.assertEqual(len(rows['a']), 0)

    def test_partial_fetch_dict2(self):
        self.cursor.execute("create table t1(a real, b int)")

        params = [ (i, i) for i in range(6) ]

        self.cursor.executemany("insert into t1(a, b) values (?,?)", params)
        self.cursor.execute("select * from t1 order by a")

        # Rows 0,1,2
        rows = self.cursor.fetchdictarray(3)
        self.assertEqual(len(rows['a']), 3)
        self.assertEqual(params[0][0], rows['a'][0])
        self.assertEqual(params[0][1], rows['b'][0])
        self.assertEqual(params[1][0], rows['a'][1])
        self.assertEqual(params[1][1], rows['b'][1])
        self.assertEqual(params[2][0], rows['a'][2])
        self.assertEqual(params[2][1], rows['b'][2])

        # Row 3
        rows = self.cursor.fetchdictarray(1)
        self.assertEqual(len(rows['a']), 1)
        self.assertEqual(params[3][0], rows['a'][0])
        self.assertEqual(params[3][1], rows['b'][0])

        # Rows 4,5
        rows = self.cursor.fetchdictarray()
        self.assertEqual(len(rows['a']), 2)
        self.assertEqual(params[4][0], rows['a'][0])
        self.assertEqual(params[4][1], rows['b'][0])
        self.assertEqual(params[5][0], rows['a'][1])
        self.assertEqual(params[5][1], rows['b'][1])

        # A new fetch should return a length 0 container
        rows = self.cursor.fetchdictarray(1)
        self.assertEqual(len(rows['a']), 0)

    #
    # Unsupported data types
    #

    def test_unsupported_decimal(self):
        self.cursor.execute("create table t1(a int, b decimal)")
        params = [(0, 2.32)]
        self.cursor.executemany("insert into t1(a, b) values (?,?)",
                                params)
        self.cursor.execute("select * from t1")
        self.assertRaises(TypeError, self.cursor.fetchdictarray, ())

    def test_unsupported_binary(self):
        if backend == "postgresql":
            # Postgres does not have support for a binary datatype
            return
        self.cursor.execute("create table t1(a int, b binary)")
        params = [(0, "2.32")]
        self.cursor.executemany("insert into t1(a, b) values (?,?)",
                                params)
        self.cursor.execute("select * from t1")
        self.assertRaises(TypeError, self.cursor.fetchdictarray, ())

    def test_unsupported_timestamp(self):
        if np.__version__ >= '1.7':
            # This is supported when using NumPy  >= 1.7
            return

        self.cursor.execute("create table t1(a int, b timestamp)")
        params = [(0, "2008-08-08 08:08:08")]
        self.cursor.executemany("insert into t1(a, b) values (?,?)",
                                params)
        self.cursor.execute("select * from t1")
        self.assertRaises(TypeError, self.cursor.fetchdictarray, ())

    #
    # misc
    #

    def test_varchar(self):
        self.cursor.execute("create table t1(a int, b varchar(4), c int)")

        # Generate strings below and above 4 char long
        params = [ (i, str(16**i), i) for i in range(6) ]

        # PostgreSQL does not allow truncation during sting injection
        if backend == 'postgresql':
            self.cursor.executemany("insert into t1(a, b, c) values (?,?::varchar(4),?)", params)
        else:
            self.cursor.executemany("insert into t1(a, b, c) values (?,?,?)", params)
        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchall()
        for param, row in zip(params, rows):
            self.assertEqual(param[0], row[0])
            if backend == 'sqlite':
                self.assertEqual(param[1], row[1])  # SQLite does not truncate (!)
            else:
                self.assertEqual(param[1][:4], row[1])
            self.assertEqual(param[2], row[2])

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchsarray()
        for param, row in zip(params, rows):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1][:4], row[1])
            self.assertEqual(param[2], row[2])

        self.cursor.execute("select a, b, c from t1 order by a")
        rows = self.cursor.fetchdictarray()
        for param, row in zip(params, zip(rows['a'], rows['b'], rows['c'])):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1][:4], row[1])
            self.assertEqual(param[2], row[2])

    def test_int_varchar(self):
        self.cursor.execute("create table t1(a int, b varchar(10))")

        params = [ (i, str(i)) for i in range(1, 6) ]

        self.cursor.executemany("insert into t1(a, b) values (?,?)", params)

        count = self.cursor.execute("select count(*) from t1").fetchone()[0]
        self.assertEqual(count, len(params))

        self.cursor.execute("select a, b from t1 order by a")
        rows = self.cursor.fetchall()
        self.assertEqual(count, len(rows))
        for param, row in zip(params, rows):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1], row[1])

        self.cursor.execute("select a, b from t1 order by a")
        rows = self.cursor.fetchsarray()
        self.assertEqual(count, len(rows))
        for param, row in zip(params, rows):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1], row[1])

        self.cursor.execute("select a, b from t1 order by a")
        rows = self.cursor.fetchdictarray()
        self.assertEqual(count, len(rows['a']))
        for param, row in zip(params, zip(rows['a'], rows['b'])):
            self.assertEqual(param[0], row[0])
            self.assertEqual(param[1], row[1])

    def test_exhaust_execute_buffer(self):
        self.cursor.execute("create table t1(a int, b varchar(10))")

        params = [ (i, str(i)) for i in range(1, 6) ]

        self.cursor.executemany("insert into t1(a, b) values (?,?)", params)

        count = self.cursor.execute("select count(*) from t1").fetchone()[0]
        self.assertEqual(count, len(params))

        self.cursor.execute("select a, b from t1")
        # First exhaust all the data in select buffer
        rows = self.cursor.fetchall()
        # Then check that the results are properly empty
        rows2 = self.cursor.fetchall()
        self.assertEqual(len(rows2), 0)

        # Now, for structured arrays
        rows3 = self.cursor.fetchsarray()
        self.assertEqual(len(rows3), 0)

        rows4 = self.cursor.fetchdictarray()
        self.assertEqual(len(rows4['a']), 0)
        self.assertEqual(len(rows4['b']), 0)


def main():
    global backend
    from optparse import OptionParser
    parser = OptionParser(usage=usage)
    parser.add_option("-v", "--verbose", action="count", help="Increment test verbosity (can be used multiple times)")
    parser.add_option("-d", "--debug", action="store_true", default=False, help="Print debugging items")
    parser.add_option("-t", "--test", help="Run only the named test")

    (options, args) = parser.parse_args()

    if len(args) > 1:
        parser.error('Only one argument is allowed.  Do you need quotes around the connection string?')

    if not args:
        # The default is to run the tests with the sqlite backend
        backend = 'sqlite'
    else:
        backend = args[0]

    print("Running with '%s' backend" % backend)
    connection_string = load_setup_connection_string(backend)
    if not connection_string:
        parser.print_help()
        raise SystemExit()
    print("Connection string:", connection_string)

    cnxn = pyodbc.connect(connection_string)
    print_library_info(cnxn)
    cnxn.close()

    suite = load_tests(GenericTestCase, options.test, connection_string)

    testRunner = unittest.TextTestRunner(verbosity=options.verbose)
    result = testRunner.run(suite)


if __name__ == '__main__':

    # Add the build directory to the path so we're testing the latest
    # build, not the installed version.

    add_to_path()

    from iopro import pyodbc
    main()
