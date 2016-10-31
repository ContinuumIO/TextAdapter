"""
unittests for IOPro/pyodbc: money type support
"""
from __future__ import absolute_import, print_function, division

from unittest_support import IOProPyodbcTest
import unittest
import numpy as np

class TestUnicode(IOProPyodbcTest):
    def test_nchar(self):
        try:
            self.conn.execute('drop table NCHAR_TEST').commit()
        except Exception:
            pass

        self.conn.execute('create table NCHAR_TEST (val NCHAR(42) not null)').commit()
        values = ((u"some small string",), (u"foo",))
        cur = self.conn.cursor()
        cur.executemany('insert into NCHAR_TEST values (?)', values)
        cur.commit()

        da = self.conn.execute('select * from NCHAR_TEST').fetchdictarray()
        self.assertEqual(np.unicode_, da['val'].dtype.type)
        self.conn.execute('drop table NCHAR_TEST').commit()


    def test_nvarchar(self):
        try:
            self.conn.execute('drop table NVARCHAR_TEST').commit()
        except Exception:
            pass

        self.conn.execute('create table NVARCHAR_TEST (val NVARCHAR(42) not null)').commit()
        values = ((u"some small string",), (u"foo",))
        cur = self.conn.cursor()
        cur.executemany('insert into NVARCHAR_TEST values (?)', values)
        cur.commit()

        da = self.conn.execute('select * from NVARCHAR_TEST').fetchdictarray()
        self.assertEqual(np.unicode_, da['val'].dtype.type)
        self.conn.execute('drop table NVARCHAR_TEST').commit()


if __name__ == '__main__':
    unittest.main()
