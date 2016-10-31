"""
unittests for IOPro/pyodbc: money type support
"""
from __future__ import absolute_import, print_function, division

from unittest_support import IOProPyodbcTest
import unittest
import numpy as np

class TestMoney(IOProPyodbcTest):
    def test_smallmoney_dictarray(self):
        try:
            self.conn.execute('drop table SMALL_MONEY_TEST').commit()
        except Exception:
            pass

        self.conn.execute('create table SMALL_MONEY_TEST (val smallmoney not null)').commit()
        values = ((42.70,), (32.50,), (12.43,))
        cur = self.conn.cursor()
        cur.executemany('insert into SMALL_MONEY_TEST values (?)', values)
        cur.commit()

        # smallmoney maps to decimal in odbc (with 4 decimal digits). In IOPro/pyodbc
        # decimal maps to double precision floating point
        da = self.conn.execute('select * from SMALL_MONEY_TEST').fetchdictarray()
        self.assertEqual(np.float64, da['val'].dtype)
        self.assertTrue(np.allclose(np.array(values).ravel('C'), da['val'], rtol=0.0, atol=1e-4))
        self.conn.execute('drop table SMALL_MONEY_TEST').commit()


    def test_money_dictarray(self):
        try:
            self.conn.execute('drop table MONEY_TEST').commit()
        except Exception:
            pass

        self.conn.execute('create table MONEY_TEST (val money not null)').commit()
        values = ((42.70,), (32.50,), (12.43,))
        cur = self.conn.cursor()
        cur.executemany('insert into MONEY_TEST values (?)', values)
        cur.commit()

        da = self.conn.execute('select * from MONEY_TEST').fetchdictarray()

        # money maps to decimal. It contains 4 decimal digits. In IOPro/pyodbc decimal
        # maps to double precision floating point.
        self.assertEqual(np.float64, da['val'].dtype)
        self.assertTrue(np.allclose(np.array(values).ravel('C'), da['val'], rtol=0.0, atol=1e-4))
        self.conn.execute('drop table MONEY_TEST').commit()


if __name__ == '__main__':
    unittest.main()
