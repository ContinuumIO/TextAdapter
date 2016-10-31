"""
unittest for IOPro/pyodbc: setting a text limit
"""

from __future__ import absolute_import, print_function, division

from unittest_support import IOProPyodbcTest, text_limit
import unittest
import numpy as np
from iopro import pyodbc

# test not requiring a connection
class TestSetTextLimitInterface(unittest.TestCase):
    def test_iopro_set_text_limit_interface(self):
        test_values = [80, 60, 10000, -1, -40, 30, -12, 0]
        old_values = [pyodbc.iopro_set_text_limit(i) for i in test_values]
        old_values.append(pyodbc.iopro_set_text_limit(old_values[0])) # restore original

        for i, val in enumerate(test_values):
            if val < 0:
                # func normalizes negative values to -1
                self.assertEqual(-1, old_values[i+1])
            else:
                self.assertEqual(val, old_values[i+1])


class TestSetLimit(IOProPyodbcTest):
    def test_simple(self):
        limit = 100
        try:
            self.conn.execute('drop table SET_TEXT_LIMIT_TEST').commit()
        except Exception:
            pass

        self.conn.execute('create table SET_TEXT_LIMIT_TEST (val varchar({0}) not null)'.format(limit*2)).commit()
        lengths = list(range(0, limit*2, limit//40))
        test_data = [('x'*i,) for i in lengths]
        cur = self.conn.cursor()
        cur.executemany('insert into SET_TEXT_LIMIT_TEST values (?)', test_data)
        cur.commit()

        with text_limit(limit):
            da = self.conn.execute('select * from SET_TEXT_LIMIT_TEST').fetchdictarray()

        val = da['val']
        self.assertTrue(np.string_, val.dtype.type)
        self.assertTrue(limit+1, val.dtype.itemsize)


if __name__ == '__main__':
    unittest.main()
