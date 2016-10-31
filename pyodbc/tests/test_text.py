"""
unittests for IOPro/pyodbc: text support
"""
from __future__ import absolute_import, print_function, division

from unittest_support import (IOProPyodbcTest, text_limit)
import unittest
import numpy as np

class TestText(IOProPyodbcTest):
    def test_text_dictarray(self):
        try:
            self.conn.execute('drop table TEXT_TEST').commit()
        except Exception:
            pass

        self.conn.execute('create table TEXT_TEST (val text not null)').commit()
        values = (("some small string",), ("foo",))
        cur = self.conn.cursor()
        cur.executemany('insert into TEXT_TEST values (?)', values)
        cur.commit()

        with text_limit(max(*[len(v[0]) for v in values])):
            da = self.conn.execute('select * from TEXT_TEST').fetchdictarray()

        self.assertEqual(np.string_, da['val'].dtype.type)
        self.assertTrue(np.all(da['val'] == np.ravel(values)))
        self.conn.execute('drop table TEXT_TEST').commit()

    def test_text_dictarray_big_entry(self):
        try:
            self.conn.execute('drop table TEXT_TEST').commit()
        except Exception:
            pass

        self.conn.execute('create table TEXT_TEST (val text not null)').commit()
        values = (("some small string",), ("foo",), ("0123456789abcde\n"*250,))
        cur = self.conn.cursor()
        cur.executemany('insert into TEXT_TEST values (?)', values)
        cur.commit()

        
        with text_limit(max(*[len(v[0]) for v in values])):
            da = self.conn.execute('select * from TEXT_TEST').fetchdictarray()

        self.assertEqual(np.string_, da['val'].dtype.type)
        self.assertTrue(np.all(da['val'] == np.ravel(values)))
        self.conn.execute('drop table TEXT_TEST').commit()



if __name__ == '__main__':
    unittest.main()
