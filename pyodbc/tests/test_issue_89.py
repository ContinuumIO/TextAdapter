"""
unittests UIPro/pyodbc: issue #89 on github
"""

from __future__ import absolute_import, print_function, division

from unittest_support import IOProPyodbcTest
import unittest
import numpy as np

class TestIssue89(IOProPyodbcTest):
    def test_issue_89(self):
        #note that the issue results in a segfault.
        #this sample will also make some basic testing on the number
        #of returned rows
        try:
            self.conn.execute('drop table ISSUE_89_TEST').commit()
        except Exception:
            pass

        row_count = 100000
        batch = 1000
        self.conn.execute('''create table ISSUE_89_TEST (
                                    name nvarchar(200),
                                    fval float(24),
                                    ival int)''').commit()

        for i in range(0,row_count, batch):
            cursor = self.conn.cursor()
            cursor.executemany('insert into ISSUE_89_TEST values (?, ?, ?)',
                               [('sample', 42, 31.0)] * batch)
        cursor.commit()
        cursor.execute('select * from ISSUE_89_TEST')
        da = cursor.fetchdictarray()
        self.assertEqual(len(da['name']), row_count)
        del da
        cursor.execute('select * from ISSUE_89_TEST')
        sa = cursor.fetchsarray()
        self.assertEqual(len(sa), row_count)
        del sa


if __name__ == '__main__':
    unittest.main()
