"""
unittests UIPro/pyodbc: issue #90 on github
"""

from __future__ import absolute_import, print_function, division

from unittest_support import IOProPyodbcTest
import unittest
import sys
import numpy as np

class TestIssue90(IOProPyodbcTest):
    def _create_test_table(self):
        try:
            self.conn.execute('drop table ISSUE_90_TEST').commit()
        except Exception:
            pass

        self.conn.execute('''create table ISSUE_90_TEST (
                                    name nvarchar(255))''').commit()


    def _issue_90(self, N):
        print("Creating table with {0} elements\n".format(N))
        self._create_test_table()
        cursor = self.conn.cursor()
        for _ in range(N):
            cursor.execute('''insert into ISSUE_90_TEST values (?)''', ('foo'))
        cursor.commit()
        sys.stdout.flush()

        da = cursor.execute('''SELECT ALL [name] FROM ISSUE_90_TEST''').fetchdictarray()
        self.assertTrue((da['name']=='foo').all())

    def test_issue_90_10001(self):
        self._issue_90(10001)

    def test_issue_90_100(self):
        self._issue_90(100)

    def test_issue_90_1000(self):
        self._issue_90(1000)

    def test_issue_90_5000(self):
        self._issue_90(5000)

    def test_issue_90_10000(self):
        self._issue_90(10000)

    def test_issue_90_15000(self):
        self._issue_90(15000)

    def test_issue_90_20000(self):
        self._issue_90(20000)

    def test_issue_90_25000(self):
        self._issue_90(25000)

    def test_issue_90_30000(self):
        self._issue_90(30000)


if __name__ == '__main__':
    unittest.main()
