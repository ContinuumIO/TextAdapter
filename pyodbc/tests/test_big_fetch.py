"""
unittest checking that big fetches that involve iterating/realloc in the C-side
is working.

There've been a number of issues related to this loop.
"""

from __future__ import absolute_import, print_function, division

from unittest_support import IOProPyodbcTest
import unittest
import sys
import numpy as np

class TestBigFetch_unicode(IOProPyodbcTest):
    def _create_test_table(self, N):
        try:
            self.conn.execute('drop table BIG_FETCH_TEST').commit()
        except Exception:
            pass

        self.conn.execute('''create table BIG_FETCH_TEST (
                                    unicode_not_null nvarchar(80) not null,
                                    int_nullable int NULL)''').commit()

        cursor = self.conn.cursor()
        crazy_nulls = set([1,2,3,5,8,13,21,34])
        for i in range(N):
            cursor.execute('''insert into BIG_FETCH_TEST(unicode_not_null, int_nullable)
                                      values (?,?)''',
                           ('something {0}'.format(i),
                            i if (i % 42) not in crazy_nulls else None ))
        cursor.commit()



    def _check_no_nulls(self, N):
        self._create_test_table(N)
        da = self.conn.execute('''
                select all [unicode_not_null]
                from BIG_FETCH_TEST
            ''').fetchdictarray()
        unicode_not_null = da['unicode_not_null']
        self.assertEqual(N, len(unicode_not_null))
        for i in xrange(N):
            self.assertEqual(unicode_not_null[i], 'something {0}'.format(i))

    def _check_nulls(self, N, query_nulls):
        self._create_test_table(N)
        da = self.conn.execute('''
                select all [int_nullable]
                from BIG_FETCH_TEST
            ''').fetchdictarray(return_nulls=query_nulls)
        int_nullable = da['int_nullable']
        if query_nulls:
            nulls = da['int_nullable_isnull']
        crazy_nulls = set([1,2,3,5,8,13,21,34])
        self.assertEqual(N, len(int_nullable))
        for i in xrange(N):
            if i % 42 in crazy_nulls:
                # this should be null
                self.assertEqual(int_nullable[i], -2147483648)
                if query_nulls:
                    self.assertTrue(nulls[i],
                                     msg='wrong null value in row {0} (expected {1} got {2})'.format(i, True, nulls[i]))
            else:
                # not null
                self.assertEqual(int_nullable[i], i)
                if query_nulls:
                    self.assertFalse(nulls[i],
                                     msg='wrong null value in row {0} (expected {1} got {2})'.format(i, False, nulls[i]))


    def test_check_no_nulls_single(self):
        self._check_no_nulls(1000)

    def test_check_no_nulls_exact(self):
        self._check_no_nulls(10000)

    def test_check_no_nulls_multiple(self):
        self._check_no_nulls(30000)

    def test_check_no_nulls_modulo(self):
        self._check_no_nulls(32000)

    def test_check_nulls_single(self):
        self._check_nulls(1000, False)

    def test_check_nulls_exact(self):
        self._check_nulls(10000, False)

    def test_check_nulls_multiple(self):
        self._check_nulls(30000, False)

    def test_check_nulls_modulo(self):
        self._check_nulls(32000, False)

    def test_check_with_nulls_single(self):
        self._check_nulls(1000, True)

    def test_check_with_nulls_exact(self):
        self._check_nulls(10000, True)

    def test_check_with_nulls_multiple(self):
        self._check_nulls(30000, True)

    def test_check_with_nulls_modulo(self):
        self._check_nulls(32000, True)

if __name__ == '__main__':
    unittest.main()
