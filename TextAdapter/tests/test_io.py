import sys
import gzip
import os
import threading
from tempfile import mkstemp, NamedTemporaryFile
import time
from datetime import datetime
import warnings
import gc
from numpy.testing.utils import WarningManager

import numpy as np
import numpy.ma as ma
from numpy.lib._iotools import ConverterError, ConverterLockError, \
                               ConversionWarning
from numpy.compat import asbytes, asbytes_nested, bytes

from nose import SkipTest
from numpy.ma.testutils import (TestCase, assert_equal, assert_array_equal,
                                assert_raises, run_module_suite)
from numpy.testing import assert_warns, assert_
import iopro
import unittest
from six import StringIO
from io import BytesIO


if 'expectedFailure' not in dir(unittest):

    def expectedFailure(func):
        def wrapper(*args, **kwargs):
            pass
        return wrapper

    setattr(unittest, 'expectedFailure', expectedFailure)


MAJVER, MINVER = sys.version_info[:2]

def strptime(s, fmt=None):
    """This function is available in the datetime module only
    from Python >= 2.5.

    """
    if sys.version_info[0] >= 3:
        return datetime(*time.strptime(s.decode('latin1'), fmt)[:3])
    else:
        return datetime(*time.strptime(s, fmt)[:3])


class TestLoadTxt(TestCase):
    def test_record(self):
        c = StringIO()
        c.write('1 2\n3 4')
        c.seek(0)
        x = iopro.loadtxt(c, dtype=[('x', np.int32), ('y', np.int32)])
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        assert_array_equal(x, a)

        d = StringIO()
        d.write('M 64.0 75.0\nF 25.0 60.0')
        d.seek(0)
        mydescriptor = {'names': ('gender', 'age', 'weight'),
                        'formats': ('S1',
                                    'i4', 'f4')}
        b = np.array([('M', 64.0, 75.0),
                      ('F', 25.0, 60.0)], dtype=mydescriptor)
        y = iopro.loadtxt(d, dtype=mydescriptor)
        assert_array_equal(y, b)

    def test_array(self):
        c = StringIO()
        c.write('1 2\n3 4')

        c.seek(0)
        x = iopro.loadtxt(c, dtype=int)
        a = np.array([[1, 2], [3, 4]], int)
        assert_array_equal(x, a)

        c.seek(0)
        x = iopro.loadtxt(c, dtype=float)
        a = np.array([[1, 2], [3, 4]], float)
        assert_array_equal(x, a)

    def test_1D(self):
        c = StringIO()
        c.write('1\n2\n3\n4\n')
        c.seek(0)
        x = iopro.loadtxt(c, dtype=int)
        a = np.array([1, 2, 3, 4], int)
        assert_array_equal(x, a)

        c = StringIO()
        c.write('1,2,3,4\n')
        c.seek(0)
        x = iopro.loadtxt(c, dtype=int, delimiter=',')
        a = np.array([1, 2, 3, 4], int)
        assert_array_equal(x, a)

    @unittest.expectedFailure
    def test_missing(self):
        c = StringIO()
        c.write('1,2,3,,5\n')
        c.seek(0)
        x = iopro.loadtxt(c, dtype=int, delimiter=',', \
            converters={3:lambda s: int(s or - 999)})
        a = np.array([1, 2, 3, -999, 5], int)
        assert_array_equal(x, a)

    @unittest.expectedFailure
    def test_converters_with_usecols(self):
        c = StringIO()
        c.write('1,2,3,,5\n6,7,8,9,10\n')
        c.seek(0)
        x = iopro.loadtxt(c, dtype=int, delimiter=',', \
            converters={3:lambda s: int(s or - 999)}, \
            usecols=(1, 3,))
        a = np.array([[2, -999], [7, 9]], int)
        assert_array_equal(x, a)

    def test_comments(self):
        c = StringIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = iopro.loadtxt(c, dtype=int, delimiter=',', \
            comments='#')
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_skiprows(self):
        c = StringIO()
        c.write('comment\n1,2,3,5\n')
        c.seek(0)
        x = iopro.loadtxt(c, dtype=int, delimiter=',', \
            skiprows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

        c = StringIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = iopro.loadtxt(c, dtype=int, delimiter=',', \
            skiprows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_usecols(self):
        a = np.array([[1, 2], [3, 4]], float)
        c = BytesIO()
        np.savetxt(c, a)
        c.seek(0)
        x = iopro.loadtxt(c, dtype=float, usecols=(1,))
        assert_array_equal(x, a[:, 1])

        a = np.array([[1, 2, 3], [3, 4, 5]], float)
        c = BytesIO()
        np.savetxt(c, a)
        c.seek(0)
        x = iopro.loadtxt(c, dtype=float, usecols=(1, 2))
        assert_array_equal(x, a[:, 1:])

        # Testing with arrays instead of tuples.
        c.seek(0)
        x = iopro.loadtxt(c, dtype=float, usecols=np.array([1, 2]))
        assert_array_equal(x, a[:, 1:])

        # Checking with dtypes defined converters.
        data = '''JOE 70.1 25.3\nBOB 60.5 27.9'''
        c = StringIO(data)
        names = ['stid', 'temp']
        dtypes = ['S3', 'f8']
        arr = iopro.loadtxt(c, usecols=(0, 2), dtype=list(zip(names, dtypes)))
        assert_equal(arr['stid'], asbytes_nested(["JOE", "BOB"]))
        assert_equal(arr['temp'], [25.3, 27.9])

    def test_fancy_dtype(self):
        c = StringIO()
        c.write('1,2,3.0\n4,5,6.0\n')
        c.seek(0)
        dt = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        x = iopro.loadtxt(c, dtype=dt, delimiter=',')
        a = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dt)
        assert_array_equal(x, a)

    def test_shaped_dtype(self):
        c = StringIO("aaaa  1.0  8.0  1 2 3 4 5 6")
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
                       ('block', int, (2, 3))])
        x = iopro.loadtxt(c, dtype=dt)
        a = np.array([('aaaa', 1.0, 8.0, [[1, 2, 3], [4, 5, 6]])],
                     dtype=dt)
        assert_array_equal(x, a)

    def test_3d_shaped_dtype(self):
        c = StringIO("aaaa  1.0  8.0  1 2 3 4 5 6 7 8 9 10 11 12")
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
                       ('block', int, (2, 2, 3))])
        x = iopro.loadtxt(c, dtype=dt)
        a = np.array([('aaaa', 1.0, 8.0, [[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])],
                     dtype=dt)
        assert_array_equal(x, a)

    def test_empty_file(self):
        warn_ctx = WarningManager()
        warn_ctx.__enter__()
        try:
            warnings.filterwarnings("ignore",
                    message="loadtxt: Empty input file:")
            c = StringIO()
            x = iopro.loadtxt(c)
            assert_equal(x.shape, (0,))
            x = iopro.loadtxt(c, dtype=np.int64)
            assert_equal(x.shape, (0,))
            assert_(x.dtype == np.int64)
        finally:
            warn_ctx.__exit__()

    @unittest.expectedFailure
    def test_unused_converter(self):
        assert_equal(True, False)
        c = StringIO()
        c.writelines(['1 21\n', '3 42\n'])
        c.seek(0)
        data = iopro.loadtxt(c, usecols=(1,),
                          converters={0: lambda s: int(s, 16)})
        assert_array_equal(data, [21, 42])

        c.seek(0)
        data = iopro.loadtxt(c, usecols=(1,),
                          converters={1: lambda s: int(s, 16)})
        assert_array_equal(data, [33, 66])

    @unittest.expectedFailure
    def test_dtype_with_object(self):
        assert_equal(True, False)
        "Test using an explicit dtype with an object"
        from datetime import date
        import time
        data = """ 1; 2001-01-01
                   2; 2002-01-31 """
        ndtype = [('idx', int), ('code', np.object)]
        func = lambda s: strptime(s.strip(), "%Y-%m-%d")
        converters = {1: func}
        test = iopro.loadtxt(StringIO(data), delimiter=";", dtype=ndtype,
                             converters=converters)
        control = np.array([(1, datetime(2001, 1, 1)), (2, datetime(2002, 1, 31))],
                           dtype=ndtype)
        assert_equal(test, control)

    def test_uint64_type(self):
        tgt = (9223372043271415339, 9223372043271415853)
        c = StringIO()
        c.write("%s %s" % tgt)
        c.seek(0)
        res = iopro.loadtxt(c, dtype=np.uint64)
        assert_equal(res, tgt)

    def test_int64_type(self):
        tgt = (-9223372036854775807, 9223372036854775807)
        c = StringIO()
        c.write("%s %s" % tgt)
        c.seek(0)
        res = iopro.loadtxt(c, dtype=np.int64)
        assert_equal(res, tgt)

    def test_universal_newline(self):
        f, name = mkstemp()
        os.write(f, b'1 21\r3 42\r')
        os.close(f)

        try:
            data = iopro.loadtxt(name)
            assert_array_equal(data, [[1, 21], [3, 42]])
        finally:
            os.unlink(name)

    def test_empty_field_after_tab(self):
        c = StringIO()
        c.write('1 \t2 \t3\tstart \n4\t5\t6\t  \n7\t8\t9.5\t')
        c.seek(0)
        dt = { 'names': ('x', 'y', 'z', 'comment'),
               'formats': ('<i4', '<i4', '<f4', '|S8')}
        x = iopro.loadtxt(c, dtype=dt, delimiter='\t')
        a = np.array(['start ', '  ', ''], dtype='|S8')
        assert_array_equal(x['comment'], a)

    def test_structure_unpack(self):
        txt = StringIO("M 21 72\nF 35 58")
        dt = { 'names': ('a', 'b', 'c'), 'formats': ('|S1', '<i4', '<f4')}
        a, b, c = iopro.loadtxt(txt, dtype=dt, unpack=True)
        assert_(a.dtype.str == '|S1')
        assert_(b.dtype.str == '<i4')
        assert_(c.dtype.str == '<f4')
        assert_array_equal(a, np.array(['M', 'F'], dtype='|S1'))
        assert_array_equal(b, np.array([21, 35], dtype='<i4'))
        assert_array_equal(c, np.array([ 72.,  58.], dtype='<f4'))

    def test_ndmin_keyword(self):
        c = StringIO()
        c.write('1,2,3\n4,5,6')
        c.seek(0)
        assert_raises(iopro.DataTypeError, iopro.loadtxt, c, ndmin=3)
        c.seek(0)
        assert_raises(iopro.DataTypeError, iopro.loadtxt, c, ndmin=1.5)
        c.seek(0)
        x = iopro.loadtxt(c, dtype=int, delimiter=',', ndmin=1)
        a = np.array([[1, 2, 3], [4, 5, 6]])
        assert_array_equal(x, a)
        d = StringIO()
        d.write('0,1,2')
        d.seek(0)
        x = iopro.loadtxt(d, dtype=int, delimiter=',', ndmin=2)
        assert_(x.shape == (1, 3))
        d.seek(0)
        x = iopro.loadtxt(d, dtype=int, delimiter=',', ndmin=1)
        assert_(x.shape == (3,))
        d.seek(0)
        x = iopro.loadtxt(d, dtype=int, delimiter=',', ndmin=0)
        assert_(x.shape == (3,))
        e = StringIO()
        e.write('0\n1\n2')
        e.seek(0)
        x = iopro.loadtxt(e, dtype=int, delimiter=',', ndmin=2)
        assert_(x.shape == (3, 1))
        e.seek(0)
        x = iopro.loadtxt(e, dtype=int, delimiter=',', ndmin=1)
        assert_(x.shape == (3,))
        e.seek(0)
        x = iopro.loadtxt(e, dtype=int, delimiter=',', ndmin=0)
        assert_(x.shape == (3,))

        # Test ndmin kw with empty file.
        warn_ctx = WarningManager()
        warn_ctx.__enter__()
        try:
            warnings.filterwarnings("ignore",
                    message="loadtxt: Empty input file:")
            f = StringIO()
            assert_(iopro.loadtxt(f, ndmin=2).shape == (0, 1,))
            assert_(iopro.loadtxt(f, ndmin=1).shape == (0,))
        finally:
            warn_ctx.__exit__()

    def test_generator_source(self):
        def count():
            for i in range(10):
                yield "%d" % i

        res = iopro.loadtxt(count())
        assert_array_equal(res, np.arange(10))


#####***REMOVED***---


class TestFromTxt(TestCase):
    #
    def test_array(self):
        "Test outputing a standard ndarray"
        data = BytesIO(b'1 2\n3 4')
        control = np.array([[1, 2], [3, 4]], dtype=int)
        test = np.ndfromtxt(data, dtype=int)
        assert_array_equal(test, control)
        #
        data.seek(0)
        control = np.array([[1, 2], [3, 4]], dtype=float)
        test = iopro.loadtxt(data, dtype=float)
        assert_array_equal(test, control)

    def test_skiprows(self):
        "Test row skipping"
        control = np.array([1, 2, 3, 5], int)
        kwargs = dict(dtype=int, delimiter=',')
        #
        data = StringIO('# comment\n1,2,3,5\n')
        test = iopro.loadtxt(data, skiprows=1, **kwargs)
        assert_equal(test, control)

    @unittest.expectedFailure
    def test_skip_footer(self):
        data = ["# %i" % i for i in range(1, 6)]
        data.append("A, B, C")
        data.extend(["%i,%3.1f,%03s" % (i, i, i) for i in range(51)])
        data[-1] = "99,99"
        kwargs = dict(delimiter=",", names=True, skip_header=5, skip_footer=10)
        test = iopro.genfromtxt(StringIO("\n".join(data)), **kwargs)
        ctrl = np.array([("%f" % i, "%f" % i, "%f" % i) for i in range(41)],
                        dtype=[(_, float) for _ in "ABC"])
        assert_equal(test, ctrl)

    @unittest.expectedFailure
    def test_skip_footer_with_invalid(self):
        warn_ctx = WarningManager()
        warn_ctx.__enter__()
        try:
            basestr = '1 1\n2 2\n3 3\n4 4\n5  \n6  \n7  \n'
            warnings.filterwarnings("ignore")
            # Footer too small to get rid of all invalid values
            assert_raises(ValueError, iopro.genfromtxt,
                          StringIO(basestr), skip_footer=1)
            a = iopro.genfromtxt(StringIO(basestr), skip_footer=1, invalid_raise=False)
            assert_equal(a, np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]]))
            #
            a = iopro.genfromtxt(StringIO(basestr), skip_footer=3)
            assert_equal(a, np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]]))
            #
            basestr = '1 1\n2  \n3 3\n4 4\n5  \n6 6\n7 7\n'
            a = iopro.genfromtxt(StringIO(basestr), skip_footer=1, invalid_raise=False)
            assert_equal(a, np.array([[1., 1.], [3., 3.], [4., 4.], [6., 6.]]))
            a = iopro.genfromtxt(StringIO(basestr), skip_footer=3, invalid_raise=False)
            assert_equal(a, np.array([[1., 1.], [3., 3.], [4., 4.]]))
        finally:
            warn_ctx.__exit__()

    def test_commented_header(self):
        "Check that names can be retrieved even if the line is commented out."
        data = StringIO("""
#gender age weight
M   21  72.100000
F   35  58.330000
M   33  21.99
        """)
        # The # is part of the first name and should be deleted automatically.
        test = iopro.genfromtxt(data, names=True, dtype=None)
        ctrl = np.array([('M', 21, 72.1), ('F', 35, 58.33), ('M', 33, 21.99)],
                  # JNB: changed test because iopro defaults to object string
                  # instead of fixed length string, and unsigned long int
                  # instead of int.
                  dtype=[('gender', 'O'), ('age', 'u8'), ('weight', 'f8')])
                  #dtype=[('gender', '|S1'), ('age', int), ('weight', float)])
        assert_equal(test, ctrl)
        # Ditto, but we should get rid of the first element
        data = StringIO("""
# gender age weight
M   21  72.100000
F   35  58.330000
M   33  21.99
        """)
        test = iopro.genfromtxt(data, names=True, dtype=None)
        assert_equal(test, ctrl)

    @unittest.expectedFailure
    def test_invalid_converter(self):
        assert_equal(True, False)
        strip_rand = lambda x : float(('r' in x.lower() and x.split()[-1]) or
                                      (not 'r' in x.lower() and x.strip() or 0.0))
        strip_per = lambda x : float(('%' in x.lower() and x.split()[0]) or
                                     (not '%' in x.lower() and x.strip() or 0.0))
        s = StringIO("D01N01,10/1/2003 ,1 %,R 75,400,600\r\n" \
                              "L24U05,12/5/2003, 2 %,1,300, 150.5\r\n"
                              "D02N03,10/10/2004,R 1,,7,145.55")
        kwargs = dict(converters={2 : strip_per, 3 : strip_rand}, delimiter=",",
                      dtype=None)
        assert_raises(ConverterError, iopro.genfromtxt, s, **kwargs)

    @unittest.expectedFailure
    def test_tricky_converter_bug1666(self):
        "Test some corner case"
        assert_equal(True, False)
        s = StringIO('q1,2\nq3,4')
        cnv = lambda s:float(s[1:])
        test = iopro.genfromtxt(s, delimiter=',', converters={0:cnv})
        control = np.array([[1., 2.], [3., 4.]])
        assert_equal(test, control)

    @unittest.expectedFailure
    def test_dtype_with_object(self):
        "Test using an explicit dtype with an object"
        assert_equal(True, False)
        from datetime import date
        import time
        data = """ 1; 2001-01-01
                   2; 2002-01-31 """
        ndtype = [('idx', int), ('code', np.object)]
        func = lambda s: strptime(s.strip(), "%Y-%m-%d")
        converters = {1: func}
        test = iopro.genfromtxt(StringIO(data), delimiter=";", dtype=ndtype,
                             converters=converters)
        control = np.array([(1, datetime(2001, 1, 1)), (2, datetime(2002, 1, 31))],
                           dtype=ndtype)
        assert_equal(test, control)
        #
        ndtype = [('nest', [('idx', int), ('code', np.object)])]
        try:
            test = iopro.genfromtxt(StringIO(data), delimiter=";",
                                 dtype=ndtype, converters=converters)
        except NotImplementedError:
            errmsg = "Nested dtype involving objects should be supported."
            raise AssertionError(errmsg)

    @unittest.expectedFailure
    def test_userconverters_with_explicit_dtype(self):
        "Test user_converters w/ explicit (standard) dtype"
        data = StringIO('skip,skip,2001-01-01,1.0,skip')
        test = iopro.genfromtxt(data, delimiter=",", names=None, dtype=float,
                             usecols=(2, 3), converters={2: bytes})
        control = np.array([('2001-01-01', 1.)],
                           dtype=[('', '|S10'), ('', float)])
        assert_equal(test, control)


    def test_integer_delimiter(self):
        "Test using an integer for delimiter"
        data = "  1  2  3\n  4  5 67\n890123  4"
        test = iopro.genfromtxt(StringIO(data), delimiter=3, dtype=int)
        control = np.array([[1, 2, 3], [4, 5, 67], [890, 123, 4]])
        assert_equal(test, control)


    # JNB: masked arrays not supported yet
    @unittest.expectedFailure
    def test_missing_with_tabs(self):
        "Test w/ a delimiter tab"
        txt = "1\t2\t3\n\t2\t\n1\t\t3"
        test = iopro.genfromtxt(StringIO(txt), delimiter="\t",
                             usemask=True,)
        ctrl_d = np.array([(1, 2, 3), (np.nan, 2, np.nan), (1, np.nan, 3)],)
        ctrl_m = np.array([(0, 0, 0), (1, 0, 1), (0, 1, 0)], dtype=bool)
        assert_equal(test.data, ctrl_d)
        assert_equal(test.mask, ctrl_m)

    def test_usecols_as_css(self):
        "Test giving usecols with a comma-separated string"
        data = "1 2 3\n4 5 6"
        test = iopro.genfromtxt(StringIO(data),
                             names="a, b, c", usecols="a, c")
        ctrl = np.array([(1, 3), (4, 6)], dtype=[(_, float) for _ in "ac"])
        assert_equal(test, ctrl)

    def test_usecols_with_integer(self):
        "Test usecols with an integer"
        test = iopro.genfromtxt(StringIO("1 2 3\n4 5 6"), usecols=0)
        assert_equal(test, np.array([1., 4.]))

    def test_usecols_with_named_columns(self):
        "Test usecols with named columns"
        ctrl = np.array([(1, 3), (4, 6)], dtype=[('a', float), ('c', float)])
        data = "1 2 3\n4 5 6"
        kwargs = dict(names="a, b, c")
        test = iopro.genfromtxt(StringIO(data), usecols=(0, -1), **kwargs)
        assert_equal(test, ctrl)
        test = iopro.genfromtxt(StringIO(data),
                             usecols=('a', 'c'), **kwargs)
        assert_equal(test, ctrl)

    def test_empty_file(self):
        "Test that an empty file raises the proper warning."
        warn_ctx = WarningManager()
        warn_ctx.__enter__()
        try:
            warnings.filterwarnings("ignore", message="genfromtxt: Empty input file:")
            data = StringIO()
            test = iopro.genfromtxt(data)
            assert_equal(test, np.array([]))
        finally:
            warn_ctx.__exit__()

    def test_user_filling_values(self):
        "Test with missing and filling values"
        ctrl = np.array([(0, 3), (4, -999)], dtype=[('a', int), ('b', int)])
        data = "N/A, 2, 3\n4, ,???"
        kwargs = dict(delimiter=",",
                      dtype=int,
                      names="a,b,c",
                      missing_values={0:"N/A", 'b':" ", 2:"???"},
                      filling_values={0:0, 'b':0, 2:-999})
        test = iopro.genfromtxt(StringIO(data), **kwargs)
        ctrl = np.array([(0, 2, 3), (4, 0, -999)],
                        dtype=[(_, int) for _ in "abc"])
        assert_equal(test, ctrl)
        test = iopro.genfromtxt(StringIO(data), usecols=(0, -1), **kwargs)
        ctrl = np.array([(0, 3), (4, -999)], dtype=[(_, int) for _ in "ac"])
        assert_equal(test, ctrl)

    # JNB: masked arrays not supported yet
    @unittest.expectedFailure
    def test_with_masked_column_uniform(self):
        "Test masked column"
        data = StringIO('1 2 3\n4 5 6\n')
        test = iopro.genfromtxt(data, dtype=None,
                             missing_values='2,5', usemask=True)
        control = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [0, 1, 0]])
        assert_equal(test, control)

    # JNB: masked arrays not supported yet
    @unittest.expectedFailure
    def test_with_masked_column_various(self):
        "Test masked column"
        data = StringIO('True 2 3\nFalse 5 6\n')
        test = iopro.genfromtxt(data, dtype=None,
                             missing_values='2,5', usemask=True)
        control = ma.array([(1, 2, 3), (0, 5, 6)],
                           mask=[(0, 1, 0), (0, 1, 0)],
                           dtype=[('f0', bool), ('f1', bool), ('f2', int)])
        assert_equal(test, control)

    # JNB: We'll let this one be for now, since there's an easy work around
    # and it's kind of silly anyway.
    @unittest.expectedFailure
    def test_replace_space(self):
        "Test the 'replace_space' option"
        txt = "A.A, B (B), C:C\n1, 2, 3.14"
        # Test default: replace ' ' by '_' and delete non-alphanum chars
        test = iopro.genfromtxt(StringIO(txt),
                             delimiter=",", names=True, dtype=None)
        ctrl_dtype = [("AA", int), ("B_B", int), ("CC", float)]
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        # Test: no replace, no delete
        test = iopro.genfromtxt(StringIO(txt),
                             delimiter=",", names=True, dtype=None,
                             replace_space='', deletechars='')
        ctrl_dtype = [("A.A", int), ("B (B)", int), ("C:C", float)]
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        # Test: no delete (spaces are replaced by _)
        test = iopro.genfromtxt(StringIO(txt),
                             delimiter=",", names=True, dtype=None,
                             deletechars='')
        ctrl_dtype = [("A.A", int), ("B_(B)", int), ("C:C", float)]
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        assert_equal(test, ctrl)

    def test_names_auto_completion(self):
        "Make sure that names are properly completed"
        data = "1 2 3\n 4 5 6"
        test = iopro.genfromtxt(StringIO(data),
                             dtype=(int, float, int), names="a")
        ctrl = np.array([(1, 2, 3), (4, 5, 6)],
                        dtype=[('a', int), ('f1', float), ('f2', int)])
        assert_equal(test, ctrl)

    def test_names_with_usecols_bug1636(self):
        "Make sure we pick up the right names w/ usecols"
        data = "A,B,C,D,E\n0,1,2,3,4\n0,1,2,3,4\n0,1,2,3,4"
        ctrl_names = ("A", "C", "E")
        test = iopro.genfromtxt(StringIO(data),
                             dtype=(int, int, int), delimiter=",",
                             usecols=(0, 2, 4), names=True)
        assert_equal(test.dtype.names, ctrl_names)
        #
        test = iopro.genfromtxt(StringIO(data),
                             dtype=(int, int, int), delimiter=",",
                             usecols=("A", "C", "E"), names=True)
        assert_equal(test.dtype.names, ctrl_names)
        #
        test = iopro.genfromtxt(StringIO(data),
                             dtype=int, delimiter=",",
                             usecols=("A", "C", "E"), names=True)
        assert_equal(test.dtype.names, ctrl_names)

    def test_gft_using_filename(self):
        # Test that we can load data from a filename as well as a file object
        wanted = np.arange(6).reshape((2,3))
        if sys.version_info[0] >= 3:
            # python 3k is known to fail for '\r'
            linesep = ('\n', '\r\n')
        else:
            linesep = ('\n', '\r\n', '\r')

        for sep in linesep:
            data = '0 1 2' + sep + '3 4 5'
            f, name = mkstemp()
            # We can't use NamedTemporaryFile on windows, because we cannot
            # reopen the file.
            try:
                os.write(f, asbytes(data))
                assert_array_equal(iopro.genfromtxt(name), wanted)
            finally:
                os.close(f)
                os.unlink(name)

    def test_gft_using_generator(self):
        def count():
            for i in range(10):
                yield "%d" % i

        res = iopro.genfromtxt(count())
        assert_array_equal(res, np.arange(10))


def test_gzip_loadtxt():
    # Thanks to another windows brokeness, we can't use
    # NamedTemporaryFile: a file created from this function cannot be
    # reopened by another open call. So we first put the gzipped string
    # of the test reference array, write it to a securely opened file,
    # which is then read from by the loadtxt function
    s = BytesIO()
    g = gzip.GzipFile(fileobj=s, mode='wb')
    g.write(asbytes('1 2 3\n'))
    g.close()
    s.seek(0)

    f, name = mkstemp(suffix='.gz')
    try:
        os.write(f, s.read())
        s.close()
        assert_array_equal(iopro.loadtxt(name), [1, 2, 3])
    finally:
        os.close(f)
        os.unlink(name)

def test_gzip_loadtxt_from_string():
    s = StringIO()
    f = gzip.GzipFile(fileobj=s, mode="w")
    f.write(asbytes('1 2 3\n'))
    f.close()
    s.seek(0)

    f = gzip.GzipFile(fileobj=s, mode="r")
    assert_array_equal(iopro.loadtxt(f), [1, 2, 3])

def run(verbosity=1):
    suite= unittest.TestSuite()
    for key, value in TestLoadTxt.__dict__.items():
        if key[0:4] == 'test':
            #print key
            suite.addTest(TestLoadTxt(key))
    for key, value in TestFromTxt.__dict__.items():
        if key[0:4] == 'test':
            #print key
            suite.addTest(TestFromTxt(key))
    return unittest.TextTestRunner(verbosity=verbosity).run(suite)

if __name__ == '__main__':
    run()
