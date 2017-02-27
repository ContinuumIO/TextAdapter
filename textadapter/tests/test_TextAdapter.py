#!/usr/bin/python

import sys
import textadapter
import unittest
from .generate import (generate_dataset, IntIter,
                       MissingValuesIter, FixedWidthIter)
import numpy as np
from numpy.testing import assert_array_equal
import gzip
import os
import io
from six import StringIO

class TestTextAdapter(unittest.TestCase):
    num_records = 100000

    def assert_equality(self, left, right):
        try:
            if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                self.assert_array_equal(left, right)
            else:
                self.assertTrue(left == right)
        except AssertionError:
            raise AssertionError('FAIL: {0} != {1}'.format(left, right))

    # Basic parsing tests
    def test_string_parsing(self):
        data = StringIO('1,2,3\n')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.set_field_types({0:'S5', 1:'S5', 2:'S5'})
        assert_array_equal(adapter[:], np.array([('1', '2', '3')], dtype='S5,S5,S5'))

        data = io.StringIO(u'1,2,3\n')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.set_field_types({0:'S5', 1:'S5', 2:'S5'})
        assert_array_equal(adapter[:], np.array([('1', '2', '3')], dtype='S5,S5,S5'))

        data = io.BytesIO(b'1,2,3\n')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.set_field_types({0:'S5', 1:'S5', 2:'S5'})
        assert_array_equal(adapter[:], np.array([('1', '2', '3')], dtype='S5,S5,S5'))

    # basic utf_8 tests
    def test_utf8_parsing(self):
        # test single byte character
        data = io.BytesIO(u'1,2,\u0033'.encode('utf_8'))
        adapter = textadapter.text_adapter(data, field_names=False)
        expected = np.array([('1', '2', '3')], dtype='u8,u8,u8')
        assert_array_equal(adapter[:], expected)

        # test multibyte character
        data = io.BytesIO(u'1,2,\u2092'.encode('utf_8'))
        adapter = textadapter.text_adapter(data, field_names=False)
        expected = np.array([('1', '2', u'\u2092')], dtype='u8,u8,O')
        assert_array_equal(adapter[:], expected)

    def test_no_whitespace_stripping(self):
        data = StringIO('1  ,2  ,3  \n')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.set_field_types({0:'S3', 1:'S3', 2:'S3'})
        assert_array_equal(adapter[:], np.array([('1  ', '2  ', '3  ')], dtype='S3,S3,S3'))

        data = StringIO('  1,  2,  3\n')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.set_field_types({0:'S3', 1:'S3', 2:'S3'})
        assert_array_equal(adapter[:], np.array([('  1', '  2', '  3')], dtype='S3,S3,S3'))

        data = StringIO('  1  ,  2  ,  3  \n')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.set_field_types({0:'S5', 1:'S5', 2:'S5'})
        assert_array_equal(adapter[:], np.array([('  1  ', '  2  ', '  3  ')], dtype='S5,S5,S5'))

        data = StringIO('\t1\t,\t2\t,\t3\t\n')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.set_field_types({0:'S3', 1:'S3', 2:'S3'})
        assert_array_equal(adapter[:], np.array([('\t1\t', '\t2\t', '\t3\t')], dtype='S3,S3,S3'))

    def test_quoted_whitespace(self):
        data = StringIO('"1  ","2  ","3  "\n')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.set_field_types({0:'S3', 1:'S3', 2:'S3'})
        assert_array_equal(adapter[:], np.array([('1  ', '2  ', '3  ')], dtype='S3,S3,S3'))

        data = StringIO('"\t1\t"\t"\t2\t"\t"\t3\t"\n')
        adapter = textadapter.text_adapter(data, field_names=False, delimiter='\t')
        adapter.set_field_types({0:'S3', 1:'S3', 2:'S3'})
        assert_array_equal(adapter[:], np.array([('\t1\t', '\t2\t', '\t3\t')], dtype='S3,S3,S3'))

    def test_fixed_simple(self):
        # TODO: fix this test on 32-bit and on Windows
        if tuple.__itemsize__ == 4:
            # This test does not work on 32-bit, so we skip it
            return
        if sys.platform == 'win32':
            # This test does not work on Windows
            return
        data = StringIO("  1  2  3\n  4  5 67\n890123  4")
        adapter = textadapter.FixedWidthTextAdapter(data, 3, infer_types=False, field_names=False)
        adapter.set_field_types({0:'i', 1:'i', 2:'i'})

        control = np.array([(1, 2, 3), (4, 5, 67), (890, 123, 4)], dtype='i,i,i')
        assert_array_equal(adapter[:], control)

    def test_spaces_around_numeric_values(self):
        data = StringIO(' 1 , -2 , 3.3 , -4.4 \n  5  ,  -6  ,  7.7 , -8.8 ')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.set_field_types({0:'u4', 1:'i8', 2:'f4', 3:'f8'})
        array = adapter[:]

        control = np.array([(1,-2,3.3,-4.4), (5,-6,7.7,-8.8)], dtype='u4,i8,f4,f8')
        assert_array_equal(array, control)

    def test_slicing(self):
        data = StringIO()
        generate_dataset(data, IntIter(), ',', self.num_records)
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.set_field_types({0:'u4',1:'u4',2:'u4',3:'u4',4:'u4'})

        assert_array_equal(adapter[0], np.array([(0, 1, 2, 3, 4)], dtype='u4,u4,u4,u4,u4'))
        expected_values = [((self.num_records-1)*5)+x for x in range(5)]
        self.assert_equality(adapter[self.num_records-1].item(), tuple(expected_values))

        #adapter.create_index()
        #self.assert_equality(adapter[-1].item(), tuple(expected_values))

        self.assert_equality(adapter['f0'][0].item(), (0,))
        self.assert_equality(adapter['f4'][1].item(), (9,))
        #self.assert_equality(adapter[self.num_records-1]['f4'], (self.num_records*5)-1)

        array = adapter[:]
        record = [x for x in range(0, 5)]
        self.assert_equality(array.size, self.num_records)
        for i in range(0, self.num_records):
            self.assert_equality(array[i].item(), tuple(record))
            record = [x+5 for x in record]

        array = adapter[:-1]
        record = [x for x in range(0, 5)]
        self.assert_equality(array.size, self.num_records-1)
        for i in range(0, self.num_records-1):
            self.assert_equality(array[i].item(), tuple(record))
            record = [x+5 for x in record]

        array = adapter[0:10]
        self.assert_equality(array.size, 10)
        record = [x for x in range(0, 5)]
        for i in range(0, 10):
            self.assert_equality(array[i].item(), tuple(record))
            record = [x+5 for x in record]

        array = adapter[1:]
        self.assert_equality(array.size, self.num_records-1)
        record = [x for x in range(5, 10)]
        for i in range(0, self.num_records-1):
            self.assert_equality(array[i].item(), tuple(record))
            record = [x+5 for x in record]

        array = adapter[0:10:2]
        self.assert_equality(array.size, 5)
        record = [x for x in range(0, 5)]
        for i in range(0, 5):
            self.assert_equality(array[i].item(), tuple(record))
            record = [x+10 for x in record]

        array = adapter[['f0', 'f4']][:]
        record = [0, 4]
        self.assert_equality(array.size, self.num_records)
        for i in range(0, self.num_records):
            self.assert_equality(array[i].item(), tuple(record))
            record = [x+5 for x in record]

        adapter.field_filter = [0, 'f4']
        array = adapter[:]
        record = [0, 4]
        self.assert_equality(array.size, self.num_records)
        for i in range(0, self.num_records):
            self.assert_equality(array[i].item(), tuple(record))
            record = [x+5 for x in record]

        adapter.field_filter = None
        array = adapter[:]
        record = [0, 1, 2, 3, 4]
        self.assert_equality(array.size, self.num_records)
        for i in range(0, self.num_records):
            self.assert_equality(array[i].item(), tuple(record))
            record = [x+5 for x in record]

        try:
            adapter[self.num_records]
        except textadapter.AdapterIndexError:
            pass
        else:
            self.fail('AdaperIndexError not thrown')

        try:
            adapter[0:self.num_records+1]
        except textadapter.AdapterIndexError:
            pass
        else:
            self.fail('AdaperIndexError not thrown')


    def test_converters(self):
        data = StringIO()
        generate_dataset(data, IntIter(), ',', self.num_records)
        adapter = textadapter.text_adapter(data, delimiter=',', field_names=False)
        #adapter.set_field_types({0:'u4', 1:'u4', 2:'u4', 3:'u4', 4:'u4'})

        def increment(input_str):
            return int(input_str) + 1

        def double(input_str):
            return int(input_str) + int(input_str)

        if sys.platform == 'win32' and tuple.__itemsize__ == 8:
            # TODO: there problems below here 64-bit Windows, I get
            # OverflowError: can't convert negative value to unigned PY_LONG_LONG
            return

        adapter.set_converter(0, increment)
        adapter.set_converter('f1', double)

        array = adapter[:]

        self.assert_equality(array.size, self.num_records)

        record = [1, 2, 2, 3, 4]
        for i in range(0, self.num_records):
            self.assert_equality(array[i].item(), tuple(record))
            record[0] += 5
            record[1] = (10 * (i+1)) + 2
            record[2] += 5
            record[3] += 5
            record[4] += 5

    def test_missing_fill_values(self):
        data = StringIO()
        generate_dataset(data, MissingValuesIter(), ',', self.num_records)

        adapter = textadapter.text_adapter(data, delimiter=',', field_names=False, infer_types=False)
        adapter.set_field_types({'f0':'u4', 1:'u4', 2:'u4', 3:'u4', 'f4':'u4'})
        adapter.set_missing_values({0:['NA', 'NaN'], 'f4':['xx','inf']})
        adapter.set_fill_values({0:99, 4:999})

        array = adapter[:]

        self.assert_equality(array.size, self.num_records)

        record = [x for x in range(0, 5)]
        for i in range(0, self.num_records):
            if i % 4 == 0 or i % 4 == 1:
                record[0] = 99
                record[4] = 999
            else:
                record[0] = record[1] - 1
                record[4] = record[3] + 1
            self.assert_equality(array[i].item(), tuple(record))
            record[1] += 5
            record[2] += 5
            record[3] += 5

        data.seek(0)
        adapter = textadapter.text_adapter(data, delimiter=',', field_names=False, infer_types=True)
        adapter.set_missing_values({0:['NA', 'NaN'], 4:['xx','inf']})

        array = adapter[:]

        self.assert_equality(array.size, self.num_records)

        record = [x for x in range(0, 5)]
        for i in range(0, self.num_records):
            if i % 4 == 0 or i % 4 == 1:
                record[0] = 0
                record[4] = 0
            else:
                record[0] = record[1] - 1
                record[4] = record[3] + 1
            self.assert_equality(array[i].item(), tuple(record))
            record[1] += 5
            record[2] += 5
            record[3] += 5

        # Test missing field
        data = StringIO('1,2,3\n4,5\n7,8,9')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.field_types = {0:'O', 1:'O', 2:'O'}
        adapter.set_fill_values({0:np.nan, 1:np.nan, 2:np.nan})
        array = adapter[:]

        # NumPy assert_array_equal no longer supports mixed O/nan types
        expected = [('1','2','3'),('4','5',np.nan),('7','8','9')]
        self.assert_equality(array.tolist(), expected)

    def test_fixed_width(self):
        data = StringIO()
        generate_dataset(data, FixedWidthIter(), '', self.num_records)
        adapter = textadapter.FixedWidthTextAdapter(data, [2,3,4,5,6], field_names=False, infer_types=False)
        adapter.set_field_types({0:'u4',1:'u4',2:'u4',3:'u4',4:'u4'})

        array = adapter[:]

        self.assert_equality(array.size, self.num_records)

        record = [0, 0, 0, 0, 0]
        for i in range(0, self.num_records):
            self.assert_equality(array[i].item(), tuple(record))
            record = [x+1 for x in record]
            if record[0] == 100:
                record[0] = 0
            if record[1] == 1000:
                record[1] = 0
            if record[2] == 10000:
                record[2] = 0
            if record[3] == 100000:
                record[3] = 0
            if record[4] == 1000000:
                record[4] = 0

        # Test skipping blank lines
        data = StringIO(' 1 2 3\n\n 4 5 6')
        adapter = textadapter.text_adapter(data, parser='fixed_width',
            field_widths=[2,2,2], field_names=False)
        array = adapter[:]
        assert_array_equal(array, np.array([(1,2,3), (4,5,6)],
            dtype=[('f0','<u8'),('f1','<u8'),('f2','<u8')]))

        # Test comment lines
        data = StringIO('# 1 2 3\n 1 2 3\n# foo\n 4 5 6')
        adapter = textadapter.text_adapter(data, parser='fixed_width',
            field_widths=[2,2,2], field_names=False)
        array = adapter[:]
        assert_array_equal(array, np.array([(1,2,3), (4,5,6)],
            dtype=[('f0','<u8'),('f1','<u8'),('f2','<u8')]))

        # Test field names line
        data = StringIO(' a b c\n 1 2 3')
        adapter = textadapter.text_adapter(data, parser='fixed_width',
            field_widths=[2,2,2], field_names=True)
        array = adapter[:]
        assert_array_equal(array, np.array([(1,2,3)],
            dtype=[('a','<u8'),('b','<u8'),('c','<u8')]))

        # Test field names line as comment line
        data = StringIO('# a b c\n 1 2 3')
        adapter = textadapter.text_adapter(data, parser='fixed_width',
            field_widths=[2,2,2], field_names=True)
        array = adapter[:]
        assert_array_equal(array, np.array([(1,2,3)],
            dtype=[('a','<u8'),('b','<u8'),('c','<u8')]))

        # Test incomplete field names line
        data = StringIO(' a\n 1 2 3')
        adapter = textadapter.text_adapter(data, parser='fixed_width',
            field_widths=[2,2,2], field_names=True)
        array = adapter[:]
        assert_array_equal(array, np.array([(1,2,3)],
            dtype=[('a','<u8'),('f1','<u8'),('f2','<u8')]))

    def test_regex(self):
        data = StringIO()
        generate_dataset(data, IntIter(), ',', self.num_records)
        adapter = textadapter.RegexTextAdapter(data, '([0-9]*),([0-9]*),([0-9]*),([0-9]*),([0-9]*)\n', field_names=False, infer_types=False)
        adapter.set_field_types({0:'u4',1:'u4',2:'u4',3:'u4',4:'u4'})

        array = adapter[:]

        self.assert_equality(array.size, self.num_records)

        record = [x for x in range(0, 5)]
        for i in range(0, self.num_records):
            self.assert_equality(array[i].item(), tuple(record))
            record = [x+5 for x in record]

        # Test skipping blank lines
        data = StringIO('1 2 3\n\n4 5 6')
        adapter = textadapter.text_adapter(data, parser='regex',
            regex_string='([0-9]) ([0-9]) ([0-9])', field_names=False)
        array = adapter[:]
        assert_array_equal(array, np.array([(1,2,3), (4,5,6)],
            dtype=[('f0','<u8'),('f1','<u8'),('f2','<u8')]))

        # Test comment lines
        data = StringIO('#1 2 3\n1 2 3\n# foo\n4 5 6')
        adapter = textadapter.text_adapter(data, parser='regex',
            regex_string='([0-9]) ([0-9]) ([0-9])', field_names=False)
        array = adapter[:]
        assert_array_equal(array, np.array([(1,2,3), (4,5,6)],
            dtype=[('f0','<u8'),('f1','<u8'),('f2','<u8')]))

        # Test field names line
        data = StringIO('a b c\n4 5 6')
        adapter = textadapter.text_adapter(data, parser='regex',
            regex_string='([0-9]) ([0-9]) ([0-9])', field_names=True)
        array = adapter[:]
        assert_array_equal(array, np.array([(4,5,6)],
            dtype=[('a','<u8'),('b','<u8'),('c','<u8')]))

        # Test field names line as comment line
        data = StringIO('#a b c\n4 5 6')
        adapter = textadapter.text_adapter(data, parser='regex',
            regex_string='([0-9]) ([0-9]) ([0-9])', field_names=True)
        array = adapter[:]
        assert_array_equal(array, np.array([(4,5,6)],
            dtype=[('a','<u8'),('b','<u8'),('c','<u8')]))

        # Test incomplete field names line
        data = StringIO('a b\n4 5 6')
        adapter = textadapter.text_adapter(data, parser='regex',
            regex_string='([0-9]) ([0-9]) ([0-9])', field_names=True)
        array = adapter[:]
        assert_array_equal(array, np.array([(4,5,6)],
            dtype=[('a','<u8'),('b','<u8'),('f2','<u8')]))

        # Test field names line that doesn't match regex
        data = StringIO('a b c\n1 2  3 4  5 6')
        adapter = textadapter.text_adapter(data, parser='regex',
            regex_string='([0-9\s]+)  ([0-9\s]+)  ([0-9\s]+)', field_names=True)
        array = adapter[:]
        assert_array_equal(array, np.array([('1 2', '3 4', '5 6')],
            dtype=[('a','O'),('b','O'),('c','O')]))

    def test_index(self):
        if sys.platform == 'win32':
            # TODO: this test fails on Windows because of file lock problems
            return

        num_records = 100000
        expected_values = [((num_records-1)*5) + x for x in range(5)]

        data = StringIO()
        generate_dataset(data, IntIter(), ',', num_records)

        # test explicit index building
        adapter = textadapter.text_adapter(data, delimiter=',', field_names=False, infer_types=False)
        adapter.set_field_types({0:'u4',1:'u4',2:'u4',3:'u4',4:'u4'})
        adapter.create_index()

        self.assert_equality(adapter[0].item(), tuple([(0*5) + x for x in range(5)]))
        self.assert_equality(adapter[10].item(), tuple([(10*5) + x for x in range(5)]))
        self.assert_equality(adapter[100].item(), tuple([(100*5) + x for x in range(5)]))
        self.assert_equality(adapter[1000].item(), tuple([(1000*5) + x for x in range(5)]))
        self.assert_equality(adapter[10000].item(), tuple([(10000*5) + x for x in range(5)]))
        self.assert_equality(adapter[num_records - 1].item(), tuple([((num_records - 1)*5) + x for x in range(5)]))
        #self.assert_equality(adapter[-1].item(), tuple(expected_values))

        # test implicitly creating disk index on the fly
        if os.path.exists('test.idx'):
            os.remove('test.idx')
        data.seek(0)
        adapter = textadapter.text_adapter(data, delimiter=',', field_names=False, infer_types=False, index_name='test.idx')
        adapter.set_field_types({0:'u4',1:'u4',2:'u4',3:'u4',4:'u4'})
        adapter.to_array()

        self.assert_equality(adapter[0].item(), tuple([(0*5) + x for x in range(5)]))
        self.assert_equality(adapter[10].item(), tuple([(10*5) + x for x in range(5)]))
        self.assert_equality(adapter[100].item(), tuple([(100*5) + x for x in range(5)]))
        self.assert_equality(adapter[1000].item(), tuple([(1000*5) + x for x in range(5)]))
        self.assert_equality(adapter[10000].item(), tuple([(10000*5) + x for x in range(5)]))
        self.assert_equality(adapter[num_records - 1].item(), tuple([((num_records - 1)*5) + x for x in range(5)]))
        #self.assert_equality(adapter[-1].item(), tuple(expected_values))

        adapter.close()

        # test loading disk index
        data.seek(0)
        adapter2 = textadapter.text_adapter(data, delimiter=',', field_names=False, infer_types=False, index_name='test.idx')
        adapter2.set_field_types({0:'u4',1:'u4',2:'u4',3:'u4',4:'u4'})

        self.assert_equality(adapter2[0].item(), tuple([(0*5) + x for x in range(5)]))
        self.assert_equality(adapter2[10].item(), tuple([(10*5) + x for x in range(5)]))
        self.assert_equality(adapter2[100].item(), tuple([(100*5) + x for x in range(5)]))
        self.assert_equality(adapter2[1000].item(), tuple([(1000*5) + x for x in range(5)]))
        self.assert_equality(adapter2[10000].item(), tuple([(10000*5) + x for x in range(5)]))
        self.assert_equality(adapter2[num_records - 1].item(), tuple([((num_records - 1)*5) + x for x in range(5)]))
        #self.assert_equality(adapter2[-1].item(), tuple(expected_values))

        adapter.close()

        os.remove('test.idx')

    def test_gzip_index(self):
        num_records = 1000000

        data = StringIO()
        generate_dataset(data, IntIter(), ',', num_records)

        #if sys.version > '3':
        if True:
            dataz = io.BytesIO()
        else:
            dataz = StringIO()
        gzip_output = gzip.GzipFile(fileobj=dataz, mode='wb')
        #if sys.version > '3':
        if True:
            gzip_output.write(data.getvalue().encode('utf8'))
        else:
            gzip_output.write(data.getvalue())
        gzip_output.close()
        dataz.seek(0)

        # test explicit index building
        adapter = textadapter.text_adapter(dataz, compression='gzip', delimiter=',', field_names=False, infer_types=False)
        adapter.set_field_types({0:'u4',1:'u4',2:'u4',3:'u4',4:'u4'})
        adapter.create_index()

        self.assert_equality(adapter[0].item(), tuple([(0*5) + x for x in range(5)]))
        self.assert_equality(adapter[10].item(), tuple([(10*5) + x for x in range(5)]))
        self.assert_equality(adapter[100].item(), tuple([(100*5) + x for x in range(5)]))
        self.assert_equality(adapter[1000].item(), tuple([(1000*5) + x for x in range(5)]))
        self.assert_equality(adapter[10000].item(), tuple([(10000*5) + x for x in range(5)]))
        self.assert_equality(adapter[100000].item(), tuple([(100000*5) + x for x in range(5)]))
        self.assert_equality(adapter[num_records - 1].item(), tuple([((num_records - 1)*5) + x for x in range(5)]))
        #self.assert_equality(adapter[-1].item(), tuple(expected_values))

        # test 'trouble' records that have caused crashes in the past
        self.assert_equality(adapter[290000].item(), tuple([(290000*5) + x for x in range(5)]))
        self.assert_equality(adapter[818000].item(), tuple([(818000*5) + x for x in range(5)]))

        # test implicitly creating disk index on the fly
        # JNB: not implemented yet
        '''adapter = textadapter.text_adapter(dataz, compression='gzip', delimiter=',', field_names=False, infer_types=False, indexing=True, index_filename='test.idx')
        adapter.set_field_types({0:'u4',1:'u4',2:'u4',3:'u4',4:'u4'})
        adapter.to_array()

        self.assert_equality(adapter[0].item(), tuple([(0*5) + x for x in range(5)]))
        self.assert_equality(adapter[10].item(), tuple([(10*5) + x for x in range(5)]))
        self.assert_equality(adapter[100].item(), tuple([(100*5) + x for x in range(5)]))
        self.assert_equality(adapter[1000].item(), tuple([(1000*5) + x for x in range(5)]))
        self.assert_equality(adapter[10000].item(), tuple([(10000*5) + x for x in range(5)]))
        self.assert_equality(adapter[100000].item(), tuple([(100000*5) + x for x in range(5)]))
        self.assert_equality(adapter[num_records - 1].item(), tuple([((num_records - 1)*5) + x for x in range(5)]))
        #self.assert_equality(adapter[-1].item(), tuple(expected_values))

        # test 'trouble' records that have caused crashes in the past
        self.assert_equality(adapter[290000].item(), tuple([(290000*5) + x for x in range(5)]))
        self.assert_equality(adapter[818000].item(), tuple([(818000*5) + x for x in range(5)]))

        # test loading disk index
        adapter2 = textadapter.text_adapter(dataz, compression='gzip', delimiter=',', field_names=False, infer_types=False, indexing=True, index_filename='test.idx')
        adapter2.set_field_types({0:'u4',1:'u4',2:'u4',3:'u4',4:'u4'})

        self.assert_equality(adapter2[0].item(), tuple([(0*5) + x for x in range(5)]))
        self.assert_equality(adapter2[10].item(), tuple([(10*5) + x for x in range(5)]))
        self.assert_equality(adapter2[100].item(), tuple([(100*5) + x for x in range(5)]))
        self.assert_equality(adapter2[1000].item(), tuple([(1000*5) + x for x in range(5)]))
        self.assert_equality(adapter2[10000].item(), tuple([(10000*5) + x for x in range(5)]))
        self.assert_equality(adapter2[100000].item(), tuple([(100000*5) + x for x in range(5)]))
        self.assert_equality(adapter2[num_records - 1].item(), tuple([((num_records - 1)*5) + x for x in range(5)]))
        #self.assert_equality(adapter[-1].item(), tuple(expected_values))

        # test 'trouble' records that have caused crashes in the past
        self.assert_equality(adapter2[290000].item(), tuple([(290000*5) + x for x in range(5)]))
        self.assert_equality(adapter2[818000].item(), tuple([(818000*5) + x for x in range(5)]))

        os.remove('test.idx')'''


    def test_header_footer(self):
        data = StringIO('0,1,2,3,4\n5,6,7,8,9\n10,11,12,13,14')
        adapter = textadapter.text_adapter(data, header=1, field_names=False)
        adapter.field_types = dict(zip(range(5), ['u4']*5))
        assert_array_equal(adapter[:], np.array([(5,6,7,8,9), (10,11,12,13,14)],
            dtype='u4,u4,u4,u4,u4'))

        data.seek(0)
        adapter = textadapter.text_adapter(data, header=2, field_names=False)
        adapter.field_types = dict(zip(range(5), ['u4']*5))
        assert_array_equal(adapter[:], np.array([(10,11,12,13,14)],
            dtype='u4,u4,u4,u4,u4'))

        data.seek(0)
        adapter = textadapter.text_adapter(data, header=1, field_names=True)
        adapter.field_types = dict(zip(range(5), ['u4']*5))
        assert_array_equal(adapter[:], np.array([(10,11,12,13,14)],
            dtype=[('5','u4'),('6','u4'),('7','u4'),('8','u4'),('9','u4')]))


    def test_delimiter(self):
        data = StringIO('1,2,3\n')
        adapter = textadapter.text_adapter(data, field_names=False)
        self.assert_equality(adapter[0].item(), (1,2,3))

        data = StringIO('1 2 3\n')
        adapter = textadapter.text_adapter(data, field_names=False)
        self.assert_equality(adapter[0].item(), (1,2,3))

        data = StringIO('1\t2\t3\n')
        adapter = textadapter.text_adapter(data, field_names=False)
        self.assert_equality(adapter[0].item(), (1,2,3))

        data = StringIO('1x2x3\n')
        adapter = textadapter.text_adapter(data, field_names=False)
        self.assert_equality(adapter[0].item(), (1,2,3))

        # Test no delimiter in single field csv data
        data = StringIO('aaa\nbbb\nccc')
        array = textadapter.text_adapter(data, field_names=False, delimiter=None)[:]
        assert_array_equal(array, np.array([('aaa',), ('bbb',), ('ccc',)], dtype=[('f0', 'O')]))

    def test_auto_type_inference(self):
        data = StringIO('0,1,2,3,4\n5.5,6,7,8,9\n10,11,12,13,14a\n15,16,xxx,18,19')
        adapter = textadapter.text_adapter(data, field_names=False, infer_types=True)
        array = adapter.to_array()
        self.assert_equality(array.dtype.fields['f0'][0], np.dtype('float64'))
        self.assert_equality(array.dtype.fields['f1'][0], np.dtype('uint64'))
        self.assert_equality(array.dtype.fields['f2'][0], np.dtype('O'))
        self.assert_equality(array.dtype.fields['f3'][0], np.dtype('uint64'))
        self.assert_equality(array.dtype.fields['f4'][0], np.dtype('O'))

        data = StringIO('0,1,2,3,4\n5.5,6,7,8,9\n10,11,12,13,14a\n15,16,xxx,18,19')
        adapter = textadapter.text_adapter(data, field_names=False, infer_types=True)
        self.assert_equality(adapter[0].dtype.fields['f0'][0], np.dtype('uint64'))
        self.assert_equality(adapter[1:3].dtype.fields['f0'][0], np.dtype('float64'))
        self.assert_equality(adapter[3].dtype.fields['f4'][0], np.dtype('uint64'))
        self.assert_equality(adapter[:].dtype.fields['f3'][0], np.dtype('uint64'))
        self.assert_equality(adapter[-1].dtype.fields['f2'][0], np.dtype('O'))
        self.assert_equality(adapter[2].dtype.fields['f4'][0], np.dtype('O'))

    def test_64bit_ints(self):
        data = StringIO(str((2**63)-1) + ',' + str(((2**63)-1)*-1) + ',' + str((2**64)-1))
        adapter = textadapter.text_adapter(data, delimiter=',', field_names=False, infer_types=False)
        adapter.set_field_types({0:'i8', 1:'i8', 2:'u8'})
        array = adapter.to_array()
        self.assert_equality(array[0].item(), ((2**63)-1, ((2**63)-1)*-1, (2**64)-1))

    def test_adapter_factory(self):
        data = StringIO("1,2,3")
        adapter = textadapter.text_adapter(data, "csv", delimiter=',', field_names=False, infer_types=False)
        self.assertTrue(isinstance(adapter, textadapter.CSVTextAdapter))

        self.assertRaises(textadapter.AdapterException, textadapter.text_adapter, data, "foobar")

    def test_field_names(self):
        # Test for ignoring of extra fields
        data = StringIO('f0,f1\n0,1,2\n3,4,5')
        adapter = textadapter.text_adapter(data, 'csv', delimiter=',', field_names=True)
        array = adapter.to_array()
        self.assert_equality(array.dtype.names, ('f0', 'f1'))
        self.assert_equality(array[0].item(), (0,1))
        self.assert_equality(array[1].item(), (3,4))

        # Test for duplicate field names
        data = StringIO('f0,field,field\n0,1,2\n3,4,5')
        adapter = textadapter.text_adapter(data, 'csv', delimiter=',', field_names=True, infer_types=False)
        adapter.set_field_types({0:'u4', 1:'u4', 2:'u4'})
        array = adapter.to_array()
        self.assert_equality(array.dtype.names, ('f0', 'field', 'field1'))

        # Test for field names list
        data = StringIO('0,1,2\n3,4,5')
        adapter = textadapter.text_adapter(data, field_names=['a', 'b', 'c'], infer_types=False)
        adapter.field_types = {0:'u4', 1:'u4', 2:'u4'}
        array = adapter[:]
        self.assertTrue(array.dtype.names == ('a', 'b', 'c'))
        assert_array_equal(array, np.array([(0,1,2), (3,4,5)], dtype=[('a', 'u4'), ('b', 'u4'), ('c', 'u4')]))

    def test_float_conversion(self):
        data = StringIO('10,1.333,-1.23,10.0E+2,999.9e-2')
        adapter = textadapter.text_adapter(data, field_names=False, infer_types=False)
        adapter.set_field_types(dict(zip(range(5), ['f8']*5)))
        array = adapter[0]
        #self.assert_equality(array[0].item(), (10.0,1.333,-1.23,1000.0,9.999))
        self.assertAlmostEqual(array[0][0], 10.0)
        self.assertAlmostEqual(array[0][1], 1.333)
        self.assertAlmostEqual(array[0][2], -1.23)
        self.assertAlmostEqual(array[0][3], 1000.0)
        self.assertAlmostEqual(array[0][4], 9.999)

    def test_generators(self):
        def int_generator(num_recs):
            for i in range(num_recs):
                yield ','.join([str(i*5), str(i*5+1), str(i*5+2), str(i*5+3), str(i*5+4)])

        adapter = textadapter.text_adapter(int_generator(self.num_records), field_names=False)
        array = adapter[:]

        self.assert_equality(array.size, self.num_records)

        record = [x for x in range(0, 5)]
        for i in range(0, self.num_records):
            self.assert_equality(array[i].item(), tuple(record))
            record[0] += 5
            record[1] += 5
            record[2] += 5
            record[3] += 5
            record[4] += 5

    def test_comments(self):
        data = StringIO('1,2,3\n#4,5,6')
        adapter = textadapter.text_adapter(data, field_names=False)
        array = adapter[:]
        self.assert_equality(array.size, 1)
        self.assert_equality(array[0].item(), (1,2,3))

        data = StringIO('1,2,3\n#4,5,6')
        adapter = textadapter.text_adapter(data, field_names=False, comment=None)
        array = adapter[:]
        self.assert_equality(array.size, 2)
        self.assert_equality(array[0].item(), ('1',2,3))
        self.assert_equality(array[1].item(), ('#4',5,6))

    def test_escapechar(self):
        data = StringIO('1,2\\2,3\n4,5\\5\\5,6')
        array = textadapter.text_adapter(data, field_names=False)[:]
        assert_array_equal(array,
            np.array([(1,22,3), (4,555,6)], dtype='u8,u8,u8'))

        data = StringIO('\\1,2,3\n4,5,6\\')
        array = textadapter.text_adapter(data, field_names=False)[:]
        assert_array_equal(array,
            np.array([(1,2,3), (4,5,6)], dtype='u8,u8,u8'))

        data = StringIO('a,b\\,b,c\na,b\\,b\\,b,c')
        array = textadapter.text_adapter(data, field_names=False)[:]
        assert_array_equal(array,
            np.array([('a', 'b,b', 'c'), ('a', 'b,b,b', 'c')], dtype='O,O,O'))

        data = StringIO('a,bx,b,c\na,bx,bx,b,c')
        array = textadapter.text_adapter(data, field_names=False, escape='x')[:]
        assert_array_equal(array,
            np.array([('a', 'b,b', 'c'), ('a', 'b,b,b', 'c')], dtype='O,O,O'))

    '''def test_dataframe_output(self):

        try:
            import pandas
        except ImportError:
            return

        # Test filling blank lines with fill values if output is dataframe
        data = StringIO('1,2,3\n\n4,5,6')
        adapter = textadapter.text_adapter(data, field_names=False)
        adapter.field_types = {0:'O', 1:'O', 2:'O'}
        adapter.set_fill_values({0:np.nan, 1:np.nan, 2:np.nan})
        df = adapter.to_dataframe()'''

    def test_csv(self):
        # Test skipping blank lines
        data = StringIO('1,2,3\n\n4,5,6')
        adapter = textadapter.text_adapter(data, field_names=False)
        array = adapter[:]
        assert_array_equal(array, np.array([(1,2,3), (4,5,6)],
            dtype=[('f0','<u8'),('f1','<u8'),('f2','<u8')]))

    def test_json(self):
        # Test json number
        data = StringIO('{"id":123}')
        adapter = textadapter.text_adapter(data, parser='json')
        array = adapter[:]
        assert_array_equal(array, np.array([(123,)], dtype=[('id', 'u8')]))

        # Test json number
        data = StringIO('{"id":"xxx"}')
        adapter = textadapter.text_adapter(data, parser='json')
        array = adapter[:]
        assert_array_equal(array, np.array([('xxx',)], dtype=[('id', 'O')]))

        # Test multiple values
        data = StringIO('{"id":123, "name":"xxx"}')
        adapter = textadapter.text_adapter(data, parser='json')
        array = adapter[:]
        assert_array_equal(array, np.array([(123, 'xxx',)], dtype=[('id', 'u8'), ('name', 'O')]))

        # Test multiple records
        data = StringIO('[{"id":123, "name":"xxx"}, {"id":456, "name":"yyy"}]')
        adapter = textadapter.text_adapter(data, parser='json')
        array = adapter[:]
        assert_array_equal(array, np.array([(123, 'xxx',), (456, 'yyy')], dtype=[('id', 'u8'), ('name', 'O')]))

        # Test multiple objects separated by newlines
        data = StringIO('{"id":123, "name":"xxx"}\n{"id":456, "name":"yyy"}')
        adapter = textadapter.text_adapter(data, parser='json')
        array = adapter[:]
        assert_array_equal(array, np.array([(123, 'xxx',), (456, 'yyy')], dtype=[('id', 'u8'), ('name', 'O')]))

        data = StringIO('{"id":123, "name":"xxx"}\n')
        adapter = textadapter.text_adapter(data, parser='json')
        array = adapter[:]
        assert_array_equal(array, np.array([(123, 'xxx',)], dtype=[('id', 'u8'), ('name', 'O')]))

        # JNB: broken; should be really be supporting the following json inputs?
        '''
        # Test subarrays
        data = StringIO('{"id":123, "names":["xxx","yyy","zzz"]}')
        adapter = textadapter.text_adapter(data, parser='json')
        array = adapter[:]
        assert_array_equal(array, np.array([(123, 'xxx', 'yyy', 'zzz',)],
            dtype=[('f0', 'u8'), ('f1', 'O'), ('f2', 'O'), ('f3', 'O')]))

        # Test subobjects
        data = StringIO('{"id":123, "names":{"a":"xxx", "b":"yyy", "c":"zzz"}}')
        adapter = textadapter.text_adapter(data, parser='json')
        array = adapter[:]
        assert_array_equal(array, np.array([(123, 'xxx', 'yyy', 'zzz',)],
            dtype=[('f0', 'u8'), ('f1', 'O'), ('f2', 'O'), ('f3', 'O')]))
        '''

        # Test ranges
        data = StringIO('{"id": 1, "name": "www"}\n'
                                 '{"id": 2, "name": "xxx"}\n'
                                 '{"id": 3, "name": "yyy"}\n'
                                 '{"id": 4, "name": "zzz"}')
        adapter = textadapter.text_adapter(data, parser='json')
        array = adapter[2:4]
        assert_array_equal(array, np.array([(3, 'yyy'), (4, 'zzz')],
            dtype=[('id', 'u8'), ('name', 'O')]))

        # Test column order
        data = StringIO('{"xxx": 1, "aaa": 2}\n')
        adapter = textadapter.text_adapter(data, parser='json')
        array = adapter[:]
        assert_array_equal(array, np.array([(1, 2)],
            dtype=[('xxx', 'u8'), ('aaa', 'u8')]))

        # Test field filter
        data = StringIO('{"id": 1, "name": "www"}\n'
                                 '{"id": 2, "name": "xxx"}\n'
                                 '{"id": 3, "name": "yyy"}\n'
                                 '{"id": 4, "name": "zzz"}')
        adapter = textadapter.text_adapter(data, parser='json')
        adapter.field_filter = ['name']
        array = adapter[:]
        assert_array_equal(array, np.array([('www',), ('xxx',), ('yyy',), ('zzz',)],
            dtype=[('name', 'O')]))

    def test_stepping(self):
        data = StringIO('0,1\n2,3\n4,5\n6,7\n8,9\n10,11\n12,13\n14,15\n16,17\n18,19')
        adapter = textadapter.text_adapter(data, field_names=False)
        assert_array_equal(adapter[::2], np.array([(0,1), (4,5), (8,9), (12,13), (16,17)], dtype='u8,u8'))
        assert_array_equal(adapter[::3], np.array([(0,1), (6,7), (12,13), (18,19)], dtype='u8,u8'))

    def test_num_records(self):
        data = StringIO('0,1\n2,3\n4,5\n6,7\n8,9\n10,11\n12,13\n14,15\n16,17\n18,19')
        adapter = textadapter.text_adapter(data, field_names=False, num_records=2)
        assert_array_equal(adapter[:], np.array([(0, 1), (2, 3)], dtype='u8,u8'))


def run(verbosity=1, num_records=100000):
    if num_records < 10:
        raise ValueError('number of records for generated datasets must be at least 10')
    TestTextAdapter.num_records = num_records
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTextAdapter)
    return unittest.TextTestRunner(verbosity=verbosity).run(suite)


if __name__ == '__main__':
    run()
