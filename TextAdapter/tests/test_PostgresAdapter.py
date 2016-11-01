from iopro import PostgresAdapter
import pandas as pd
import numpy as np
import string
import pytest
import sys
from pandas.util.testing import assert_frame_equal
from collections import OrderedDict

# PostgreSQL test data was generated from iopro/tests/setup_postgresql_data.py

# create large random number of records for cast test data, for stress testing
CASTS_TEST_NUM_RECORDS = 23456

@pytest.fixture(scope='module')
def postgres(request):

    class Dummy(object):

        def __init__(self, host, dbname, user):
            self.host = host
            self.dbname = dbname
            self.user = user

        def url(self):
            return 'host={0} dbname={1} user={2}'.format(self.host, self.dbname, self.user)

    postgresql = Dummy(request.config.option.pg_host,
                       request.config.option.pg_dbname,
                       request.config.option.pg_user)
    return postgresql

def test_connection(postgres):
    with pytest.raises(IOError) as excinfo:
        adapter = PostgresAdapter("bad URI", table='ints_test')

def test_ints(postgres):
    adapter = PostgresAdapter(postgres.url(), table='ints_test')
    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min,
                          np.iinfo(np.int64).min),
                         (0, 0, 0),
                         (np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('int2'), str('i2')),(str('int4'), str('i4')),(str('int8'), str('i8'))])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

def test_serial(postgres):
    adapter = PostgresAdapter(postgres.url(), table='serial_test')
    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min,
                          np.iinfo(np.int64).min),
                         (0, 0, 0),
                         (np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('int2'), str('i2')),(str('int4'), str('i4')),(str('int8'), str('i8'))])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

@pytest.mark.xfail(sys.version_info <= (3,),
    reason='Possible bug in libpq under Python 2')
def test_floats(postgres):
    adapter = PostgresAdapter(postgres.url(), 'floats_test')
    expected = np.array([(np.finfo(np.float32).min,
                          np.finfo(np.float64).min),
                         (0.0, 0.0),
                         (-1.1, 1.1)],
                        dtype=[(str('float4'), str('f4')),(str('float8'), str('f8'))])
    result = adapter[:]

    # JNB: There is currently a bug in NumPy that prevents two record arrays
    # containing float types from being compared with assert_array_almost_equal,
    # so just compare each column and dtype for now.
    np.testing.assert_array_almost_equal(expected[str('float4')], result[str('float4')])
    np.testing.assert_array_almost_equal(expected['float8'], result['float8'])
    assert(expected.dtype == result.dtype)

@pytest.mark.xfail(reason='Not sure about best way to convert infinite precision '
                          'postgres decimal data to Python decimal objects. '
                          'Also not quite sure about the format of the postgres '
                          'decimal data.')
def test_numeric(postgres):
    adapter = PostgresAdapter(postgres.url(), 'numeric_test')
    expected = np.array([(1234567890.01789, 1234567890.0123456789)],
                        dtype=[(str('numeric1'), 'O'),(str('numeric2'), 'O')])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

def test_fixed_strings(postgres):
    adapter = PostgresAdapter(postgres.url(), 'fixed_strings_test')
    expected = np.array([('aaa       ',),
                         ('bbb       ',),
                         ('ccc       ',)],
                        dtype=[(str('fixed'), 'U10')])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

    adapter.field_types = ['U2']
    result = adapter[:]
    expected = np.array([('aa',),
                         ('bb',),
                         ('cc',)],
                        dtype=[(str('fixed'), 'U2')])
    np.testing.assert_array_equal(expected, result)

    adapter.field_types = ['U']
    result = adapter[:]
    expected = np.array([('aaa       ',),
                         ('bbb       ',),
                         ('ccc       ',)],
                        dtype=[(str('fixed'), 'U10')])
    np.testing.assert_array_equal(expected, result)

def test_var_strings(postgres):
    adapter = PostgresAdapter(postgres.url(), 'var_strings_test')
    expected = np.array([('aaa', string.ascii_lowercase),
                         ('bbb', string.ascii_uppercase),
                         ('ccc', string.ascii_letters)],
                        dtype=[(str('varchar'), 'U10'), (str('text'), 'O')])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

    adapter.field_types = ['U1', 'O']
    expected = np.array([('a', string.ascii_lowercase),
                         ('b', string.ascii_uppercase),
                         ('c', string.ascii_letters)],
                        dtype=[(str('varchar'), 'U1'), (str('text'), 'O')])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

    adapter.field_types = ['U', 'O']
    expected = np.array([('aaa', string.ascii_lowercase),
                         ('bbb', string.ascii_uppercase),
                         ('ccc', string.ascii_letters)],
                        dtype=[(str('varchar'), 'U10'), (str('text'), 'O')])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

def test_unicode_strings(postgres):
    adapter = PostgresAdapter(postgres.url(), 'unicode_strings_test')
    expected = np.array([(u'\u4242xxx      ', u'xxx\u4242')],
                        dtype=[(str('fixed'), 'U10'), (str('text'), 'O')])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

@pytest.mark.skipif(sys.version_info <= (3,),
    reason="No support for Unicode dtype field names in NumPy for Python 2")
def test_unicode_table_name(postgres):
    adapter = PostgresAdapter(postgres.url(), 'unicode_table_name_test')
    expected = np.array([('foo',)], dtype=[('name\u4242', 'O')])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

def test_geometric_types(postgres):
    adapter = PostgresAdapter(postgres.url(), 'geometric_test')
    adapter.field_shapes = {'path': 5}
    adapter.field_names = {4: 'path2'}
    expected = np.array([((1.1, 2.2),
                          [1, 2, 3],
                          [1, 2, 3, 4],
                          [3, 4, 1, 2],
                          [(1, 2), (3, 4), (5, 6), (0, 0), (0, 0)],
                          [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
                          [1, 2, 3])],
                        dtype=[(str('point'), 'f8', 2),
                               (str('line'), 'f8', 3),
                               (str('lseg'), 'f8', 4),
                               (str('box'), 'f8', 4), 
                               (str('path2'), 'f8', (5, 2)),
                               (str('polygon'), 'O'),
                               (str('circle'), 'f8', 3)])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

    adapter.field_shapes = {'path2': 5}
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

    adapter.field_shapes = {4: 5}
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

def test_slicing(postgres):
    adapter = PostgresAdapter(postgres.url(), table='ints_test')

    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min,
                          np.iinfo(np.int64).min)],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[0]
    np.testing.assert_array_equal(expected, result)

    expected = np.array([(0, 0, 0)],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[1]
    np.testing.assert_array_equal(expected, result)

    expected = np.array([(np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[-1]
    np.testing.assert_array_equal(expected, result)

    expected = np.array([(0, 0, 0),
                         (np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[1:]
    np.testing.assert_array_equal(expected, result)

    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min,
                          np.iinfo(np.int64).min),
                         (0, 0, 0)],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[:-1]
    np.testing.assert_array_equal(expected, result)

    expected = np.array([(0, 0, 0)],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[1:2]
    np.testing.assert_array_equal(expected, result)

    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min,
                          np.iinfo(np.int64).min),
                         (0, 0, 0),
                         (np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

    result = adapter[0:10]
    np.testing.assert_array_equal(expected, result)

    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min,
                          np.iinfo(np.int64).min),
                         (np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[::2]
    np.testing.assert_array_equal(expected, result)

    expected = np.array([(np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max),
                         (np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min,
                          np.iinfo(np.int64).min)],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[::-2]
    np.testing.assert_array_equal(expected, result)

    expected = np.array([(np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[2:1:-2]
    np.testing.assert_array_equal(expected, result)

    expected = np.array([],
                        dtype=[(str('int2'), 'i2'),(str('int4'), 'i4'),(str('int8'), 'i8')])
    result = adapter[0:2:-1]
    np.testing.assert_array_equal(expected, result)

    result = adapter[2:0:1]
    np.testing.assert_array_equal(expected, result)

    result = adapter[1:1]
    np.testing.assert_array_equal(expected, result)

    result = adapter[1:1:-1]
    np.testing.assert_array_equal(expected, result)

def test_field_filter(postgres):
    adapter = PostgresAdapter(postgres.url(), 'ints_test', field_filter=['int2', 'int8'])
    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int64).min),
                         (0, 0),
                         (np.iinfo(np.int16).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('int2'), str('i2')), (str('int8'), str('i8'))])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

    adapter = PostgresAdapter(postgres.url(), 'ints_test', field_filter=[])
    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min,
                          np.iinfo(np.int64).min),
                         (0, 0, 0),
                         (np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('int2'), str('i2')), (str('int4'), str('i4')), (str('int8'), str('i8'))])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

def test_field_names(postgres):
    adapter = PostgresAdapter(postgres.url(), 'ints_test')

    adapter.field_names = ['a', 'b', 'c']
    assert(adapter.field_names == ['a', 'b', 'c'])

    adapter.field_names = {1: 'b'}
    assert(adapter.field_names == ['int2', 'b', 'int8'])

    adapter.field_names = ['a', 'b', 'c']
    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min,
                          np.iinfo(np.int64).min),
                         (0, 0, 0),
                         (np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('a'), str('i2')), (str('b'), str('i4')), (str('c'), str('i8'))])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

    adapter = PostgresAdapter(postgres.url(), 'ints_test', field_filter=['int2', 'int4'])
    adapter.field_names = ['a', 'b']
    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min),
                         (0, 0),
                         (np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max)],
                        dtype=[(str('a'), str('i2')), (str('b'), str('i4'))])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

    adapter = PostgresAdapter(postgres.url(), 'ints_test', field_filter=['int2'])
    with pytest.raises(ValueError):
        adapter.field_names = ['a', 'b']

    adapter = PostgresAdapter(postgres.url(), 'ints_test')
    adapter.field_names = {0: 'a'}
    expected = np.array([(np.iinfo(np.int16).min,
                          np.iinfo(np.int32).min,
                          np.iinfo(np.int64).min),
                         (0, 0, 0),
                         (np.iinfo(np.int16).max,
                          np.iinfo(np.int32).max,
                          np.iinfo(np.int64).max)],
                        dtype=[(str('a'), str('i2')), (str('int4'), str('i4')), (str('int8'), str('i8'))])
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

def test_field_types(postgres):
    adapter = PostgresAdapter(postgres.url(), 'casts_test')
    adapter.field_names = ['a', 'b', 'c']

    assert(adapter.field_types == ['U10', 'i4', 'f8'])

    adapter.field_types = ['i4', 'f4', 'U10']
    assert(adapter.field_types == ['i4', 'f4', 'U10'])

    adapter.field_types = {'a': 'i4'}
    assert(adapter.field_types == ['i4', 'i4', 'f8'])

    adapter.field_types = {1: 'f8'}
    assert(adapter.field_types == ['U10', 'f8', 'f8'])

    adapter.field_types = ['i4', 'f4', 'U10']
    expected = np.zeros((CASTS_TEST_NUM_RECORDS,), dtype=[(str('a'), str('i4')), (str('b'), str('f4')), (str('c'), str('U10'))])
    for i in range(CASTS_TEST_NUM_RECORDS):
        expected[i] = (i, i, float('{0}.{0}'.format(i)))
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

    adapter.field_types = {'a': 'f4'}
    expected = np.zeros((CASTS_TEST_NUM_RECORDS,), dtype=[(str('a'), str('f4')), (str('b'), str('i4')), (str('c'), str('f8'))])
    for i in range(CASTS_TEST_NUM_RECORDS):
        expected[i] = (i, i, float('{0}.{0}'.format(i)))
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)

def test_dataframe(postgres):
    adapter = PostgresAdapter(postgres.url(), table='ints_test', dataframe=True)
    expected = pd.DataFrame({'int2': np.array([np.iinfo(np.int16).min, 0, np.iinfo(np.int16).max], dtype='i2'),
                             'int4': np.array([np.iinfo(np.int32).min, 0, np.iinfo(np.int32).max], dtype='i4'),
                             'int8': np.array([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], dtype='i8')})
    result = adapter[:]
    np.testing.assert_array_equal(expected, result)
    
    adapter = PostgresAdapter(postgres.url(), 'casts_test', dataframe=True)
    expected = np.zeros((CASTS_TEST_NUM_RECORDS,), dtype=[(str('char'), str('O')),
                                                          (str('int4'), str('i4')),
                                                          (str('float8'), str('f8'))])
    for i in range(CASTS_TEST_NUM_RECORDS):
        expected[i] = (str(i).ljust(10), i, float('{0}.{0}'.format(i)))
    expected = pd.DataFrame.from_records(expected, index=np.arange(CASTS_TEST_NUM_RECORDS, dtype='u8'))
    result = adapter[:]
    assert_frame_equal(expected, result)

    adapter = PostgresAdapter(postgres.url(), 'casts_test', dataframe=True, field_filter=['int4', 'float8'])
    adapter.field_types = ['i2', 'f4']
    adapter.field_names = ['a', 'b']
    expected = np.zeros((CASTS_TEST_NUM_RECORDS,), dtype=[(str('a'), str('i2')), (str('b'), str('f4'))])
    for i in range(CASTS_TEST_NUM_RECORDS):
        expected[i] = (i, float('{0}.{0}'.format(i)))
    expected = pd.DataFrame.from_records(expected, index=np.arange(CASTS_TEST_NUM_RECORDS, dtype='u8'))
    result = adapter[:]
    assert_frame_equal(expected, result)

    adapter.field_types = {'a': 'f4'}
    expected = np.zeros((CASTS_TEST_NUM_RECORDS,), dtype=[(str('a'), str('f4')), (str('b'), str('f8'))])
    for i in range(CASTS_TEST_NUM_RECORDS):
        expected[i] = (i, float('{0}.{0}'.format(i)))
    expected = pd.DataFrame.from_records(expected, index=np.arange(CASTS_TEST_NUM_RECORDS, dtype='u8'))
    result = adapter[:]
    assert_frame_equal(expected, result)

    adapter = PostgresAdapter(postgres.url(),
                              'geometric_test',
                              dataframe=True,
                              field_filter=['point', 'line', 'polygon'])
    result = adapter[:]
    point_data = np.empty(1, dtype='O')
    point_data[0] = [1.1, 2.2]
    line_data = np.empty(1, dtype='O')
    line_data[0] = [1.0, 2.0, 3.0]
    polygon_data = np.empty(1, dtype='O')
    polygon_data[0] = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    expected = pd.DataFrame(OrderedDict([('point', point_data),
                                         ('line', line_data),
                                         ('polygon', polygon_data)]),
                                        index=np.array([0], dtype='u8'))
    assert_frame_equal(expected, result)

    adapter = PostgresAdapter(postgres.url(), 'fixed_strings_test', dataframe=True)
    result = adapter[:]
    expected = pd.DataFrame(['aaa       ', 'bbb       ', 'ccc       '],
                            columns=['fixed'],
                            index=np.array([0, 1, 2], dtype='u8'))
    assert_frame_equal(expected, result)

    with pytest.raises(RuntimeError):
        adapter.field_shapes = {'fixed': 2}

def test_missing_values(postgres):
    # Don't test missing values for PostGIS types for now. Since PostGIS type metadata
    # is stored by postgresql as actual data in the record, an empty or missing
    # value in a PostGIS column contains no metadata about what type it
    # actually is (and postgresql doesn't know about GIS types so doesn't
    # store that column metadata anywhere). In order to handle missing data
    # for PostGIS types, we'll probably need to come up with some sort of
    # generic PostGIS object or dtype which can be set to NULL for ***REMOVED***
    adapter = PostgresAdapter(postgres.url(), table='missing_values_test')
    adapter.field_shapes = {'path': 2}
    result = adapter[:]
    expected = np.array([('', 0, np.nan, [np.nan, np.nan], [(np.nan, np.nan), (np.nan, np.nan)], [])],
        dtype=[(str('char'), str('U5')),
               (str('int4'), str('i4')),
               (str('float4'), str('f4')),
               (str('point'), str('f8'), 2),
               (str('path'), str('f8'), (2, 2)),
               (str('polygon'), str('O'))])
    assert expected.dtype == result.dtype
    assert result[0][0] == ''
    assert result[0][1] == 0
    assert np.isnan(result[0][2])
    assert np.isnan(result[0][3][0])
    assert np.isnan(result[0][3][1])
    assert np.isnan(result[0][4][0][0])
    assert np.isnan(result[0][4][0][1])
    assert np.isnan(result[0][4][1][0])
    assert np.isnan(result[0][4][1][1])
    assert len(result[0][5]) == 0

def test_empty_table(postgres):
    adapter = PostgresAdapter(postgres.url(), table='empty_test')
    result = adapter[:]
    expected = np.array([], dtype=[(str('dummy'), str('i4'))])
    np.testing.assert_array_equal(expected, result)

def test_points(postgres):
    adapter = PostgresAdapter(postgres.url(),
        query='select point2d, point3d, point4d from points')
    result = adapter[:]
    expected = np.array([('POINT (0.000000 1.000000)',
                          'POINT (0.000000 1.000000 2.000000)',
                          'POINT (0.000000 1.000000 2.000000 3.000000)')],
                        dtype=[('point2d', 'O'),
                               ('point3d', 'O'),
                               ('point4d', 'O')])
    np.testing.assert_array_equal(expected, result)

    adapter.field_types = ['f8', 'O', 'f8']
    result = adapter[:]
    expected = np.array([([0.0, 1.0],
                          'POINT (0.000000 1.000000 2.000000)',
                          [0.0, 1.0, 2.0, 3.0])],
                        dtype=[('point2d', 'f8', (2,)),
                               ('point3d', 'O'),
                               ('point4d', 'f8', (4,))])
    np.testing.assert_array_equal(expected, result)

def test_multipoints(postgres):
    adapter = PostgresAdapter(postgres.url(),
        query='select point2d, point3d, point4d from multipoints')
    adapter.field_shapes = {'point2d': 1, 'point3d': 4}
    result = adapter[:]
    expected = np.array([([[0, 1]],
                          [[0, 1, 2], [3, 4, 5], [0, 0, 0], [0, 0, 0]],
                          'MULTIPOINT ((0.000000 1.000000 2.000000 3.000000), '
                                      '(4.000000 5.000000 6.000000 7.000000))')],
                        dtype=[('point2d', 'f8', (1, 2)),
                               ('point3d', 'f8', (4, 3)),
                               ('point4d', 'O')])
    np.testing.assert_array_equal(expected, result)

def test_lines(postgres):
    adapter = PostgresAdapter(postgres.url(),
        query='select line2d, line3d, line4d from lines')
    adapter.field_shapes = {'line2d': 1, 'line3d': 3}
    result = adapter[:]
    expected = np.array([([[0.0, 1.0]],
                          [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 0.0, 0.0]],
                          'LINESTRING (0.000000 1.000000 2.000000 3.000000, '
                                      '4.000000 5.000000 6.000000 7.000000)'),
                         ([[0.0, 1.0]],
                          [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                          'LINESTRING (0.000000 1.000000 2.000000 3.000000, '
                                      '4.000000 5.000000 6.000000 7.000000)')],
                        dtype=[('line2d', 'f8', (1,2)),
                               ('line3d', 'f8', (3,3)),
                               ('line4d', 'O')])
    np.testing.assert_array_equal(expected, result)
    
def test_multilines(postgres):
    adapter = PostgresAdapter(postgres.url(),
        query='select line2d, line3d, line4d from multilines')
    adapter.field_shapes = {'line3d': (2, 3), 'line4d': (2, 2)}
    result = adapter[:]
    expected = np.array([('MULTILINESTRING ((0.000000 1.000000, 2.000000 3.000000), '
                                           '(4.000000 5.000000, 6.000000 7.000000))',
                          [[[0, 1, 2], [3, 4, 5], [0, 0, 0]], [[6, 7, 8], [9, 10, 11], [12, 13, 14]]],
                          [[(0, 1, 2, 3), (4, 5, 6, 7)], [(8, 9, 10, 11), (12, 13, 14, 15)]])],
                        dtype=[('line2d', 'O'),
                               ('line3d', 'f8', (2, 3, 3)),
                               ('line4d', 'f8', (2, 2, 4))])
    np.testing.assert_array_equal(expected, result)
    
def test_polygons(postgres):
    adapter = PostgresAdapter(postgres.url(),
        query='select polygon2d, polygon3d, polygon4d from polygons')
    adapter.field_shapes = {'polygon3d': (4, 5), 'polygon4d': (3, 4)}
    result = adapter[:]
    expected = np.array([('POLYGON ((0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                   '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                   '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000))',
                          [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                           [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                           [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                          [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3]],
                           [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3]],
                           [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3]]])],
                        dtype=[('polygon2d', 'O'),
                               ('polygon3d', 'f8', (4, 5, 3)),
                               ('polygon4d', 'f8', (3, 4, 4))])
    np.testing.assert_array_equal(expected, result)

def test_multipolygons(postgres):
    adapter = PostgresAdapter(postgres.url(),
        query='select polygon2d, polygon3d, polygon4d from multipolygons')
    adapter.field_shapes = {'polygon3d': (2, 4, 5), 'polygon4d': (2, 3, 4)}
    result = adapter[:]
    expected = np.array([('MULTIPOLYGON (((0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                         '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                         '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000)), '
                                        '((0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                         '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                         '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000)))',
                          [[[[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                           [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                           [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                          [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                           [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                           [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
                          [[[(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)],
                           [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)],
                           [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)]],
                          [[(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)],
                           [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)],
                           [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)]]])],
                        dtype=[('polygon2d', 'O'),
                               ('polygon3d', 'f8', (2, 4, 5, 3)),
                               ('polygon4d', 'f8', (2, 3, 4, 4))])
    np.testing.assert_array_equal(expected, result)

def test_dataframe_gis(postgres):
    adapter = PostgresAdapter(postgres.url(), dataframe=True, table='points')
    result = adapter[:]
    expected = pd.DataFrame(np.array([('POINT (0.000000 1.000000)',
                                       'POINT (0.000000 1.000000 2.000000)',
                                       'POINT (0.000000 1.000000 2.000000 3.000000)')],
                            dtype=[('point2d', 'O'),
                                   ('point3d', 'O'),
                                   ('point4d', 'O')]))
    np.testing.assert_array_equal(expected, result)

    adapter = PostgresAdapter(postgres.url(), dataframe=True, table='multipoints')
    result = adapter[:]
    expected = pd.DataFrame(np.array([('MULTIPOINT ((0.000000 1.000000), (2.000000 3.000000))',
                                       'MULTIPOINT ((0.000000 1.000000 2.000000), (3.000000 4.000000 5.000000))',
                                       'MULTIPOINT ((0.000000 1.000000 2.000000 3.000000), '
                                                   '(4.000000 5.000000 6.000000 7.000000))')],
                            dtype=[('point2d', 'O'),
                                   ('point3d', 'O'),
                                   ('point4d', 'O')]))
    np.testing.assert_array_equal(expected, result)
