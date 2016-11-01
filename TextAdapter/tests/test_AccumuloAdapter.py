import iopro
import numpy as np
import pytest
from pytest import config

# Accumulo test data was generated from iopro/tests/setup_accumulo_data.py

def server():
    return config.option.acc_host

def username():
    return config.option.acc_user

def password():
    return config.option.acc_password

def test_connection():
    with pytest.raises(IOError):
        adapter = iopro.AccumuloAdapter(server='bad hostname',
                                        username=username(),
                                        password=password(),
                                        table='ints')
    with pytest.raises(IOError):
        adapter = iopro.AccumuloAdapter(server=server(),
                                        username='bad username',
                                        password=password(),
                                        table='ints')
    with pytest.raises(IOError):
        adapter = iopro.AccumuloAdapter(server=server(),
                                        username=username(),
                                        password=password(),
                                        table='bad table name')

def test_uints():
    adapter = iopro.AccumuloAdapter(server=server(),
                                    username=username(),
                                    password=password(),
                                    table='uints',
                                    field_type='u8')
    result = adapter[:]
    expected = np.arange(100000, dtype='u8')
    np.testing.assert_array_equal(expected, result)

def test_ints():
    adapter = iopro.AccumuloAdapter(server=server(),
                                    username=username(),
                                    password=password(),
                                    table='ints',
                                    field_type='i8')
    result = adapter[:]
    expected = np.arange(-50000, 50000, dtype='i8')
    np.testing.assert_array_equal(expected, result)

def test_floats():
    adapter = iopro.AccumuloAdapter(server=server(),
                                    username=username(),
                                    password=password(),
                                    table='floats')
    result = adapter[:]
    expected = np.arange(-50000, 50000, dtype='f8') + 0.5
    np.testing.assert_almost_equal(expected, result)

def test_strings():
    adapter = iopro.AccumuloAdapter(server=server(),
                                    username=username(),
                                    password=password(),
                                    table='strings',
                                    field_type='S10')
    result = adapter[:]
    expected = np.empty(100000, dtype='S10')
    for i in range(100000):
        expected[i] = 'xxx' + str(i)
    np.testing.assert_equal(expected, result)

def test_slicing():
    adapter = iopro.AccumuloAdapter(server=server(),
                                    username=username(),
                                    password=password(),
                                    table='uints',
                                    field_type='u8')
    result = adapter[0:10000]
    expected = np.arange(10000, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    result = adapter[9990:10010]
    expected = np.arange(9990, 10010, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    result = adapter[10000:]
    expected = np.arange(10000, 100000, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    result = adapter[:]
    expected = np.arange(100000, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    result = adapter[0:10000:2]
    expected = np.arange(0, 10000, 2, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    result = adapter[10000:50000:2]
    expected = np.arange(10000, 50000, 2, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    result = adapter[0:10000:3]
    expected = np.arange(0, 10000, 3, dtype='u8')
    np.testing.assert_array_equal(expected, result)

def test_start_stop_keys():
    adapter = iopro.AccumuloAdapter(server=server(),
                                    username=username(),
                                    password=password(),
                                    table='uints',
                                    field_type='u8')
    adapter.start_key = 'row000010'
    result = adapter[:]
    expected = np.arange(10, 100000, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    adapter.start_key_inclusive = False
    result = adapter[:]
    expected = np.arange(11, 100000, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    adapter.start_key = None
    adapter.stop_key = 'row000020'
    result = adapter[:]
    expected = np.arange(20, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    adapter.stop_key_inclusive = True
    result = adapter[:]
    expected = np.arange(21, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    adapter.start_key = 'row000010'
    adapter.stop_key = 'row000020'
    adapter.start_key_inclusive = True
    adapter.stop_key_inclusive = False
    result = adapter[:]
    expected = np.arange(10, 20, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    adapter.start_key = 'row000010'
    adapter.stop_key = 'row000015'
    result = adapter[0:10]
    expected = np.arange(10, 15, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    adapter.start_key = 'row000010'
    adapter.stop_key = 'row000020'
    result = adapter[5:20]
    expected = np.arange(15, 20, dtype='u8')
    np.testing.assert_array_equal(expected, result)

    adapter.start_key = 'row000010'
    adapter.start_key_inclusive = False
    adapter.stop_key = 'row000020'
    adapter.stop_key_inclusive = True
    result = adapter[:]
    expected = np.arange(11, 21, dtype='u8')
    np.testing.assert_array_equal(expected, result)

def test_missing_data():
    adapter = iopro.AccumuloAdapter(server=server(),
                                    username=username(),
                                    password=password(),
                                    table='missing_data',
                                    field_type='u8')
    adapter.missing_values = ['NA', 'nan']
    adapter.fill_value = 999
    result = adapter[:]
    expected = np.arange(12, dtype='u8')
    expected[expected % 2 == 0] = 999
    expected[expected % 3 == 0] = 999
    np.testing.assert_array_equal(expected, result)

    adapter = iopro.AccumuloAdapter(server=server(),
                                    username=username(),
                                    password=password(),
                                    table='missing_data',
                                    field_type='f8')
    adapter.missing_values = ['NA', 'nan']
    adapter.fill_value = np.NaN
    result = adapter[:]
    expected = np.array([np.NaN, 1.0, np.NaN, np.NaN, np.NaN, 5.0,
                         np.NaN, 7.0, np.NaN, np.NaN, np.NaN, 11.0], dtype='f8')
    np.testing.assert_array_equal(expected, result)

    adapter = iopro.AccumuloAdapter(server=server(),
                                    username=username(),
                                    password=password(),
                                    table='missing_data',
                                    field_type='S10')
    adapter.missing_values = None
    adapter.fill_value = None
    result = adapter[:]
    expected = np.array(['NA', '000001', 'NA', 'nan', 'NA', '000005', 'NA',
                         '000007', 'NA', 'nan', 'NA', '000011'], dtype='S10')
    np.testing.assert_array_equal(expected, result)
