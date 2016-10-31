from __future__ import division
import numpy
cimport numpy
from libcpp cimport bool
from libcpp.string cimport string
import warnings
import math

numpy.import_array()

cdef extern from '_stdint.h':
    # Actual type lengths are defined in _stdint.h
    # Sizes here are just place holders
    ctypedef unsigned long long uint64_t
    ctypedef unsigned int uint32_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned char uint8_t
    ctypedef long long int64_t
    ctypedef int int32_t
    ctypedef short int16_t
    ctypedef char int8_t

cdef extern from "accumulo_adapter.h":
    ctypedef enum AdapterError:
        ADAPTER_SUCCESS
        ADAPTER_SUCCESS_TRUNCATION
        ADAPTER_ERROR_INVALID_SEEK
        ADAPTER_ERROR_OUTPUT_TYPE
        ADAPTER_ERROR_OUTPUT_TYPE_SIZE
        ADAPTER_ERROR_EOF
        ADAPTER_ERROR_INVALID_TABLE_NAME
        ADAPTER_ERROR_INT_CONVERSION
        ADAPTER_ERROR_FLOAT_CONVERSION
        ADAPTER_ERROR_SOCKET
        ADAPTER_ERROR_LOGIN
        ADAPTER_ERROR_TABLE_NAME
    ctypedef enum FieldType:
        UINT_FIELD,
        INT_FIELD,
        FLOAT_FIELD,
        STR_FIELD
    ctypedef struct accumulo_adapter_t:
        FieldType field_type
        int output_type_size
        string start_key
        bool start_key_inclusive
        string stop_key
        bool stop_key_inclusive
    accumulo_adapter_t * open_accumulo_adapter(string server,
                                               int port,
                                               string username,
                                               string password,
                                               string table,
                                               AdapterError *error)
    void close_accumulo_adapter(accumulo_adapter_t *adapter)
    void add_missing_value(accumulo_adapter_t *adapter, string value)
    void clear_missing_values(accumulo_adapter_t *adapter)
    void set_fill_value(accumulo_adapter_t *adapter, void *value, int size)
    void clear_fill_value(accumulo_adapter_t *adapter)
    AdapterError seek_record(accumulo_adapter_t *, int)
    AdapterError read_records(accumulo_adapter_t *, int, int, void *, int *)

# Size of buffer when reading an indeterminate number of records
DEFAULT_BUFFER_SIZE = 1000

cdef class AccumuloAdapter:
    """
    Accumulo Adapter for reading data from Accumulo database into NumPy arrays
    or Pandas dataframes.
    
    Constructor Inputs:
        server: Accumulo server address
        port: Accumulo port
        username: Accumulo user name
        password: Accumulo user password
        table: Accumulo table to read data from
        field_type: str, NumPy dtype to interpret table values as
        start_key: str, key of record where scanning will start from
        stop_key: str, key of record where scanning will stop at
        start_key_inclusive: If True, start_key is inclusive (default is True)
        stop_key_inclusive: If True, stop_key is inclusive (default is False)
        missing_values: list, missing value strings. Any values in table equal
                        to one of these strings will be replaced with fill_value.
        fill_value: fill value used to replace missing value when scanning
    """
    cdef accumulo_adapter_t *_adapter
    cdef object _field_type
    cdef object _missing_values
    cdef object _fill_value

    def __cinit__(self,
                  server='localhost',
                  port=42424,
                  username='',
                  password='',
                  table=None,
                  field_type='f8',
                  start_key=None,
                  stop_key=None,
                  start_key_inclusive=True,
                  stop_key_inclusive=False,
                  missing_values=None,
                  fill_value=None):
        """
        Accumulo Adapter constructor.
        See above for constructor argument descriptions.
        """
        if table is None:
            raise ValueError('Table name is required')

        # JNB: force ascii encoding until I figure out whether Accumulo
        # supports unicode, and if so, how to get correct encoding.
        cdef AdapterError error = ADAPTER_SUCCESS
        self._adapter = open_accumulo_adapter(server.encode('ascii'),
                                              port,
                                              username.encode('ascii'),
                                              password.encode('ascii'),
                                              table.encode('ascii'),
                                              &error)
        if self._adapter == NULL:
            if error == ADAPTER_ERROR_SOCKET:
                raise IOError('Unable to connect to Accumulo server')
            elif error == ADAPTER_ERROR_LOGIN:
                raise IOError('Invalid Accumulo username or password')
            elif error == ADAPTER_ERROR_TABLE_NAME:
                raise IOError('Invalid Accumulo table name')
            else:
                raise IOError('Error connecting to Accumulo server')

        self._adapter.output_type_size = 0

        dtype = numpy.dtype(field_type)
        if dtype.kind == 'u':
            self._adapter.field_type = UINT_FIELD
        elif dtype.kind == 'i':
            self._adapter.field_type = INT_FIELD
        elif dtype.kind == 'f':
            self._adapter.field_type = FLOAT_FIELD
        elif dtype.kind == 'S':
            self._adapter.field_type = STR_FIELD
        else:
            raise ValueError('Output field type {0} not supported'.format(field_type))

        self._adapter.output_type_size = dtype.itemsize
        self._field_type = field_type

        self.set_missing_values(missing_values)

        if fill_value is not None:
            self.fill_value = fill_value
        if dtype.kind == 'f':
            self.fill_value = numpy.nan
        elif dtype.kind in ['u', 'i']:
            self.fill_value = 0

        self.set_start_key(start_key)
        self.set_stop_key(stop_key)
        self.set_start_key_inclusive(start_key_inclusive)
        self.set_stop_key_inclusive(stop_key_inclusive)

    def close(self):
        """
        Close Accumulo connection
        """
        if self._adapter != NULL:
            close_accumulo_adapter(self._adapter)
        self._adapter = NULL

    def __dealloc__(self):
        """
        Accumulo Adapter destructor
        """
        self.close()

    def _check_connection(self):
        if self._adapter == NULL:
            raise RuntimeError('Connection already closed. '
                               'Please create a new adapter.')

    @property
    def field_type(self):
        self._check_connection()
        """
        Get NumPy dtype string for output NumPy array
        """
        return self._field_type

    def __getitem__(self, index):
        self._check_connection()
        
        if isinstance(index, (int, long)):
            return self._to_array(start=index, stop=index+1)
        elif isinstance(index, slice):
            return self._to_array(index.start, index.stop, index.step)
        else:
            raise ValueError('invalid slice')

    def get_start_key(self):
        """
        Get/set key of record where reading/scanning will start from.
        The start_key_inclusive attribute specifies whether this key is inclusive.
        """
        self._check_connection()
        return self._adapter.start_key.decode('ascii')

    def set_start_key(self, key):
        self._check_connection()
        if key is None:
            key = ''
        self._adapter.start_key = key.encode('ascii')

    start_key = property(get_start_key, set_start_key)

    def get_stop_key(self):
        self._check_connection()
        """
        Get/set key of record where reading/scanning will stop.
        The stop_key_inclusive attribute specifies whether this key is inclusive.
        """
        return self._adapter.stop_key.decode('ascii')

    def set_stop_key(self, key):
        self._check_connection()
        if key is None:
            key = ''
        self._adapter.stop_key = key.encode('ascii')

    stop_key = property(get_stop_key, set_stop_key)

    def get_start_key_inclusive(self):
        self._check_connection()
        """
        Toggle whether start key is inclusive. Default is true.
        """
        return self._adapter.start_key_inclusive

    def set_start_key_inclusive(self, inclusive):
        self._check_connection()
        self._adapter.start_key_inclusive = inclusive

    start_key_inclusive = property(get_start_key_inclusive, set_start_key_inclusive)

    def get_stop_key_inclusive(self):
        """
        Toggle whether stop key is inclusive. Default is False.
        """
        self._check_connection()
        return self._adapter.stop_key_inclusive

    def set_stop_key_inclusive(self, inclusive):
        self._check_connection()
        self._adapter.stop_key_inclusive = inclusive

    stop_key_inclusive = property(get_stop_key_inclusive, set_stop_key_inclusive)

    def set_missing_values(self, missing_values):
        self._check_connection()
        if missing_values is None or len(missing_values) == 0:
            clear_missing_values(self._adapter)
            self._missing_values = None
        else:
            for m in missing_values:
                add_missing_value(self._adapter, m.encode('ascii'))
            self._missing_values = missing_values

    def get_missing_values(self):
        """
        Get/Set missing value strings. Any values in Accumulo table equal
        to one of these strings will be replaced with fill_value.
        """
        self._check_connection()
        return self._missing_values

    missing_values = property(get_missing_values, set_missing_values)

    def set_fill_value(self, fill_value):
        cdef numpy.ndarray carray
        self._check_connection()
        if fill_value is None:
            clear_fill_value(self._adapter)
        else:
            carray = numpy.array([fill_value], self._field_type)
            if carray.dtype.kind == 'O':
                raise ValueError('Invalid fill value')
            set_fill_value(self._adapter, carray.data, carray.itemsize)
        self._fill_value == fill_value

    def get_fill_value(self):
        """
        Fill value used to replace missing_values
        """
        self._check_connection()
        return self._fill_value

    fill_value = property(get_fill_value, set_fill_value)

    def _to_array(self, start=0, stop=None, step=1):
        """
        Read Accumulo table values into NumPy array
        Inputs:
            start: record index to start reading from
            stop: record index to stop reading at
            step: number of records to skip between reads
        """
        self._check_connection()
        start_key = self._adapter.start_key.decode('ascii')
        stop_key = self._adapter.stop_key.decode('ascii')

        cdef numpy.ndarray carray
        cdef int num_records_found
        cdef int dummy
        truncation = False

        if self._field_type is None:
            raise RuntimeError('Field type must be set before reading records')

        if start is None:
            start = 0
        if step is None:
            step = 1
        if start < 0:
            raise ValueError('seeking from end of table not supported')
        if step == 0:
            raise ValueError('slice step cannot be zero')
        if step < 0:
            raise ValueError('reading records in reverse not supported')
        if stop is not None and stop <= start:
            return numpy.array([], dtype=self._field_type)

        # Reinitialize accumulo iterator to point to beginning of table/query
        result = seek_record(self._adapter, start)
        if result == ADAPTER_ERROR_INVALID_SEEK:
            raise ValueError('Invalid start record')
        elif result == ADAPTER_ERROR_INVALID_TABLE_NAME:
            raise ValueError('Invalid table name')

        total_num_records_found = 0
        carray = numpy.ndarray(0, dtype=self.field_type)

        if stop is not None:
            # Round up since we read the first record of each step, so it doesn't
            # matter if, for example, we can only fit 3.5 steps in the full
            # range of records we want to read - the first record of last 0.5 step
            # counts as the 4th record we can read.
            max_num_records = int(math.ceil((stop - start) / step))

        while stop is None or total_num_records_found < max_num_records:
            if stop is None:
                num_records = DEFAULT_BUFFER_SIZE
            else:
                num_records = max_num_records - total_num_records_found
                if num_records > DEFAULT_BUFFER_SIZE:
                    num_records = DEFAULT_BUFFER_SIZE

            # Resize output array to make room for another buffer of records,
            # or, if first time through and the total number of records is known,
            # allocate final array size.
            carray.resize(total_num_records_found + num_records)

            num_records_found = 0
            offset = total_num_records_found * self._adapter.output_type_size
            result = read_records(self._adapter, num_records, step, carray.data + <int>offset, &num_records_found)
            if result == ADAPTER_SUCCESS_TRUNCATION:
                # set truncation flag so we can throw exception later
                truncation = True
            elif result in [ADAPTER_ERROR_INT_CONVERSION, ADAPTER_ERROR_FLOAT_CONVERSION]:
                raise RuntimeError("Unable to convert table values to '{0}' dtype".format(self.field_type))
            elif result != ADAPTER_SUCCESS and result != ADAPTER_ERROR_EOF:
                raise RuntimeError('Invalid record read')
            total_num_records_found += num_records_found
            if ((stop is None and num_records_found < num_records) or
                    result == ADAPTER_ERROR_EOF):
                # we've found the last records, so break out of read loop
                break

            # Seek so that next read starts on step boundary.
            # Do a read with no output since a call to seek_record creates a
            # new scanner and seeks from beginning of table.
            result = read_records(self._adapter, step - 1, 1, NULL, &dummy)
            if result == ADAPTER_ERROR_EOF:
                # we've found the last record, so break out of read loop
                break

        # Throw away any extra space
        if carray.size > total_num_records_found:
            carray.resize(total_num_records_found)

        if truncation:
            warnings.warn('{0} records successfully read, but at least one '
                          'result was truncated to fit in the specified '
                          'NumPy dtype'.format(total_num_records_found))
        return numpy.asarray(carray)        
