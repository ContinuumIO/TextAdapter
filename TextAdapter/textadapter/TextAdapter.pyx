import numpy
import re
import sys
import io
import warnings
import os
import types
import logging
import csv
import encodings
import math
from TextAdapter.lib import errors
from six import string_types, StringIO
from cpython.ref cimport Py_INCREF

if sys.version > '3':
    from urllib.request import urlopen
else:
    from urllib import urlopen

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

try:
    import boto
    from boto.s3 import key
    boto_installed = True
except ImportError:
    boto_installed = False

try:
    import pandas
    pandas_installed = True
except ImportError:
    pandas_installed = False


# Quick hack for telling str2str_object_converter function in Converters.pyx
# how to decode strings into objects. This should probably be passed in to the
# converter function's void *args parameter.
config = {'encoding': 'utf_8'}

supported_encodings = ['ascii', 'utf_8']


# Building as separate extensions never worked right,
# so brute force into one single extension
include 'IO.pyx'
include 'Index.pyx'
include 'Converters.pyx'

cdef extern from "numpy/ndarraytypes.h":
    cdef enum requirements:
        NPY_WRITEABLE


# Since we allocate our own chunk of memory that ends up as the final numpy array,
# this object is used by the numpy array to handle deallocation properly.
# (see create_array function below).
cdef class ArrayDealloc:
    cdef void* data

    def __cinit__(self):
        self.data = NULL

    def __dealloc__(self):
        if self.data:
            free(self.data)


# Enable C api. This must be called before any
# other numpy C api methods are called.
numpy.import_array()


__all__ = (
        "text_adapter",
        "s3_text_adapter",
        "CSVTextAdapter",
        "RegexTextAdapter",
        "FixedWidthTextAdapter",
        "JSONTextAdapter",
        "AdapterException"
        )


default_missing_values = ['NA', 'NaN', 'inf', '-inf', 'None', 'none', '']

default_fill_values = {'u':0, 'i':0, 'f':numpy.nan, 'c':0, 'b':False,
                       'O':'', 'S':'', 'M':None, 'U':''}


# Decorator for converter function that takes converter function
# output and passes it to field dtype's object to make sure it's
# the proper dtype value.
cdef class ConverterDecorator(object):
    cdef object converter
    cdef object dtype

    def __cinit__(self, converter, dtype):
        self.converter = converter
        self.dtype = dtype
    def __call__(self, value):
        return self.dtype.type(self.converter(value))
    cdef getSize(self):
        return self.dtype.itemsize


cdef class TextAdapter(object):
    """
    TextAdapter objects read CSV data and output a numpy array or Pandas
    DataFrame containing the data converted to appropriate data types. The
    following features are currently implemented:

    * parse CSV data very fast using minimal memory - The TextAdapter core is
      written in C to ensure text is parsed as fast as data can be read from
      disk. Text is read and parsed in small chunks instead of reading whole
      datasets into memory at once, which enables very large datasets to be
      read and parsed without running out of memory.

    * use basic slicing notation to examine subsets of CSV records - Basic
      Numpy style slicing notation can be used to specify the subset of records
      to be read from file.

    * define records and fields by delimiter character, fixed field widths, or
      regular expression - In additional to specifying a delimiter character,
      fields can be specified by fixed field widths as well as a regular
      expression. This enables a larger variety of CSV and CSV-like text files
      to be parsed.

    * parse gzipped CSV data - Gzipped data can be parsed without having to
      uncompress it first. Parsing speed is about the same as an uncompressed
      version of same data.

    * build index of CSV records to enable random access, and save index to
      disk - An index of record offsets can be build to allow fast random
      access to records.

    * define converter functions for each field in python, or cython
      compiled function - Converter functions can be specified for converting
      parsed text to proper dtype for storing in Numpy array. Converter
      functions can be defined in Python or Cython.

    * define dtype of Numpy record array to store parsed data - As each record
      is parsed, the fields are converted to specified dtype and stored in
      Numpy struct array.
    """
    cdef text_adapter_t *adapter
    cdef object encoding
    cdef object compression
    cdef object REC_CHUNK_SIZE
    cdef object exact_index
    cdef object indexing
    cdef kh_string_t* kh_string_table
    cdef object mapping
    cdef object _field_names
    cdef object _field_filter
    cdef object fill_values
    cdef object loose_fill
    cdef object build_converter
    cdef object converter_objects
    cdef object default_output
    cdef object logger
    cdef object missing_values
    cdef object data

    def __cinit__(self):
        if type(self) == TextAdapter:
            raise errors.AdapterException('TextAdapter cannot be used directly. '
                'Use CSVTextAdapter, FixedWidthTextAdapter, RegexTextAdapter,'
                'or JSONTextAdapter')


    def __init_text_adapter(self, fh, encoding='utf_8', compression=None,
            comment='#', quote='"', num_records=0, header=0, footer=0,
            indexing=False, index_name=None, output='ndarray',
            debug=False):
        """
        Construct new TextAdapter object.

        Args:
            fh: filename, file object, StringIO object, BytesIO object, S3 key,
                http url, or python generator
            encoding: type of character encoding (current ascii and utf8 are supported)
            compression : type of data compression (currently only gzip is supported)
            comment: character used to indicate comment line
            quote: character used to quote fields
            num_records: limits parsing to specified number of records; defaults
                to all records
            header: number of lines in file header; these lines are skipped when parsing
            footer: number of lines in file footer; these lines are skipped when parsing
            indexing: create record index on the fly as characters are read
            index_name: name of file to write index to
            output: type of output object (NumPy array or Pandas DataFrame)
        """
        cdef InputData *input_data

        self.adapter = NULL
        self._field_names = None
        self.mapping = {}
        self.compression = compression
        self._field_filter = None

        # Number of numpy records to allocate at a time as more rows of text
        # are parsed. This number was arrived at after some very rough
        # benchmarking and is not in any way scientific.
        self.REC_CHUNK_SIZE = 100000

        encoding = encodings.normalize_encoding(encoding)
        if encoding in encodings.aliases.aliases.keys():
            encoding = encodings.aliases.aliases[encoding]
        if encoding not in supported_encodings:
            raise ValueError('{0} encoding not supported yet '.format(encoding))
        config['encoding'] = encoding
        self.encoding = encoding

        self.logger = logging.getLogger('TextAdapter')
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        if output == 'dataframe' and not pandas_installed:
            raise warnings.warn('Pandas module is not installed. '
                'Output type will be set to numpy array.')
            self.default_output = 'ndarray'
        else:
            self.default_output = output

        # TODO: download and parse remote text file in chunks
        if isinstance(fh, string_types) and \
                (fh[0:7] == 'http://' or fh[0:6] == 'ftp://'):
            url_socket = urlopen(fh)
            fh = io.BytesIO(url_socket.read())

        # Setup IO functions and input data.
        # Each open_* function set ups function pointers for the appropriate
        # seek, read, and close functions, and returns a pointer to an
        # InputData struct.
        if isinstance(fh, (StringIO, io.StringIO, io.BytesIO)):
            if isinstance(fh, (StringIO, io.StringIO)):
                self.data = fh.read().encode('utf_8')
            else:
                self.data = fh.read()
            input_data = open_memmap(self.data, len(self.data))
        elif boto_installed and isinstance(fh, key.Key):
            self.data['s3_key'] = fh
            self.data['offset'] = 0
            input_data = open_s3(self.data)
        elif isinstance(fh, string_types):
            input_data = open_file(fh.encode('ascii'))
            if compression is None and fh.endswith('.gz'):
                compression = 'gzip'
        elif isinstance(fh, types.GeneratorType):
            # TODO: Figure out better way to do this
            temp = [x for x in fh]
            if isinstance(temp[0], string_types):
                temp = '\n'.join(temp)
                self.data = temp.encode('utf_8')
            else:
                self.data = b'\n'.join(temp)
            input_data = open_memmap(self.data, len(self.data))
        else:
            raise TypeError('Cannot open data source of type %s '
                            'must be file name, url, StringIO, BytesIO, '
                            'or generator ' % str(type(fh)))

        if input_data == NULL:
            raise IOError('Could not initialize data source')

        # Setup read/seek functions for compressed data. Functions pointers
        # for read and seek functions are assigned to the read_compressed and
        # seek_compressed members of the passed in InputData struct. Also, the
        # compressed_buffer in the InputData struct is initialized. This is a
        # buffer for compressed raw data before it is decompressed into the
        # main buffer.
        input_data.compressed_input = NULL
        input_data.compressed_prebuffer = NULL
        if compression == 'gzip':
            init_gzip(input_data)
        elif compression != None:
            error_msg = 'unknown compression type {0}'.format(compression)
            raise errors.ArgumentError(error_msg)

        # Create and initialize low level TextAdapter C struct
        self.adapter = open_text_adapter(input_data)
        if self.adapter == NULL:
            raise IOError('Could not initialize text adapter')

        # num_records sets a hard limit on the number of records read.
        self.adapter.num_records = num_records

        self.indexing = indexing
        if indexing is False and index_name is not None:
            self.indexing = True

        if self.indexing is True:
            self.exact_index = ExactIndex(index_name=index_name,
                density = DEFAULT_INDEX_DENSITY, num_records=num_records)
            self.adapter.index = <void*>self.exact_index
            self.adapter.input_data.index = <void*>self.exact_index
            self.adapter.index_density = self.exact_index.get_density()
            self.adapter.indexer = &indexer_callback
            self.adapter.index_lookup = &index_lookup_callback
            if self.compression == 'gzip':
                self.adapter.add_gzip_access_point = &add_gzip_access_point_callback
                self.adapter.input_data.get_gzip_access_point = &get_gzip_access_point_callback
            else:
                self.adapter.add_gzip_access_point = NULL
                self.adapter.input_data.get_gzip_access_point = NULL
            self.adapter.num_records = self.exact_index.total_num_records
        else:
            self.exact_index = None
            self.adapter.index = NULL
            self.adapter.input_data.index = NULL
            self.adapter.index_density = 0
            self.adapter.indexer = NULL
            self.adapter.index_lookup = NULL
            self.adapter.add_gzip_access_point = NULL
            self.adapter.input_data.get_gzip_access_point = NULL

        # initialize number of fields to 1 because we're first going to parse
        # by line instead of field value in order to skip the header
        # (with the entire line being one field). After this base class constructor
        # returns, the child class constructor will figure out the correct
        # number of fields and call set_num_fields again.
        set_num_fields(self.adapter.fields, 1)

        # skip header lines
        offset = 0
        if header > 0:
            line_iter = create_line_iter(self.adapter)
            for i in range(header):
                offset += len(line_iter.__next__())
        self.adapter.input_data.header = offset

        self.adapter.comment_char = ord('\0')
        if comment is not None:
            self.adapter.comment_char = ord(comment[0])

        self.adapter.quote_char = 0
        if quote is not None:
            self.adapter.quote_char = ord(quote[0])

        default_converters[<unsigned int>STRING_OBJECT_CONVERTER_FUNC] = \
            <converter_func_ptr>&str2str_object_converter;
        self.build_converter = {}
        self.converter_objects = []

        # Initialize hash table for interning string objects that have been
        # created from text data. The hash table key is a string from the input text data,
        # and the hash table value is the string object to be stored in the final
        # numpy array result. Hash table implementation comes from Klib project
        # (https://github.com/attractivechaos/klib).
        self.kh_string_table = kh_init_string()


    def close(self):

        if self.indexing is True:
            self.exact_index.close()
            self.indexing = False
            self.exact_index = None
            self.adapter.index = NULL
            self.adapter.input_data.index = NULL
            self.adapter.index_density = 0
            self.adapter.indexer = NULL
            self.adapter.index_lookup = NULL
            self.adapter.add_gzip_access_point = NULL
            self.adapter.input_data.get_gzip_access_point = NULL


    def __dealloc__(self):

        # keys in string hash table are copies of input strings;
        # we still need to delete them
        if self.kh_string_table != NULL:
            for i in range(self.kh_string_table.n_buckets):
                if kh_exist(self.kh_string_table, i):
                    free(self.kh_string_table.keys[i])
            kh_destroy_string(self.kh_string_table)

        if self.adapter != NULL:
            if self.adapter.input_data.compressed_input != NULL:
                close_gzip(self.adapter.input_data)
            if self.adapter.input_data.close != NULL:
                self.adapter.input_data.close(self.adapter.input_data)
            close_text_adapter(self.adapter)


    @property
    def size(self):
        """ Returns number of records in data. """
        cdef uint64_t num_recs_read = 0
        if self.adapter.num_records > 0:
            return self.adapter.num_records
        else:
            # If number of records is not known, parse data and count records
            tokenizer = self.adapter.tokenize
            if self.adapter.tokenize == &json_tokenizer:
                self.adapter.tokenize = &json_record_tokenizer
            else:
                self.adapter.tokenize = &record_tokenizer
            num_fields = self.adapter.fields.num_fields
            self.adapter.fields.num_fields = 1

            result = seek_record(self.adapter, 0)
            if result != ADAPTER_SUCCESS:
                self.__raise_adapter_exception(result)
            result = read_records(self.adapter, UINT64_MAX, 1, NULL, &num_recs_read)

            self.adapter.tokenize = tokenizer
            self.adapter.fields.num_fields = num_fields
            self.adapter.num_records = num_recs_read
            return num_recs_read


    @property
    def field_count(self):
        """ Returns number of fields in data. """
        return self.adapter.fields.num_fields


    def get_field_filter(self):
        """ Returns list of field names or field indices that will be read. """
        return self._field_filter


    def set_field_filter(self, fields):
        """
        Set fields to read.

        Args:
<<<<<<< HEAD:iopro/textadapter/TextAdapter.pyx
<<<<<<< HEAD
            ***REMOVED*** list of field names or indices to read
=======
            fields: list of field names or indices to read
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
            fields: list of field names or indices to read
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c:TextAdapter/textadapter/TextAdapter.pyx
        """

        if fields is None or len(fields) == 0:
            self._field_filter = range(self.field_count)
        else:
            field_filter = []
<<<<<<< HEAD:iopro/textadapter/TextAdapter.pyx
<<<<<<< HEAD
            for field in ***REMOVED***
=======
            for field in fields:
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
            for field in fields:
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c:TextAdapter/textadapter/TextAdapter.pyx
                if isinstance(field, (int, long)):
                    field_filter.append(field)
                elif isinstance(field, string_types):
                    if field not in self._field_names:
                        raise IndexError(field + ' is not a valid field name')
                    index = self._field_names.index(field)
                    field_filter.append(index)
                else:
                    raise TypeError('Field must be int or string')
            self._field_filter = sorted(field_filter)
        self.logger.debug('set_field_filter() setting field filter to [{0}]'
            .format(','.join(str(x) for x in self._field_filter)))


    field_filter = property(get_field_filter, set_field_filter)


    def get_field_names(self):
        """ Returns list of field names. """
        return self._field_names

    def set_field_names(self, field_names):
        """
        Set field names

        Args:
            field_names: list of field names. A name must be supplied for every
                field in data.
        """
        if len(field_names) != self.field_count:
            raise ValueError('Number of field names ({0}) does not match number of '
                             'fields ({1}) in data source'.format(len(field_names), self.field_count))
        self._field_names = field_names
        self.logger.debug('set_field_names() setting field names to [{0}]'
            .format(','.join(str(x) for x in self._field_names)))

    field_names = property(get_field_names, set_field_names)


    def get_field_types(self):
        """ Returns dict of fields and field dtypes. """
        return self.mapping


    def set_field_types(self, types=None):
        """
        Set field dtype for each field. dtype should be in NumPy dtype format.

        Args:
            types: dict of mapping between field name or index, and field type
        """
        if self.adapter.fields.num_fields == 0:
            raise errors.ConfigurationError('Adapter has no fields specified')

        # Reset field types to type infer and default to u8
        self.mapping = {}
        for field in range(self.adapter.fields.num_fields):
            self.adapter.fields.field_info[field].infer_type = 1

        if isinstance(types, dict):
            for field, dtype in types.items():
                if isinstance(field, string_types):
                    field = self._field_names.index(field)
                numpy_dtype = numpy.dtype(dtype)
                if numpy.version.version[0:3] == '1.6' and \
                        (numpy_dtype.kind == 'M' or numpy_dtype.kind == 'm'):
                    raise TypeError('NumPy 1.6 datetime/timedelta not supported')
                self.mapping[int(field)] = numpy_dtype
                self.adapter.fields.field_info[field].infer_type = 0
        elif types is not None:
            raise TypeError('types must be dict of fields/dtypes, '
                            'or None to reset field types')
        self.logger.debug('set_field_types() setting dtypes to {0}'
            .format(','.join(str(field) + ':' + str(dtype) for field,dtype in self.mapping.iteritems())))


    field_types = property(get_field_types, set_field_types)


    def create_index(self, index_name=None, density=DEFAULT_INDEX_DENSITY):
        """
        Build index of record offsets to allow fast random access.

        Args:
            index_name: Name of file to save index to (default is None which does
                not save to disk).
            density: Density of index. Value of 1 will index every record,
                value of 2 will index every other record, etc.
        """
        cdef AdapterError result

        if index_name is not None:
            if os.path.exists(index_name):
                os.remove(index_name)

        self.indexing = True
        self.exact_index = ExactIndex(index_name=index_name, density = density)
        self.adapter.index = <void*>self.exact_index
        self.adapter.input_data.index = <void*>self.exact_index
        self.adapter.index_density = density
        self.adapter.indexer = &indexer_callback
        self.adapter.index_lookup = &index_lookup_callback
        if self.compression == 'gzip':
            self.adapter.add_gzip_access_point = &add_gzip_access_point_callback
            self.adapter.input_data.get_gzip_access_point = &get_gzip_access_point_callback
        else:
            self.adapter.add_gzip_access_point = NULL
            self.adapter.input_data.get_gzip_access_point = NULL

        result = seek_record(self.adapter, 0)
        if result != ADAPTER_SUCCESS:
            self.__raise_adapter_exception(result)

        if self.compression == 'gzip':
            build_gzip_index(self.adapter)
        else:
            build_index(self.adapter)

        self.exact_index.finalize(self.adapter.num_records)


    def set_converter(self, field, converter):
        """
        Set converter function for field.

        Args:
            field: field to apply converter function
            converter: python function object, or function pointer as longint object
        """

        if isinstance(field, (int, long)):
            if field < 0 or field > self.field_count:
                raise ValueError('Invalid field number')
        elif isinstance(field, string_types):
            if field in self._field_names:
                field = self._field_names.index(field)
            else:
                raise ValueError('Invalid field name')
        else:
            raise TypeError('field must be int in string')

        self.build_converter[field] = \
            lambda: self.__build_python_converter(field, converter)


    def __build_python_converter(self, field, converter):
        # wrap converter function with function to convert output
        # to proper dtype
        dec = ConverterDecorator(converter, self.mapping[field])
        self.converter_objects.append(dec)
        set_converter(self.adapter.fields, field, NULL,
                      self.mapping[field].itemsize,
                      <converter_func_ptr>&python_converter, <void*>dec)


    def set_missing_values(self, missing_values):
        """
        Set strings for each field that represents a missing value.

        Arguments:
        missing_values - dict, list, tuple, or comma separated list of strings
            that represent missing values. If list, tuple, or comma separated
            string,the nth entry is a list of missing value strings for the nth
            field. If dict, each key is a field number or field name, with the
            value being a list of missing value strings for that field.
        """
        if isinstance(missing_values, dict):
            self.missing_values = {}
            for key, value in missing_values.items():
                for v in value:
                    if key not in self.missing_values.keys():
                        self.missing_values[key] = []
                    self.missing_values[key].append(v.encode(self.encoding))
            for (key, values) in self.missing_values.items():
                field = key
                if isinstance(key, string_types):
                    field = self._field_names.index(key)
<<<<<<< HEAD:iopro/textadapter/TextAdapter.pyx
<<<<<<< HEAD
                if field < 0 or field >= self.adapter.fields.num_***REMOVED***
=======
                if field < 0 or field >= self.adapter.fields.num_fields:
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
                if field < 0 or field >= self.adapter.fields.num_fields:
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c:TextAdapter/textadapter/TextAdapter.pyx
                    raise errors.NoSuchFieldError('invalid field number ' + str(field))
                if isinstance(values, string_types):
                    values = [values]
                elif values is None:
                    init_missing_values(self.adapter.fields, NULL, <uint64_t>field, 0)
                    return
                if len(values) == 0:
                    error_msg = 'No missing values specified for field ' + str(field)
                    raise errors.ArgumentError(error_msg)
                init_missing_values(self.adapter.fields, NULL, <uint64_t>field,
                    <uint32_t>len(values))
                for v in values:
                    if isinstance(v, string_types):
                        v = v.encode(self.encoding)
                    elif not isinstance(v, bytes):
                        error_msg = ('Invalid missing value; must be string '
                                    '(is {0})'.format(str(type(v))))
                        raise errors.ArgumentError(error_msg)
                    add_missing_value(self.adapter.fields, NULL, <uint64_t>field,
                        <char*>v, <uint64_t>len(v))
        else:
            raise errors.ArgumentError('missing values must be dict of field '
                                       'numbers and missing value strings '
                                       '(is {0})'.format(str(type(missing_values))))


    def set_fill_values(self, fill_values, loose=False):
        """
        Set fill values for each field.

        Arguments:
        fill_values - dict, list, or tuple of values for each field, or single fill
            value to use for all fields. If list or tuple, the nth entry is a fill
            value for the nth field. If dict, each key is a field number or field
            name, with the value being a fill value used for that field.
        """
        self.fill_values = fill_values
        self.loose_fill = loose


    def __set_fill_values(self, fill_values):

        cdef numpy.ndarray carray
        cdef void *fill_value
        cdef int loose = 0

        for field in self._field_filter:

            if fill_values is None:
                value = default_fill_values[self.mapping[field].kind]
            elif field in fill_values.keys():
                value = fill_values[field]
            elif self._field_names[field] in fill_values.keys():
                value = fill_values[self._field_names[field]]
            else:
                value = default_fill_values[self.mapping[field].kind]

            if value is None:
                set_fill_value(self.adapter.fields, NULL, <uint64_t>field,
                    NULL, 0, loose)
                return

            if self.loose_fill:
                loose = 1

            a = numpy.array([value], dtype=self.mapping[field])
            carray = a
            fill_value = <void*>carray.data
            set_fill_value(self.adapter.fields, NULL, <uint64_t>field, fill_value,
                get_field_size(self.adapter.fields, NULL, field), loose)


    def to_array(self):
        """ Read all records and return numpy array """
        if infer_types(self.adapter.fields):
            array = self.__read_slice_infer_types(0, self.adapter.num_records, 1)
        else:
            array = self.__read_slice(0, self.adapter.num_records, 1)
        if self.adapter.num_records == 0:
            self.adapter.num_records = array.size
        return array


    def to_dataframe(self):
        """ Read all records and return Pandas dataframe """
        if not pandas_installed:
            raise errors.AdapterException('Pandas module is not installed.')
        return pandas.DataFrame.from_records(self.to_array())


    def __getitem__(self, index):
        """ Read records by record number or slice """

        class TextAdapterFields(object):

            def __init__(self, adapter, fields):
                self.adapter = adapter
                self.default_field_filter = self.adapter.field_filter
                self.temp_field_filter = fields

            def __getitem__(self, index):

                self.adapter.field_filter = self.temp_field_filter
                array = self.adapter[index]
                self.adapter.field_filter = self.default_field_filter
                return array

        array = numpy.empty(0)

        if isinstance(index, (int, long)):

            # If we want to read from end of file,
            # first build index of record offsets.
            if index < 0:
                # calculate index from beginning of file
                index = self.size + index
            if infer_types(self.adapter.fields):
                array = self.__read_slice_infer_types(index, index + 1, 1)
            else:
                array = self.__read_slice(index, index + 1, 1)

        elif isinstance(index, slice):

            recstart, recstop, recstep = self.__parse_slice(index)
            if recstop == 0 and self.adapter.num_records > 0:
                recstop = self.adapter.num_records

            if infer_types(self.adapter.fields):
                array = self.__read_slice_infer_types(recstart, recstop, recstep)
            else:
                array = self.__read_slice(recstart, recstop, recstep)
            if recstart == 0 and recstop == 0 and recstep == 1 and self.adapter.num_records == 0:
                self.adapter.num_records = array.size

        elif isinstance(index, string_types):
            return TextAdapterFields(self, [index])
        elif isinstance(index, (list, tuple)):
            return TextAdapterFields(self, index)
        else:
            raise IndexError('index must be int, slice, fancy indexing, or field name')

        array.shape = (-1,)

        if self.default_output == 'dataframe':
            return pandas.DataFrame.from_records(array)
        else:
            return array


    def __parse_slice(self, index):

        recstart = index.start
        recstop = index.stop
        recstep = index.step

        if recstart is None:
            recstart = 0
        else:
            if recstart < 0:
                recstart = self.size + recstart

        if recstop is None:
            recstop = 0
        else:
            if recstop < 0:
                recstop = self.size + recstop

        if recstep is None:
            recstep = 1
        else:
            if recstep < 0:
                raise NotImplementedError('reverse stepping not implemented yet')

        return (recstart, recstop, recstep)


    def __read_slice_infer_types(self, start_rec, stop_rec, step_rec):
        """ Read records by slice and infer dtypes """

        # Set default dtypes for fields not set with set_field_types
        for field in self._field_filter:
            if self.adapter.fields.field_info[field].infer_type:
                self.mapping[field] = numpy.dtype('u8')
        self.__build_converters()
        self.__set_fill_values(None)

        # Try to make a good guess about the field dtypes from first 1000 records.
        # If we're reading less than 1000 records then don't bother with guess.
        if stop_rec == 0 or (stop_rec > 0 and stop_rec - start_rec > 1000):
            result = seek_record(self.adapter, start_rec)
            if result != ADAPTER_SUCCESS and result != ADAPTER_ERROR_READ_EOF:
                self.__raise_adapter_exception(result)
            self.__infer_types(start_rec, 1000, step_rec)

        try:
            array = self.__read_slice(start_rec, stop_rec, step_rec)
        except errors.DataTypeError as e:

            num_recs = 0
            if stop_rec > 0:
                num_recs = stop_rec - e.record

            self.__infer_types(e.record, num_recs, step_rec)
            array = self.__read_slice(start_rec, stop_rec, step_rec)

        return array


    def __infer_types(self, start_rec, num_recs, step_rec):
        """ infer types for first x records """

        cdef uint64_t recs_read = 0

        result = seek_record(self.adapter, start_rec)
        if result != ADAPTER_SUCCESS:
            self.__raise_adapter_exception(result)

        self.adapter.infer_types_mode = 1
        result = read_records(self.adapter, num_recs, step_rec, NULL, &recs_read)
        self.adapter.infer_types_mode = 0

        if result != ADAPTER_SUCCESS and result != ADAPTER_ERROR_READ_EOF:
            self.__raise_adapter_exception(result)

        # Now we know correct dtypes for first 1000 records.
        # Set field/dtype mapping.
        mapping = self.__get_mapping_from_converters()
        for field, dtype in mapping.items():
            self.mapping[field] = numpy.dtype(dtype)
            self.logger.debug('__infer_types() setting dtype {0} for field {1}'
                .format(self.mapping[field].str, str(field)))


    def __read_slice(self, start_rec, stop_rec, step_rec):
        """ Read records by slice """
        cdef uint64_t recs_read = 0
        cdef AdapterError result
        cdef ConvertErrorInfo info
        cdef numpy.ndarray carray
        cdef char *data

        # If number of fields is zero, then field/dtype mapping was
        # probably not set up.
        if self.mapping is None or \
                (isinstance(self.mapping, dict) and len(self.mapping.keys()) == 0):
            self.__raise_adapter_exception(ADAPTER_ERROR_NO_FIELDS);
        self.__build_converters()
        self.__set_fill_values(self.fill_values)

        result = seek_record(self.adapter, start_rec)
        if result != ADAPTER_SUCCESS:
            self.__raise_adapter_exception(result)

        # setup field names for dtype
        if self._field_names is None:
            field_names = ['f' + str(i) for i in range(self.field_count)]
        else:
            field_names = self._field_names

        # Make sure there are no duplicate field names
        # (duplicate blank or None is okay)
        field_namesDict = {}
        for i, name in enumerate(field_names):
            if name is None or name == '':
                continue
            if name in field_namesDict:
                field_namesDict[name] += 1
                field_names[i] = name + str(field_namesDict[name])
            else:
                field_namesDict[name] = 0

        # create proper record dtype from field names and dtypes list
        dtype = []
        try:
            for field in self._field_filter:
                dtype.append((field_names[field], self.mapping[field]))
        except IndexError:
            # leave field names blank as a last resort
            dtype = [('', self.mapping[field]) for field in self._field_filter]
            warning = ('Invalid index {0} for field names [{1}]. Default field '
                       'names will be used.'.format(str(field), ','.join(field_names)))
            self.logger.warning(warning)

        # calculate record size in bytes
        rec_size = 0
        for field in self._field_filter:
            rec_size = rec_size + get_field_size(self.adapter.fields, NULL, field)

        # If we want to read whole file but don't know the number of records,
        # read REC_CHUNK_SIZE records at a time until we reach eof.
        if stop_rec == -1 or stop_rec == 0:

            data = <char*>calloc(self.REC_CHUNK_SIZE, rec_size)
            if <int>data == 0:
                raise MemoryError('out of memory')

            num_recs = 0
            offset = 0
            while True:
                result = read_records(self.adapter, self.REC_CHUNK_SIZE, step_rec,
                    <char*>(data+<uint64_t>offset), &recs_read)

                # Don't worry about EOF error because we don't know how many
                # records we need to read
                if result != ADAPTER_SUCCESS and result != ADAPTER_ERROR_READ_EOF:
                    free(data)
                    self.__raise_adapter_exception(result)

                num_recs = num_recs + recs_read

                # If we've hit eof, resize array to get rid of empty unused
                # records at the end.
                if recs_read < self.REC_CHUNK_SIZE:
                    # Trim unused memory at end of memory block before
                    # creating numpy array out of it
                    data = <char*>realloc(data, num_recs * rec_size)
                    if <int>data == 0:
                        raise MemoryError('out of memory')
                    array = create_array(data, numpy.dtype(dtype), num_recs)
                    break

                offset = offset + (rec_size * self.REC_CHUNK_SIZE)

                # Expand output memory block for next chunk of records
                data = <char*>realloc(data,
                    (num_recs + self.REC_CHUNK_SIZE) * rec_size)
                if <int>data == 0:
                    raise MemoryError('out of memory')

        # We know exactly how many records to read,
        # so allocate array and read them
        else:
            num_recs = math.ceil(abs((stop_rec - start_rec) / step_rec))

            data = <char*>calloc(num_recs, rec_size)
            if <int>data == 0:
                raise MemoryError('out of memory')

            result = read_records(self.adapter, stop_rec - start_rec, step_rec,
                <char*>data, &recs_read)
            if result != ADAPTER_SUCCESS:
                free(data)
                self.__raise_adapter_exception(result)
            if recs_read < num_recs:
                data = <char*>realloc(data, recs_read * rec_size)
                if <int>data == 0:
                    raise MemoryError('out of memory')

            array = create_array(data, numpy.dtype(dtype), recs_read)

        if self.indexing is True:
            self.exact_index.finalize(self.adapter.num_records)

        return array


    def __build_converters(self):
        """
        Set function converters according to field/dtype dict in self.mapping
        """
        reset_converters(self.adapter.fields)

        for field, dtype in self.mapping.items():
            if isinstance(field, (int, long)):
                index = field
            elif isinstance(field, string_types):
                index = self._field_names.index(field)
            else:
                raise TypeError('field to map must be int or string')
            if self._field_filter is None or index in self._field_filter:
                if field in self.build_converter:
                    self.build_converter[field]()
                else:
                    self.__set_dtype_converter(index, dtype)


    def __get_mapping_from_converters(self):
        """
        Set field/dtype dict from converter functions set by
        type inference engine
        """
        cdef converter_func_ptr converter
        mapping = {}

        for field in self._field_filter:
            size = self.adapter.fields.field_info[field].output_field_size
            converter = self.adapter.fields.field_info[field].converter
            if converter == &str2uint_converter:
                dtype_string = 'u%d' % size
            elif converter == &str2int_converter:
                dtype_string = 'i%d' % size
            elif converter == &str2float_converter:
                dtype_string = 'f%d' % size
            elif converter == &str2str_object_converter:
                dtype_string = 'O'
            else:
                continue

            mapping[field] = numpy.dtype(dtype_string)

        return mapping


    def __set_dtype_converter(self, field, dtype):
        """ Set converter for field and dtype """

        cdef converter_func_ptr converter_func
        cdef void *arg = NULL

        if dtype.kind == 'u':
            converter_func = str2uint_converter
        elif dtype.kind == 'i':
            converter_func = str2int_converter
        elif dtype.kind == 'f':
            converter_func = str2float_converter
        elif dtype.kind == 'S':
            converter_func = str2str_converter
        elif dtype.kind == 'c':
            converter_func = str2complex_converter
        elif dtype.kind == 'M':
            converter_func = str2datetime_converter
        else:
            converter_func = str2str_object_converter

        if converter_func == &str2datetime_converter:
            arg = <void*>dtype
        elif converter_func == &str2str_object_converter:
            arg = <void*>self.kh_string_table

        set_converter(self.adapter.fields, field, NULL, dtype.itemsize,
            converter_func, arg)


    def __raise_adapter_exception(self, result):
        cdef ConvertErrorInfo info = get_error_info()

        if result == ADAPTER_ERROR_CONVERT:
            token = ''
            if info.token != NULL:
                token = info.token
            raise text_adapter_exception(result, info.convert_result,
                token, info.record_num, info.field_num,
                self.mapping[info.field_num])
        else:
            raise text_adapter_exception(result, record=info.record_num)


parser_classes = {'csv':CSVTextAdapter,
                  'fixed_width':FixedWidthTextAdapter,
                  'regex':RegexTextAdapter,
                  'json':JSONTextAdapter}

def text_adapter(source, parser='csv', *args, **kwargs):
    """
    Create a text adapter for reading CSV, JSON, or fixed width
    text files, or a text file defined by regular expressions.

    Args:
        source: filename, file object, StringIO object, BytesIO object,
            http url, or python generator
        parser: type of text adapter ('csv', 'fixed_width', 'regex', or 'json')
        encoding: type of character encoding (current ascii and utf8 are supported)
        compression : type of data compression (currently only gzip is supported)
        comment: character used to indicate comment line
        quote: character used to quote fields
        num_records: limits parsing to specified number of records; defaults
            to all records
        header: number of lines in file header; these lines are skipped when parsing
        footer: number of lines in file footer; these lines are skipped when parsing
        field_names: If True, parse field names from first non comment line of
                     text file. If list, use values as field names. If False,
                     use default numpy naming scheme for field names('f0', 'f1', etc).
        indexing: create record index on the fly as characters are read
        index_name: name of file to write index to
        output: type of output object (numpy array or Pandas dataframe)
        debug: If True, enable debug logging
        delimiter (CSV parser only): character used to define fields
        escape (CSV parser only): character used to "escape" characters with
                                  special meaning
        group_whitespace_delims (CSV parser only): If delimiter is whitespace,
                                                   value of True treats multiple
                                                   whitespace characters as
                                                   single delimiter
        whitespace_delims (CSV parser only): If True, treat all whitespace characters
                                             as delimiter
        field_widths (Fixed width parser only): List of field widths
        regex_string (regex parser only): regex string used to define fields
    """
    try:
        parser = parser_classes[parser]
    except KeyError:
        raise errors.AdapterException("Unknown text parser '%s'" % parser)

    return parser(source, *args, **kwargs)

def s3_text_adapter(access_key, secret_key, bucket_name, key_name,
    remote_s3_index=False, parser='csv', *args, **kwargs):
    """
    Create a text adapter for reading a text file from S3. Text file can be
    CSV, JSON, fixed width, or defined by regular expressions

    Args:
        access_key: AWS access key
        secret_key: AWS secret key
        bucket_name: S3 bucket name
        key_name: S3 bucket key
        remote_s3_index: optional text adapter index file stored on S3
        parser: type of text adapter ('csv', 'fixed_width', 'regex', or 'json')
        encoding: type of character encoding (current ascii and utf8 are supported)
        compression : type of data compression (currently only gzip is supported)
        comment: character used to indicate comment line
        quote: character used to quote fields
        num_records: limits parsing to specified number of records; defaults
            to all records
        header: number of lines in file header; these lines are skipped when parsing
        footer: number of lines in file footer; these lines are skipped when parsing
        field_names: If True, parse field names from first non comment line of
                     text file. If list, use values as field names. If False,
                     use default numpy naming scheme for field names('f0', 'f1', etc).
        indexing: create record index on the fly as characters are read
        index_name: name of file to write index to
        output: type of output object (NumpPy array or Pandas DataFrame)
        debug: If True, enable debug logging
        delimiter (CSV parser only): character used to define fields
        escape (CSV parser only): character used to "escape" characters with
                                  special meaning
        group_whitespace_delims (CSV parser only): If delimiter is whitespace,
                                                   value of True treats multiple
                                                   whitespace characters as
                                                   single delimiter
        whitespace_delims (CSV parser only): If True, treat all whitespace characters
                                             as delimiter
        field_widths (Fixed width parser only): List of field widths
        regex_string (regex parser only): regex string used to define fields
    """

    if not boto_installed:
        raise errors.AdapterException("Cannot use s3 interface: boto not installed")

    try:
        parser = parser_classes[parser]
    except KeyError:
        raise errors.AdapterException("Unknown text parser '%s'" % parser)

    conn = boto.connect_s3(access_key, secret_key)
    bucket = conn.get_bucket(bucket_name)
    k = bucket.lookup(key_name)

    if remote_s3_index is True:
        return parser(k, index_name=k, *args, **kwargs)
    else:
        return parser(k, *args, **kwargs)


cdef class CSVTextAdapter(TextAdapter):
    """
    CSV adapter for parsing CSV data using delimiter character.

    Arguments (in addition to TextAdapter base class arguments):
    delimiter - delimiter character that separates fields in record
    """

    def __cinit__(self, fh, delimiter=True, encoding='utf_8', compression=None,
                  comment='#', quote='"', escape='\\', num_records=0,
                  header=0, footer=0, field_names=True, infer_types=True,
                  indexing=False, index_name=None, output='ndarray',
                  group_whitespace_delims=True, whitespace_delims=False, debug=False):

        self.__init_text_adapter(fh, encoding, compression, comment, quote,
            num_records, header, footer, indexing, index_name, output, debug)

        self.adapter.tokenize = &delim_tokenizer;

        self.adapter.escape_char = 0
        if escape is not None:
            self.adapter.escape_char = ord(escape[0])

        # Find first line of text that isn't a comment.
        # Save last comment line. If comment lines exist, we'll use that
        # instead of first non comment text line for field names.
        offset = 0
        comment_offset = 0
        comment_line = None
        try:
            line_iter = create_line_iter(self.adapter)
            line = next(line_iter)
            offset = len(line)

            if comment is not None and line.strip() != '' and \
                    line.strip()[0] == comment:
                comment_line = line
                comment_offset = offset

            while line.strip() == '' or \
                    (comment is not None and line.strip()[0] == comment):
                line = next(line_iter)
                offset = offset + len(line)

                if comment is not None and line.strip() != '' and \
                        line.strip()[0] == comment:
                    comment_line = line
                    comment_offset = offset

        except StopIteration:
            raise EOFError()

        # If a tab delimiter is explicitly specified,
        # override group whitespace delim argument
        if group_whitespace_delims and delimiter != '\t':
            self.adapter.group_whitespace_delims = 1

        if whitespace_delims:
            delimiter = ' '
            self.adapter.any_whitespace_as_delim = 1

        # Try to guess delimiter
        if delimiter is True:
            try:
                d=csv.Sniffer().sniff(line.strip(comment))
                delimiter = str(d.delimiter)
                if delimiter == ' ':
                    whitespace_delims = True
                    self.adapter.any_whitespace_as_delim = 1
            except csv.Error:
                raise errors.ParserError('could not guess delimiter character')

        # Set delimiter character
        if delimiter is None:
            self.adapter.delim_char = 0
        else:
            try:
                self.adapter.delim_char = ord(delimiter[0])
            except OverflowError:
                raise errors.ArgumentError("delimiter must be ASCII character")

        # set number of fields in file
        if whitespace_delims:
            field_values = line.strip(comment).split()
            set_num_fields(self.adapter.fields, len(field_values))
        elif delimiter is not None:
            # csv module doesn't handle unicode on python 2, so just always
            # encode the line string as bytes
            if sys.version < '3':
                line2 = line.encode('utf_8')
            else:
                line2 = line
            csv_reader = csv.reader([line2.strip(comment)],
                delimiter=delimiter, skipinitialspace=True,
                escapechar=escape, quotechar=quote)
            field_values = [value for value in csv_reader]
            if len(field_values[0]) == 1 and len(field_values[0][0]) == 0:
                set_num_fields(self.adapter.fields, 0)
            else:
                set_num_fields(self.adapter.fields, len(field_values[0]))
        else:
            set_num_fields(self.adapter.fields, 1)

        self.field_filter = range(self.field_count)

        # Set field names from either last comment line if it exists,
        # or first line of data.
        if field_names is True:

            if comment_line is not None:
                # csv module doesn't handle unicode on python 2, so just always
                # encode the line string as bytes
                if sys.version < '3':
                    comment_line2 = comment_line.encode('utf_8')
                else:
                    comment_line2 = comment_line
                csv_reader = csv.reader([comment_line2.strip(comment)],
                    delimiter=delimiter, skipinitialspace=True,
                    escapechar=escape, quotechar=quote)
                field_names = [field_name for field_name in csv_reader]
            elif delimiter is not None:
                # csv module doesn't handle unicode on python 2, so just always
                # encode the line string as bytes
                if sys.version < '3':
                    line2 = line.encode('utf_8')
                else:
                    line2 = line
                csv_reader = csv.reader([line2.strip(comment)],
                    delimiter=delimiter, skipinitialspace=True,
                    escapechar=escape, quotechar=quote)
                field_names = [field_name for field_name in csv_reader]
            else:
                field_names = [line.strip(comment)]

            if len(field_names) > 0:

                # If we used the first data line for field names and not the
                # last comment line, reset the header size to the end of the
                # first data line.
                if comment_line is None:
                    self.adapter.input_data.header = self.adapter.input_data.header + offset

                temp_field_names = []
                for name in field_names[0]:
                    name = name.strip()
                    temp_field_names.append(name)
                self.field_names = temp_field_names
        elif isinstance(field_names, list):
            self.field_names = field_names
        else:
            self.field_names = ['f' + str(i) for i in range(self.field_count)]

        # Default missing value strings
        missing_values = {}
        for field in range(self.adapter.fields.num_fields):
            missing_values[field] = default_missing_values
        self.set_missing_values(missing_values)


cdef class FixedWidthTextAdapter(TextAdapter):
    """
    CSV adapter for parsing CSV data with fixed width fields.

    Arguments (in addition to TextAdapter base class arguments):
    field_widths - list of field widths, or single field width for all fields
    """
    __doc__ += TextAdapter.__doc__

    def __cinit__(self, fh, field_widths, encoding='utf_8', compression=None,
                  comment='#', quote='"', num_records=0, header=0, footer=0,
                  field_names=True, infer_types=True, indexing=False,
                  index_name=None, output='ndarray', debug=False):

        self.__init_text_adapter(fh, encoding, compression, comment, quote,
            num_records, header, footer, indexing, index_name, output, debug)

        self.adapter.tokenize = &fixed_width_tokenizer;

        # set number of fields and field widths
        if isinstance(field_widths, (int, long)):
            if field_widths <= 0:
                raise ValueError('field width must be greater than 0')
            try:
                record_iter = create_record_iter(self.adapter)
                line = next(record_iter)
            except StopIteration:
                raise EOFError()

            set_num_fields(self.adapter.fields, len(line)/field_widths)
            for i in range(0, self.adapter.fields.num_fields):
                set_field_width(self.adapter.fields, i, field_widths)
        elif isinstance(field_widths, (list, tuple)):
            for i in field_widths:
                if i <= 0:
                    raise ValueError('field width must be greater than 0')
            set_num_fields(self.adapter.fields, len(field_widths))
            for i in range(0, self.adapter.fields.num_fields):
                set_field_width(self.adapter.fields, i, field_widths[i])
        else:
            raise ValueError('field widths must be specified by int or list of ints')

        self.field_filter = range(self.field_count)

        # Set field names from either last comment line if it exists,
        # or first line of data
        if field_names is True:

            # Find first line of text that isn't a comment.
            # Save last comment line. If comment lines exist, we'll use that
            # instead of first non comment text line for field names.
            offset = 0
            comment_offset = 0
            comment_line = None
            try:
                line_iter = create_line_iter(self.adapter)
                line = next(line_iter)
                offset = len(line)

                if comment is not None and line.strip() != '' and \
                        line.strip()[0] == comment:
                    comment_line = line
                    comment_offset = offset

                while line.strip() == '' or (comment is not None and \
                        line.strip()[0] == comment):
                    line = next(line_iter)
                    offset = offset + len(line)

                    if comment is not None and line.strip() != '' and \
                            line.strip()[0] == comment:
                        comment_line = line
                        comment_offset = offset

            except StopIteration:
                raise EOFError()

            if comment_line is not None:
                coment_line = comment_line.strip()
                comment_line = comment_line.strip(comment)

            field_offset = 0
            temp_field_names = []
            for width in field_widths:
                if comment_line is not None and (field_offset+width) < len(comment_line):
                    temp_field_names.append(comment_line[field_offset:field_offset+width].strip())
                elif line is not None and (field_offset+width) < len(line):
                    temp_field_names.append(line[field_offset:field_offset+width].strip())
                else:
                    temp_field_names.append('')
                field_offset = field_offset + width
            self.field_names = [str(name) for name in temp_field_names]

            # If we used the first data line for field names and not the last
            # comment line, reset the header size to the end of the first data line.
            if comment_line is None:
                self.adapter.input_data.header = self.adapter.input_data.header + offset

        # Default missing value strings
        missing_values = {}
        for field in range(self.adapter.fields.num_fields):
            missing_values[field] = default_missing_values
        self.set_missing_values(missing_values)


cdef class JSONTextAdapter(TextAdapter):

    cdef JsonTokenizerArgs *json_args

    def __cinit__(self, fh, encoding='utf_8', compression=None, num_records=0,
                  field_names=None, infer_types=True, indexing=False,
                  index_name=None, output='ndarray', debug=False):

        self.__init_text_adapter(fh, encoding, compression, None, None,
            num_records, 0, 0, indexing, index_name, output, debug)

        self.json_args = <JsonTokenizerArgs*>calloc(1, sizeof(JsonTokenizerArgs))
        self.json_args.jc = new_JSON_checker(20);
        self.adapter.tokenize_args = <void*>self.json_args
        self.adapter.reset_json_args = 1

        try:
            record_iter = create_json_record_iter(self.adapter)
            line = next(record_iter)
        except StopIteration:
            raise EOFError()

        import json
        if sys.version < '2.7':
            record = json.loads(line, object_hook=OrderedDict)
        else:
            record = json.loads(line, object_pairs_hook=OrderedDict)

        def count_fields(rec, num):
            for field in rec:
                if isinstance(field, list):
                    num = count_fields(field, num)
                elif isinstance(field, dict):
                    num = count_fields(field.values(), num)
                else:
                    num += 1
            return num

        if isinstance(record, dict):
            num_fields = count_fields(record.values(), 0)
        else:
            num_fields = count_fields(record, 0)

        set_num_fields(self.adapter.fields, num_fields)

        self.adapter.tokenize = &json_tokenizer;
        self.field_filter = range(self.field_count)
        if field_names is not None:
            self.field_names = field_names
        elif isinstance(record, dict):
            self.field_names = [str(key) for key in record.keys()]
        else:
            self.field_names = ['f' + str(i) for i in range(self.field_count)]

        # Default missing value strings
        missing_values = {}
        for field in range(self.adapter.fields.num_fields):
            missing_values[field] = default_missing_values
        self.set_missing_values(missing_values)

    def __dealloc__(self):
        if self.json_args != NULL:
            free(self.json_args)


cdef class RegexTextAdapter(TextAdapter):
    """
    CSV adapter for parsing csv data using regular expressions.

    Arguments (in addition to TextAdapter base class arguments):
    regex_string - regular expression string that defines fields in record
    """
    __doc__ += TextAdapter.__doc__

    cdef RegexTokenizerArgs *regex_args
    cdef object temp

    def __cinit__(self, fh, regex_string, encoding='utf_8', compression=None,
                  comment='#', quote='"', num_records=0, header=0, footer=0,
                  field_names=True, infer_types=True, indexing=False,
                  index_name=None, output='ndarray', debug=False):

        self.__init_text_adapter(fh, encoding, compression, comment, quote,
            num_records, header, footer, indexing, index_name, output, debug)

        cdef char *error
        cdef int error_offset

        self.adapter.tokenize = &regex_tokenizer

        regex_string = '^' + regex_string
        pattern = re.compile(regex_string)

        set_num_fields(self.adapter.fields, pattern.groups)

        self.field_filter = range(self.field_count)

        self.regex_args = <RegexTokenizerArgs*>calloc(1, sizeof(RegexTokenizerArgs))
        self.temp = regex_string.encode(self.encoding)
        self.regex_args.pcre_regex = pcre_compile(<char*>self.temp, 0,
            &error, &error_offset, NULL);
        self.regex_args.extra_regex = pcre_study(self.regex_args.pcre_regex, 0, &error);
        self.adapter.tokenize_args = <void*>self.regex_args

       # Set field names from either last comment line if it exists,
        # or first line of data
        if field_names is True:

            # Find first line of text that isn't a comment.
            # Save last comment line. If comment lines exist, we'll use that
            # instead of first non comment text line for field names.
            offset = 0
            comment_offset = 0
            comment_line = None
            try:
                line_iter = create_line_iter(self.adapter)
                line = next(line_iter)
                offset = len(line)

                if comment is not None and line.strip() != '' and \
                        line.strip()[0] == comment:
                    comment_line = line
                    comment_offset = offset

                while line.strip() == '' or (comment is not None and \
                        line.strip()[0] == comment):
                    line = next(line_iter)
                    offset = offset + len(line)

                    if comment is not None and line.strip() != '' and \
                            line.strip()[0] == comment:
                        comment_line = line
                        comment_offset = offset

            except StopIteration:
                raise EOFError()

            if comment_line is not None:
                coment_line = comment_line.strip()
                comment_line = comment_line.strip(comment)

            temp_field_names = [''] * self.field_count
            if comment_line is not None:
                temp_field_names = comment_line.split()
                temp_field_names += [''] * (self.field_count - len(temp_field_names))
            elif line is not None:
                temp_field_names = line.split()
                temp_field_names += [''] * (self.field_count - len(temp_field_names))
            # numpy doesn't like unicode strings as dtype field names on python 2
            self.field_names = [str(name) for name in temp_field_names]

            # If we used the first data line for field names and not the last
            # comment line, reset the header size to the end of the first data line.
            if comment_line is None:
                self.adapter.input_data.header = self.adapter.input_data.header + offset

        # Default missing value strings
        missing_values = {}
        for field in range(self.adapter.fields.num_fields):
            missing_values[field] = default_missing_values
        self.set_missing_values(missing_values)


    def __dealloc__(self):
        if self.regex_args != NULL:
            free(self.regex_args)

# Use text_adapter machinery to iterate over each record in data
cdef object create_line_iter(text_adapter_t *adapter):
    line_iter = LineIter()
    seek_record(adapter, 0)
    line_iter.adapter = adapter
    return line_iter

cdef class LineIter(object):

    cdef text_adapter_t *adapter

    def __iter(self):
        return self

    def __next__(self):
        cdef numpy.ndarray carray
        cdef tokenize_func_ptr tokenizer
        cdef converter_func_ptr converter
        cdef uint32_t num_fields
        cdef numpy.ndarray temp
        cdef void *fill_value

        tokenizer = self.adapter.tokenize
        self.adapter.tokenize = &line_tokenizer
        num_fields = self.adapter.fields.num_fields
        self.adapter.fields.num_fields = 1

        # NOTE: This is kinda hacky, but set up empty string as fill value
        # otherwise adapter will fail to convert blank lines and report error.
        a = numpy.array([''], dtype='O')
        temp = a
        fill_value = <void*>temp.data
        set_fill_value(self.adapter.fields, NULL, 0, fill_value,
            1, 1)

        self.adapter.fields.field_info[0].output_field_size = sizeof(PyObject*)
        converter = self.adapter.fields.field_info[0].converter
        self.adapter.fields.field_info[0].converter = \
            <converter_func_ptr>&str2str_object_converter

        carray = numpy.ndarray(1, dtype='O')
        cdef uint64_t num_recs_read = 0
        result = read_records(self.adapter, 1, 1,
            <char*>(carray.data), &num_recs_read)
        line = numpy.asarray(carray)

        self.adapter.tokenize = tokenizer
        self.adapter.fields.num_fields = num_fields
        self.adapter.fields.field_info[0].converter = converter

        if num_recs_read == 0:
            raise StopIteration

        if line[0] is None:
            line[0] = ''
        return line[0] + '\n'


# Use text_adapter machinery to iterate over each record in data
cdef object create_record_iter(text_adapter_t *adapter):
    record_iter = RecordIter()
    seek_record(adapter, 0)
    record_iter.adapter = adapter
    return record_iter

cdef class RecordIter(object):

    cdef text_adapter_t *adapter

    def __iter(self):
        return self

    def __next__(self):
        cdef numpy.ndarray carray
        cdef tokenize_func_ptr tokenizer
        cdef converter_func_ptr converter
        cdef uint32_t num_fields

        tokenizer = self.adapter.tokenize
        self.adapter.tokenize = &line_tokenizer
        num_fields = self.adapter.fields.num_fields
        self.adapter.fields.num_fields = 1
        self.adapter.fields.field_info[0].output_field_size = sizeof(PyObject*)
        converter = self.adapter.fields.field_info[0].converter
        self.adapter.fields.field_info[0].converter = \
            <converter_func_ptr>&str2str_object_converter

        carray = numpy.ndarray(1, dtype='O')
        result = read_records(self.adapter, 1, 1, <char*>(carray.data), NULL)
        line = numpy.asarray(carray)

        self.adapter.tokenize = tokenizer
        self.adapter.fields.num_fields = num_fields
        self.adapter.fields.field_info[0].converter = converter

        if line[0] is None:
            raise StopIteration

        return line[0] + '\n'


cdef object create_json_record_iter(text_adapter_t *adapter):
    record_iter = JSONRecordIter()
    seek_record(adapter, 0)
    record_iter.adapter = adapter
    return record_iter

cdef class JSONRecordIter(object):

    cdef text_adapter_t *adapter

    def __iter(self):
        return self

    def __next__(self):
        cdef numpy.ndarray carray
        cdef tokenize_func_ptr tokenizer
        cdef converter_func_ptr converter
        cdef uint32_t num_fields

        tokenizer = self.adapter.tokenize
        self.adapter.tokenize = &json_record_tokenizer
        num_fields = self.adapter.fields.num_fields
        self.adapter.fields.num_fields = 1
        self.adapter.fields.field_info[0].output_field_size = sizeof(PyObject*)
        converter = self.adapter.fields.field_info[0].converter
        self.adapter.fields.field_info[0].converter = \
            <converter_func_ptr>&str2str_object_converter

        carray = numpy.ndarray(1, dtype='O')
        result = read_records(self.adapter, 1, 1, <char*>(carray.data), NULL)
        line = numpy.asarray(carray)

        self.adapter.tokenize = tokenizer
        self.adapter.fields.num_fields = num_fields
        self.adapter.fields.field_info[0].converter = converter

        if line[0] is None:
            raise StopIteration

        return line[0]

cdef create_array(char *data, object dtype, uint64_t num_recs):
    """
    Create numpy array out of pre allocated data filled in by call to read_records().
    """

    cdef numpy.npy_intp dims[1]
    cdef ArrayDealloc array_dealloc
    cdef numpy.ndarray carray

    dims[0] = num_recs
    carray = PyArray_NewFromDescr(numpy.ndarray, dtype, 1, dims, NULL, data,
        NPY_WRITEABLE, <object>NULL)
    Py_INCREF(dtype)

    # Use ArrayDealloc object to make sure array data is properly deallocted
    # when array is destroyed
    array_dealloc = ArrayDealloc.__new__(ArrayDealloc)
    array_dealloc.data = data
    Py_INCREF(array_dealloc)

    carray.base = <PyObject*>array_dealloc
    return numpy.asarray(carray)


convert_error_reasons = {
    CONVERT_SUCCESS: "no conversion error",
    CONVERT_ERROR_OVERFLOW: "overflow",
    CONVERT_ERROR_INPUT_TYPE: "input type",
    CONVERT_ERROR_INPUT_SIZE: "input size",
    CONVERT_ERROR_OUTPUT_SIZE: "output size",
    CONVERT_ERROR_INPUT_STRING: "input string",
    CONVERT_ERROR_USER_CONVERTER: "user converter",
    CONVERT_ERROR_OBJECT_CONVERTER: "object converter"
}

def text_adapter_exception(error, convert_error=None, token=None, record=None,
                           field=None, dtype=None):
    """
    Returns appropriate adapter exception for reporting reading, parsing,
    and converting issues.

    Arguments:
    * `error` - AdapterError enum value
    * `convert_error` - ConvertError enum value, if this is a convert error
    * `token` - token string where error occurred, if applicable
    * `record` - record where parsing or convert error happened
    * `field` - field where parsing or convert error happened
    """
    if error == ADAPTER_ERROR_CONVERT:
        reason = convert_error_reasons.get(convert_error, "unknown")
        e = errors.DataTypeError('Could not convert token "{0}" at record {1} field {2} to {3}.' \
                          'Reason: {4}'.format(token, str(record), str(field), dtype, reason))
    elif error == ADAPTER_ERROR_SEEK:
        e = errors.SourceError('Seek error')
    elif error == ADAPTER_ERROR_SEEK_EOF:
        e = errors.AdapterIndexError('Record {0} is out of bounds'.format(str(record)))
    elif error == ADAPTER_ERROR_SEEK_S3:
        e = errors.SourceError('S3 seek error')
    elif error == ADAPTER_ERROR_READ:
        e = errors.SourceError('Read error')
    elif error == ADAPTER_ERROR_READ_S3:
        e = errors.SourceError('S3 read error')
    elif error == ADAPTER_ERROR_NO_FIELDS:
        e =  errors.NoSuchFieldError('No fields found or no converters set')
    elif error == ADAPTER_ERROR_INDEX:
        e = errors.DataIndexError('Could not find record offset in index')
    elif error == ADAPTER_ERROR_PROCESS_TOKEN:
        e = errors.ParserError('Error processing token')
        e.token = token
    elif error == ADAPTER_ERROR_READ_TOKENS:
        e = errors.ParserError('Error reading tokens')
        e.token = token
    elif error == ADAPTER_ERROR_READ_RECORDS:
        e = errors.ParserError('Error reading records')
        e.token = token
    elif error == ADAPTER_ERROR_READ_EOF:
        e = errors.AdapterIndexError('Invalid record number or slice')
    elif error == ADAPTER_ERROR_INVALID_CHAR_CODE:
        e = errors.ParserError('Invalid character in input data')
        e.token = token
    else:
        e = errors.AdapterException('Unknown text adapter error (code %d)' % error)

    e.field = field
    e.record = record

    return e
