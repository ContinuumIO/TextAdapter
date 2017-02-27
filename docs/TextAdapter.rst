<<<<<<< HEAD
<<<<<<< HEAD
***REMOVED***
TextAdapter
***REMOVED***
=======
-----------
TextAdapter
-----------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
-----------
TextAdapter
-----------
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c

.. contents::

The TextAdapter module reads CSV data and produces a NumPy array containing the
parsed data. The following features are currently implemented:

* The TextAdapter engine is written
  in C to ensure text is parsed as fast as data can be read from the source.
  Text is read and parsed in small chunks instead of reading entire data into
  memory at once, which enables very large files to be read and parsed without
  running out of memory.

* Python slicing notation can be used to specify a subset of records to be
  read from the data source, as well as a subset of fields.

* Fields can be specified in any one of three ways: by a delimiter character, 
  using fixed field widths, or by a regular expression. This enables a larger 
  variety of CSV-like and other types of text files to be parsed.

* A gzipped file can be parsed without having to uncompress it first. Parsing speed
  is about the same as an uncompressed version of same file.

* An index of record offsets in a file can be built to allow fast random access to
  records. This index can be saved to disk and loaded again later.

* Converter functions can be specified for converting parsed text to proper dtype
  for storing in NumPy array.

* The TextAdapter engine has automatic type inference so the user does not have to
  specify dtypes of the output array. The user can still specify dtypes manually if
  desired.

* Remote data stored in Amazon S3 can be read. An index can be built and stored
  with S3 data. Index can be read remotely, allowing for random access to S3 data.

Methods
-------
The TextAdapter module contains the following factory methods for creating TextAdapter objects:

**text_adapter** (source, parser='csv', compression=None, comment='#',
                  quote='"', num_records=0, header=0, field_names=True,
                  indexing=False, index_name=None, encoding='utf-8')

    | Create a text adapter for reading CSV, JSON, or fixed width
    | text files, or a text file defined by regular expressions.

    | source - filename, file object, StringIO object, BytesIO object, S3 key,
      http url, or python generator
    | parser - Type of parser for parsing text. Valid parser types are 'csv', 'fixed width', 'regex', and 'json'.
    | encoding - type of character encoding (current ascii and utf8 are supported)
    | compression - type of data compression (currently only gzip is supported)
    | comment - character used to indicate comment line
    | quote - character used to quote fields
    | num_records - limits parsing to specified number of records; defaults
      to all records
    | header - number of lines in file header; these lines are skipped when parsing
    | footer - number of lines in file footer; these lines are skipped when parsing
    | indexing - create record index on the fly as characters are read
    | index_name - name of file to write index to
    | output - type of output object (numpy array or pandas dataframe)


If parser is set to 'csv', additional parameters include:
    | delimiter - Delimiter character used to define fields in data source. Default is ','.

If parser is set to 'fixed_width', additional parameters include:
    | field_widths - List of field widths

If parser is set to 'regex', additional parameters include:
    | regex - Regular expression used to define records and fields in data source.
      See the regular expression example in the Advanced Usage section.

**s3_text_adapter** (access_key, secret_key, bucket_name, key_name, remote_s3_index=False)
                     parser='csv', compression=None, comment='#',
                     quote='"', num_records=0, header=0, field_names=True,
                     indexing=False, index_name=None, encoding='utf-8')

    | Create a text adapter for reading a text file from S3. Text file can be
    | CSV, JSON, fixed width, or defined by regular expressions

In addition to the arguments described for the text_adapter function above,
the s3_text_adapter function also has the following parameters:

    | access_key - AWS access key
    | secret_key - AWS secret key
    | bucket_name - name of S3 bucket
    | key_name - name of key in S3 bucket
    | remote_s3_index - use remote S3 index (index name must be key name + '.idx' extension)


The TextAdapter object returned by the text_adapter factory method contains the following methods:

**set_converter** (field, converter)
    | Set converter function for field

    | field - field to apply converter function
    | converter - python function object

**set_missing_values** (missing_values)
    | Set strings for each field that represents a missing value

    | missing_values - dict of field name or number,
      and list of missing value strings

    Default missing values: 'NA', 'NaN', 'inf', '-inf', 'None', 'none', ''

**set_fill_values** (fill_values, loose=False)
    | Set fill values for each field

    | fill_values - dict of field name or number, and fill value
    | loose - If value cannot be converted, and value does not match
      any of the missing values, replace with fill value anyway.

    Default fill values for each data type:
    | int - 0
    | float - numpy.nan
    | char - 0
    | bool - False
    | object - numpy.nan
    | string - numpy.nan

**create_index** (index_name=None, density=1)
    | Create an index of record offsets in file

    | index_name - Name of file on disk used to store index. If None, index
      will be created in memory but not saved.
    | density - density of index. Value of 1 will index every record, value of
      2 will index every other record, etc.

**to_array** ()
    | Parses entire data source and returns data as NumPy array object

**to_dataframe** ()
    | Parses entire data source and returns data as Pandas DataFrame object

The TextAdapter object contains the following properties:

**size** (readonly)
    | Number of records in data source. This value is only set if entire data
      source has been read or indexed, or number of recods was specified in
      text_adapter factory method when creating object.

**field_count** (readonly)
    | Number of fields in each record

**field_names**
    | Field names to use when creating output NumPy array. Field names can be
      set here before reading data or in text_adapter function with
      field_names parameter.

**field_types**
    | NumPy dtypes for each field, specified as a dict of fields and associated
      dtype. (Example: {0:'u4', 1:'f8', 2:'S10'})

**field_filter**
    | Fields in data source to parse, specified as a list of field numbers
      or names (Examples: [0, 1, 2] or ['f1', 'f3', 'f5']). This filter stays
      in effect until it is reset to empty list, or is overridden with array
      slicing (Example: adapter[[0, 1, 3, 4]][:]).

    See the NumPy data types documentation for more details:
      http://docs.continuum.io/anaconda/numpy/reference/arrays.dtypes.html

The TextAdapter object supports array slicing:

    | Read all records:
      adapter[:]

    | Read first 100 records:
      adapter[0:100]

    | Read last record (only if data has been indexed or entire dataset
      has been read once before):
      adapter[-1]

    | Read first field in all records by specifying field number:
      adapter[0][:]

    | Read first field in all records by specifying field name:
      adapter['f0'][:]

    | Read first and third fields in all records:
      adapter[[0, 2]][:]

<<<<<<< HEAD
<<<<<<< HEAD
***REMOVED***
***REMOVED***
=======
Basic Usage
-----------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
Basic Usage
-----------
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c

Create TextAdapter object for data source::

    >>> import TextAdapter
    >>> adapter = TextAdapter.text_adapter('data.csv', parser='csv')

Parse text and store records in NumPy array using slicing notation::

    >>> # read all records
<<<<<<< HEAD
<<<<<<< HEAD
***REMOVED***
=======
    >>> array = adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
    >>> array = adapter[:]
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c

    >>> # read first ten records
    >>> array = adapter[0:10]

    >>> # read last record
<<<<<<< HEAD
<<<<<<< HEAD
 ***REMOVED***
=======
    >>> array = adapter[-1]
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c

    >>> # read every other record
    >>> array = adapter[::2]

Advanced Usage
<<<<<<< HEAD
***REMOVED***---
=======
    >>> array = adapter[-1]

    >>> # read every other record
    >>> array = adapter[::2]

Advanced Usage
--------------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
--------------
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c

user defined converter function for field 0::

    >>> import TextAdapter
    >>> import io

    >>> data = '1, abc, 3.3\n2, xxx, 9.9'
    >>> adapter = TextAdapter.text_adapter(io.StringIO(data), parser='csv', field_names=False)

    >>> # Override default converter for first field
    >>> adapter.set_converter(0, lambda x: int(x)*2)
<<<<<<< HEAD
<<<<<<< HEAD
***REMOVED***
=======
    >>> adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
    >>> adapter[:]
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c
    array([(2L, ' abc', 3.3), (4L, ' xxx', 9.9)],
              dtype=[('f0', '<u8'), ('f1', 'S4'), ('f2', '<f8')])

overriding default missing and fill values::

    >>> import TextAdapter
    >>> import io

    >>> data = '1,abc,inf\n2,NA,9.9'
    >>> adapter = TextAdapter.text_adapter(io.StringIO(data), parser='csv', field_names=False)

    >>> # Define field dtypes (example: set field 1 to string object and field 2 to float)
<<<<<<< HEAD
<<<<<<< HEAD
 ***REMOVED*** = {1:'O', 2:'f4'}
=======
    >>> adapter.field_types = {1:'O', 2:'f4'}
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
    >>> adapter.field_types = {1:'O', 2:'f4'}
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c

    >>> # Define list of strings for each field that represent missing values
    >>> adapter.set_missing_values({1:['NA'], 2:['inf']})

    >>> # Set fill value for missing values in each field
    >>> adapter.set_fill_values({1:'xxx', 2:999.999})
<<<<<<< HEAD
<<<<<<< HEAD
***REMOVED***
=======
    >>> adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
    >>> adapter[:]
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c
    array([(' abc', 999.9990234375), ('xxx', 9.899999618530273)],
              dtype=[('f0', 'O'), ('f1', '<f4')])

creating and saving tuple of index arrays for gzip file, and reloading indices::

    >>> import TextAdapter
    >>> adapter = TextAdapter.text_adapter('data.gz', parser='csv', compression='gzip')

    >>> # Build index of records and save index to disk.
    >>> adapter.create_index(index_name='index_file')

    >>> # Create new adapter object and load index from disk.
    >>> adapter = TextAdapter.text_adapter('data.gz', parser='csv', compression='gzip', indexing=True, index_name='index_file')

    >>> # Read last record
    >>> adapter[-1]
    array([(100, 101, 102)],dtype=[('f0', '<u4'), ('f1', '<u4'), ('f2', '<u4')])

Use regular expression for finer control of extracting data::

    >>> import TextAdapter
    >>> import io

    >>> # Define regular expression to extract dollar amount, percentage, and month.
    >>> # Each set of parentheses defines a field.
    >>> data = '$2.56, 50%, September 20 1978\n$1.23, 23%, April 5 1981'
    >>> regex_string = '([0-9]\.[0-9][0-9]+)\,\s ([0-9]+)\%\,\s ([A-Za-z]+)'
    >>> adapter = TextAdapter.text_adapter(io.StringIO(data), parser='regex', regex_string=regex_string, field_names=False, infer_types=False)

    >>> # set dtype of field to float
<<<<<<< HEAD
<<<<<<< HEAD
 ***REMOVED*** = {0:'f4', 1:'u4', 2:'S10'}
***REMOVED***
=======
    >>> adapter.field_types = {0:'f4', 1:'u4', 2:'S10'}
    >>> adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
    >>> adapter.field_types = {0:'f4', 1:'u4', 2:'S10'}
    >>> adapter[:]
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c
    array([(2.56, 50L, 'September'), (1.23, 23L, 'April')],
        dtype=[('f0', '<f8'), ('f1', '<u8'), ('f2', 'S9')])
