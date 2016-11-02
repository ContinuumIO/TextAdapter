<<<<<<< HEAD
***REMOVED***-----
Accumulo Adapter
***REMOVED***-----
=======
----------------
Accumulo Adapter
----------------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1

.. contents::

The AccumuloAdapter module reads data from Accumulo key/value stores and produces
a NumPy array containing the parsed values.

* The AccumuloAdapter engine is written in C to ensure returned data is parsed
  as fast as data can be read from the source. Data is read and parsed in small
  chunks instead of reading entire data into memory at once.

* Python slicing notation can be used to specify a subset of records to be
  read from the data source.

Adapter Methods
<<<<<<< HEAD
***REMOVED***----
=======
---------------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
Accumulo Adapter Constructor:

**AccumuloAdapter** (server='localhost', port=42424, username='', password='', table=None, field_type='f8', start_key=None, stop_key=None, start_key_inclusive=True, stop_key_inclusive=False, missing_values=None, fill_value=None):

    | Create an adaptor for connecting to an Accumulo key/value store.

    | server: Accumulo server address
    | port: Accumulo port
    | username: Accumulo user name
    | password: Accumulo user password
    | table: Accumulo table to read data from
    | field_type: str, NumPy dtype to interpret table values as
    | start_key: str, key of record where scanning will start from
    | stop_key: str, key of record where scanning will stop at
    | start_key_inclusive: If True, start_key is inclusive (default is True)
    | stop_key_inclusive: If True, stop_key is inclusive (default is False)
    | missing_values: list, missing value strings. Any values in table equal
                      to one of these strings will be replaced with fill_value.
    | fill_value: fill value used to replace missing value when scanning

**close** ()
    | Close connection to the database.

The AccumuloAdapter object supports array slicing:

    | Read all records:
      adapter[:]

    | Read first ten records:
      adapter[0:10]

    | Read last record:
      adapter[-1]

<<<<<<< HEAD
    | ***REMOVED***
=======
    | Read every other record:
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
      adapter[::2]


Adapter Properties
<<<<<<< HEAD
***REMOVED***-------
=======
------------------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
**field_type** (readonly)
    | Get dtype of output NumPy array

**start_key**
    | Get/set key of record where reading/scanning will start.
    | The start_key_inclusive property specifies whether this key is inclusive
    | (default is inclusive).

**stop_key**
    | Get/set key of record where reading/scanning will stop.
    | The stop_key_inclusive property specifies whether this key is inclusive
    | (default is exclusive).

**start_key_inclusive**
    | Toggle whether start key is inclusive. Default is true.

**stop_key_inclusive**
    | Toggle whether stop key is inclusive. Default is False.

**missing_values**
    | Get/Set missing value strings. Any values in Accumulo table matching one
    | of these strings will be replaced with fill_value.

**fill_value**
    | Fill value used to replace missing_values. Fill value type should match
    | specified output type.

<<<<<<< HEAD
***REMOVED***
***REMOVED***

Create AccumuloAdapter object for data source::

***REMOVED***
=======
Basic Usage
-----------

Create AccumuloAdapter object for data source::

    >>> import iopro
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    >>> adapter = iopro.AccumuloAdapter(server='172.17.0.1',
                                        port=42424,
                                        username='root',
                                        password='password',
                                        field_type='f4',
                                        table='iopro_tutorial_data')

IOPro adapters use slicing to retrieve data. To retrieve records from the table
or query, the standard NumPy slicing notation can be used:

    >>> # read all records
<<<<<<< HEAD
***REMOVED***
=======
    >>> array = adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    array([ 0.5,  1.5,  2.5,  3.5,  4.5], dtype=float32)

    >>> # read first three records
    >>> array = adapter[0:3]
    array([ 0.5,  1.5,  2.5], dtype=float32)

    >>> # read every other record from the first four records
    >>> array = adapter[:4:2]
    array([ 0.5,  2.5], dtype=float32)

The Accumulo adapter does not support seeking from the last record.

The field_types property can be used to see what type the output NumPy array
will have:

    >>> adapter.field_type
    'f4'

Since Accumulo is essentially a key/value store, results can be filtered
based on key. For example, a start key using the start_key property. This will
retrieve all values with a key equal to or greater than the start key.

    >>> adapter.start_key = 'row02'
<<<<<<< HEAD
***REMOVED***
=======
    >>> adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    array([ 1.5,  2.5,  3.5,  4.5], dtype=float32)

Likewise, a stop key. This will retrieve all values with a key less than the
stop key but equal to or greater than the start key.

    >>> adapter.stop_key = 'row04'
<<<<<<< HEAD
***REMOVED***
=======
    >>> adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    array([ 1.5,  2.5], dtype=float32)

By default, the start key is inclusive. This can be changed by setting the
start_key_inclusive property to False.

    >>> adapter.start_key_inclusive = False
<<<<<<< HEAD
***REMOVED***
=======
    >>> adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    array([ 2.5], dtype=float32)

By default, the stop key is exclusive. This can be changed by setting the
stop_key_inclusive property to True.

    >>> adapter.stop_key_inclusive = True
<<<<<<< HEAD
***REMOVED***
=======
    >>> adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    array([ 2.5,  3.5], dtype=float32)

The Accumulo adapter can handle missing values. If it is known that the strings
'NA' and 'nan' signify missing float values, the missing_values property can be
used to tell the adapter to treat these strings as missing values: Also, the
fill_value property can be used to specify what value to replace missing values
with.

    >>> adapter = iopro.AccumuloAdapter('172.17.0.1', 42424, 'root', 'password', 'iopro_tutorial_missing_data', field_type='S10')
<<<<<<< HEAD
***REMOVED***
=======
    >>> adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    array([b'NA', b'nan'], dtype='|S10')

    >>> adapter = iopro.AccumuloAdapter('172.17.0.1', 42424, 'root', 'secret', 'iopro_tutorial_missing_data', field_type='f8')
    >>> adapter.missing_values = ['NA', 'nan']
    >>> adapter.fill_value = np.nan
<<<<<<< HEAD
***REMOVED***
=======
    >>> adapter[:]
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    array([ nan,  nan])

Close database connection:
    >>> adapter.close()
