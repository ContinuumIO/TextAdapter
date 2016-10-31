
Overview
========

This project is an enhancement of the Python database module for ODBC
that implements the Python DB API 2.0 specification.  You can see the
original project here:

:homepage: http://code.google.com/p/pyodbc
:source:   http://github.com/mkleehammer/pyodbc
:source:   http://code.google.com/p/pyodbc/source/list

The enhancements are documented in this file.  For general info about
the pyodbc package, please refer to the original project
documentation.

This module enhancement requires:

* Python 2.4 or greater
* ODBC 3.0 or greater
* NumPy 1.5 or greater (1.7 is required for datetime64 support)

The enhancements in this module consist mainly in the addition of some
new methods for fetching the data after a query and put it in a
variety of NumPy containers.

Using NumPy as data containers instead of the classical list of tuples
has a couple of advantages:

1) The NumPy container is much more compact, and hence, it
requires much less memory, than the original approach.

2) As a NumPy container can hold arbitrarily large arrays, it requires
much less object creation than the original approach (one Python
object per datum retrieved).

This means that this enhancements will allow to fetch data out of
relational databases in a much faster way, while consuming
significantly less resources.


Installing
==========

Please follow the instructions in 'INSTALL.rst'.


API additions
=============

Variables
~~~~~~~~~

* `pyodbc.npversion`  The version for the NumPy additions

Methods
~~~~~~~

Cursor.fetchdictarray(size=cursor.arraysize)
--------------------------------------------

This is similar to the original `Cursor.fetchmany(size)`, but the data
is returned in a dictionary where the keys are the names of the
columns and the values are NumPy containers.

For example, it a SELECT is returning 3 columns with names 'a', 'b'
and 'c' and types `varchar(10)`, `integer` and `timestamp`, the
returned object will be something similar to::

  {'a': array([...], dtype='S11'),
   'b': array([...], dtype=int32),
   'c': array([...], dtype=datetime64[us])}

Note that the `varchar(10)` type is translated automatically to a
string type of 11 elements ('S11').  This is because the ODBC driver
needs one additional space to put the trailing '\0' in strings, and
NumPy needs to provide the room for this.

Also, it is important to stress that all the `timestamp` types are
translated into a NumPy `datetime64` type with a resolution of
microseconds by default.

Cursor.fetchsarray(size=cursor.arraysize)
-----------------------------------------

This is similar to the original `Cursor.fetchmany(size)`, but the data
is returned in a NumPy structured array, where the name and type of
the fields matches to those resulting from the SELECT.

Here it is an example of the output for the SELECT above::

  array([(...),
         (...)], 
        dtype=[('a', '|S11'), ('b', '<i4'), ('c', ('<M8[us]', {}))])

Note that, due to efficiency considerations, this method is calling the
`fetchdictarray()` behind the scenes, and then doing a conversion to
get an structured array.  So, in general, this is a bit slower than
its `fetchdictarray()` counterpart.


Data types supported
====================

The new methods listed above have support for a subset of the standard
ODBC.  In particular:

* String support (SQL_VARCHAR) is supported.

* Numerical types, be them integers or floats (single and double
  precision) are fully supported.  Here it is the complete list:
  SQL_INTEGER, SQL_TINYINT, SQL_SMALLINT, SQL_FLOAT and SQL_DOUBLE.

* Dates, times, and timestamps are mapped to the `datetime64` and
  `timedelta` NumPy types.  The list of supported data types are:
  SQL_DATE, SQL_TIME and SQL_TIMESTAMP,

* Binary data is not supported yet.

* Unicode strings are not supported yet.


NULL values
===========

As there is not (yet) a definitive support for missing values (NA) in
NumPy, this module represents NA data as particular values depending
on the data type.  Here it is the current table of the particular
values::

  int8: -128 (-2**7)
  uint8: 255 (2**8-1)
  int16: -32768 (-2**15)
  uint16: 65535 (2**16-1)
  int32: -2147483648 (-2**31)
  uint32: 4294967295 (2**32-1)
  int64: -9223372036854775808 (-2**63)
  uint64: 18446744073709551615 (2**64-1)
  float32: NaN
  float64: NaN
  datetime64: NaT
  timedelta64: NaT (or -2**63)
  string: 'NA'


Improvements for 1.1 release
============================

- The rowcount is not trusted anymore for the `fetchdict()` and
  `fetchsarray()` methods.  Now the NumPy containers are built
  incrementally, using realloc for a better use of resources.

- The Python interpreter does not exit anymore when fetching an exotic
  datatype not supported by NumPy.

- The docsctrings for `fetchdict()` and `fetchsarray()` have been improved.


Bug reports
===========

This software is still under development.  Please feel free to report
any problems you might find.  We will try to come up with an answer as
soon as possible.  Thanks!
