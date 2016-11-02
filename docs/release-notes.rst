TextAdapter Release Notes
=========================

2016-10-30: 2.0.0
-----------------

* Release the closed source of IOPro as Freee Software.  Touch up some
naming issues and small tweaks to build scripts, documentation, etc.

IOPro Release Notes
===================

2016-07-30:  1.9.0
------------------

* Remove warnings and documentation for unsupported Numba use
* Rewrite most documentation for clarity and accuracy
* Improve unit tests


2016-04-05:  1.8.0:
-------------------

* Add PostgresAdapter for reading data from PostgreSQL databases
* Add AccumuloAdapter for reading data from Accumulo databases


2015-10-09:  1.7.2:
-------------------

* Fix an issue with pyodbc where result NumPy arrays could return
  uninitialized data after the actual data null character.  Now it pads
  the results with nulls.


2015-05-04:  1.7.1
------------------

* Properly cache output string objects for better performance


2015-03-02:  1.7.0
-------------------

* Add Python 3 support
* Add support for parsing utf8 text files
* Add ability to set/get field types in MongoAdapter


2015-02-02:  1.6.11
-------------------

* Fix issue with escape char not being parsed correctly inside quoted strings


2014-12-17:  1.6.10
-------------------

* Fix issue with using field filters with json parser


2014-12-02:  1.6.9
------------------

* Fix issue with json field names getting mixed up


2014-11-20:  1.6.8
------------------

* Fix issue with return nulls returning wrong "null" for large queries
  (more than 10000 rows) in some circumstances.
* Fix issue with reading slices of json data
* Change json parser so that strings fields of numbers do not get converted
  to number type by default
* Allow json field names to be specified with field_names constructor
  argument
* If user does not specify json field names, use json attribute names as
  field names in array result


2014-07-03:  1.6.7
------------------

* Fix issue when reading more than 10000 rows containing unicode strings in platform where ODBC uses UTF-16/UCS2 encoding (notably Windows and unixODBC). The resulting data could be corrupt.


2014-06-16:  1.6.6
------------------

* Fix possible segfault when dealing with unicode strings in platforms where ODBC uses UTF-16/UCS2 encoding (notably Windows and unixODBC)
* Add iopro_set_text_limit function to iopro. It globally limits the size of text fields read by fetchdictarray and fetchsarray. By default it is set to 1024 characters.
* Fix possible segfault in fetchdictarray and fetchsarray when failing to allocate some NumPy array. This could notably happen in the presence of "TEXT" fields. Now it will raise an OutOfMemory error.
* Add lazy loading of submodules in IOPro. This reduces upfront import time of IOPro. Features are imported as they are used for the first time.


2014-05-07:  1.6.5
------------------

* Fix crash when building textadapter index


2014-04-29:  1.6.4
------------------

* Fix default value for null strings in IOPro/pyodbc changed to be an empty string instead of 'NA'. NA was not appropriate as it can collide with valid data (Namibia country code is 'NA', for example), and it failed with single character columns.
* Ignore SQlRowCount when performing queries with fetchsarray and fetchdictarray, since SQLRowCount sometimes returns incorrect number of rows.


2014-03-25:  1.6.3
------------------

* Fix SQL TINYINT is now returned as an unsigned 8 bit integer in fetchdictarray/fetchsarray. This is to match the range specified in SQL (0...255). It was being returned as a signed 8 bit integer before (range -128...127)
* Add Preliminary unicode string support in fetchdictarray/fetchsarray.


2014-02-12:  1.6.2
------------------

* Disable Numba support for version 0.12 due to lack of string support.


2014-01-30:  1.6.1
------------------

* Fix a regression that made possible some garbage in string fields when using fetchdictarray/fetchsarray.
* Fix a problem where heap corruption could happen in IOPro.pyodbc fetchdictarray/fetchsarray related to nullable string fields.
* Fix the allocation guard debugging code: iopro.pyodbc.enable_mem_guards(True|False) should no longer crash.
* Merge Vertica fix for cancelling queries


2013-10-30:  1.6.0
------------------

* Add JSON support
* Misc bug fixes
* Fix crash in IOPro.pyodbc when dealing with nullable datetimes in fetch_dictarray and fetch_sarray.


2013-06-12:  1.5.5
------------------

* Fix issue parsing negative ints with leading whitespace in csv data.


2013-06-10:  1.5.4
------------------

* Allow delimiter to be set to None for csv files with single field.
* Fill in missing csv fields with fill values.
* Fill in blank csv lines with fill values for pandas dataframe output.
* Allow list of field names for TextAdapter field_names parameter.
* Change default missing fill value to empty string for string fields.


2013-06-05:  1.5.3
------------------

* Temporary fix for IndexError exception in TextAdapter.__read_slice method.


2013-05-28:  1.5.2
------------------

* Add ability to specify escape character in csv data


2013-05-23:  1.5.1
------------------

* fixed coredump when using datetime with numpy < 1.7


2013-05-22:  1.5.0
------------------

* Added a cancel method to the Cursor object in iopro.pyodbc.
  This method wraps ODBC SQLCancel.
* DECIMAL and NUMERIC types are now working on iopro.pyodbc on regular fetch
  functions. They are still unsupported in fetchsarray and fetchdict and
  fetchsarray
* Add ftp support
* Performance improvements to S3 support
* Misc bug fixes


2013-04-05:  1.4.3
------------------

* Update loadtxt and genfromtxt to reflect numpy versions' behavior
  for dealing with whitespace (default to any whitespace as delimiter,
  and treat multiple whitespace as one delimiter)
* Add read/write field_names property
* Add support for pandas dataframes as output
* Misc bug fixes
