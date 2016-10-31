Installing NumPy-aware pyodbc
=============================

Compiling
---------

For compiling this library, just use the standard Python procedure::

  $ python setup.py build_ext -i

You will need a C++ compiler installed.


Running the test suite
***REMOVED******REMOVED***

For testing the library, look into the original documentation in:

:homepage: http://code.google.com/p/pyodbc

For testing the NumPy-aware methods specifically, please configure
your ODBC system by installing the next packages::

  - ODBC system:  unixODBC >= 2.2.14
  - SQLite backend:  SQLite ODBC >= 0.96
  - MySQL backend:  mysql-connector-odbc >= 5.1.10
  - Postgres backend:  psqlODBC >= 09.01.0100

Then make sure that you have proper 'odcbinst.ini' and 'odbc.ini'
files configured in '/etc'.  Also, edit the 'tests_numpy/odbc.cfg'
file and configure it to your needs.  Here it is an example::

  [sqlite]
  connection-string=DSN=odbcsqlite;Database=test-sqlite.db
  [mysql]
  connection-string=DSN=myodbc;Database=test
  [postgresql]
  connection-string=DSN=odbcpostgresql;Database=test
 
On Windows, try:

  [sqlite]
  connection-string=DSN=SQLite3 Datasource;Database=test-sqlite.db
  [mysql]
  connection-string=DSN=MySQL55;Database=test

You may want to have a look at the examples included in the 'samples/'
directory.  These offer configurations for SQLite, MySQL and
PostgreSQL ODBC drivers, but you can add a new ODBC driver for any
other database you want.  The only restriction is that they must
support the ODBC standard 3.0 or higher.

To run the test suite for the different backends configured in
'tests_numpy/odbc.cfg', just do::

  $ PYTHONPATH=. python tests_numpy/generic.py [your-backend]

This will run the NumPy-aware test suite for the selected backend.  In
case you don't provide a backend, it defaults to `sqlite`.

In case some test fails, please report this back.

Installing
***REMOVED***

After you run the test suite, you are ready to install.  Just do it with::

  $ [sudo] python setup.py install

That's all folks!
