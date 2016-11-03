TextAdapter
======

TextAdapter is a Python module containing optimized data adapters for
importing data from a variety of data sources into NumPy arrays and Pandas
DataFrame.  Current data adapters include TextAdapter for JSON, free-form,
and CSV-like text files.  DBAdapter, also based on IOPro, accesses
MongoAdapter for mongo databases, PostgresAdapter for PostgreSQL databases,
AccumuloAdapter for Accumulo databases, and an optimized pyodbc module for
accessing any relational database that supports the ODBC interface (SQL
Server, PostgreSQL, MySQL, etc).

Build Requirements
------------------

Building TextAdapter requires a number of dependencies. In addition to a
C/C++ dev environment, the following modules are needed, which can be
installed via conda.

* NumPy
* Pandas
* zlib 1.2.8 (C lib)
* pcre 8.31 (C lib)

Building Conda Package
----------------------

Note: If building under Windows, make sure the following commands are issued
within the Visual Studio command prompt for version of Visual Studio that
matches the version of Python you're building for.  Python 2.6 and 2.7 needs
Visual Studio 2008, Python 3.3 and 3.4 needs Visual Studio 2010, and Python
3.5 needs Visual Studio 2015.

* Build TextAdapter using the following command:
  `conda build buildscripts/condarecipe --python 3.5`

* TextAdapter can now be installed from the built conda package:
  `conda install iopro --use-local`

Building By Hand
----------------

Note: If building under Windows, make sure the following commands are issued
within the Visual Studio command prompt for version of Visual Studio that
matches the version of Python you're building for.  Python 2.6 and 2.7 needs
Visual Studio 2008, Python 3.3 and 3.4 needs Visual Studio 2010, and Python
3.5 needs Visual Studio 2015.

For building TextAdapter for local development/testing:

1. Install most of the above dependencies into environment called
   'TextAdapter': `conda env create -f environment.yml`

   Be sure to activate new TextAdapter environment before proceeding.


2. Build TextAdapter using Cython/distutils:
   `python setup.py build_ext --inplace`

Testing
-------

Tests can be run by calling the iopro module's test function.  By default
only the TextAdapter tests will be run:

```python
python -Wignore -c 'import iopro; iopro.test()'
```

(Note: `numpy.testing` might produce a FurtureWarning that is not directly
relevant to these unit tests).

