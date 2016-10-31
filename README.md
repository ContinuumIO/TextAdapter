IOPro
======

IOPro is a Python module containing optimized data adapters for importing
data from a variety of data sources into NumPy arrays and Pandas DataFrame.
Current data adapters include TextAdapter for CSV like text files,
MongoAdapter for mongo databases, PostgresAdapter for PostgreSQL databases,
AccumuloAdapter for Accumulo databases, and an optimized pyodbc module for
accessing any relational database that supports the ODBC interface (SQL
Server, PostgreSQL, MySQL, etc).

Build Requirements
***REMOVED***

Building IOPro requires a number of dependencies. In addition to a C/C++ dev
environment, the following modules are needed, which can be installed via conda
(the postgresql and thrift modules are not available in the public Anaconda
channel and will need to be built separately using the conda build recipes as
described below):

* NumPy
* Pandas
* zlib 1.2.8 (C lib, needed for TextAdapter)
* pcre 8.31 (C lib, needed for TextAdapter)
* mongo-driver 0.7.1 (C lib, needed for MongoAdapter)
* postgresql 0.9.3 (C lib, needed for PostgresAdapter)
* Thrift 0.9.3 (C++ interface, needed for AccumuloAdapter)
* openssl 1.0.2 (C lib, needed for Thrift lib)
* unixodbc 2.3.4 (C lib, Linux/OSX, needed for pyodbc)

Building Conda Package
***REMOVED******REMOVED***

Note: If building under Windows, make sure the following commands are issued
within the Visual Studio command prompt for version of Visual Studio that
matches the version of Python you're building for.  Python 2.6 and 2.7 needs
Visual Studio 2008, Python 3.3 and 3.4 needs Visual Studio 2010, and Python
3.5 needs Visual Studio 2015.

1. Build the postgresql dependency using the following command (replace Python
   version number with desired version):
   `conda build buildscripts/dependency-recipes/postgres --python 3.5`

2. Build the thrift dependency using the following command:
   `conda build buildscripts/dependency-recipes/thrift --python 3.5`

3. Build IOPro using the following command:
   `conda build buildscripts/condarecipe --python 3.5`

4. IOPro can now be installed from the built conda package:
   `conda install iopro --use-local`

Building By Hand
***REMOVED***---

Note: If building under Windows, make sure the following commands are issued
within the Visual Studio command prompt for version of Visual Studio that
matches the version of Python you're building for.  Python 2.6 and 2.7 needs
Visual Studio 2008, Python 3.3 and 3.4 needs Visual Studio 2010, and Python
3.5 needs Visual Studio 2015.

For building IOPro for local development/testing:

1. Install most of the above dependencies into environment called 'iopro':
   `conda env create -f environment.yml`

   Be sure to activate new iopro environment before proceeding.

2. Build the postgresql and thrift conda packages as described above. Install
   into iopro environment using commands:
   `conda install postgresql --use-local`; 
   `conda install thrift --use-local`

3. (Linux/OSX) Install unixodbc dependency into iopro environment:
   `conda install unixodbc`.  
   May be on a different channel on some platforms, e.g.:
   `conda install -c derickl unixodb
  

4. Build IOPro using Cython/distutils:
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

To test the MongoAdapter, you'll need to have a running Mongo database.  The
mongo tests can be run with the iopro module's test_mongo function:

```python
iopro.test_mongo(host, port)
```

where host is the host name where the Mongo database is running, and the
port is the port of the Mongo database.  By default, these are 'localhost'
and 27017.  The MongoAdapter tests will generate their own test data by
creating a collection called 'MongoAdapter_tests' in the Mongo database
specified by the above parameters.

To test the PostgresAdapter, you'll need a running PostgreSQL database.
There is a script called 'iopro/tests/setup_postgresql_data.py' that can be
used to generate test data for the PostgresAdapter tests (modify as needed
for your database).  The PostgreSQL tests can be run with the iopro module's
test_postgres function:


```python
iopro.test_postgres(host, dbname, user)
```

where the host is the host name where the PostgreSQL database is running,
the dbname is the name of the test database, and the user is a valid user
for the test database.

To test the AccumuloAdapter, you'll need a running Accumulo database.
Accumulo is a Java based key/value store with several moving parts.  I've
found that the easiest way to get going is to download and run the following
docker image for Accumulo:

> https://github.com/mraad/accumulo-docker

Follow the instructions in the github repo README for running Accumulo.  Be
sure to also run the Accumulo C++ proxy inside the docker container (this is
what the AccumuloAdapter actually talks to).

There is a script called 'iopro/tests/setup_accumulo_data.py' that can be
used to generate test data for the AccumuloAdapter tests (modify as needed
for your database).  The Accumulo tests can be run with the iopro module's
test_accumulo function:

```python
iopro.test_accumulo(host, user, password)
```

where the host is the host name where the test Accumulo database is running,
the user is a valid user for the test Accumulo database, and the password is
the password for the user.
