.. IOPro documentation master file, created by
   sphinx-quickstart on Thu Aug  9 10:20:31 2012.

-----
IOPro
-----

IOPro loads NumPy arrays (and Pandas DataFrames) directly from files, SQL
databases, and NoSQL stores--including ones with millions of rows--without
creating millions of temporary, intermediate Python objects, or requiring
expensive array resizing operations. 

IOPro provides a drop-in replacement for the 
NumPy functions :code:`loadtxt()` and :code:`genfromtxt()`, but dramatically
improves performance and reduces memory overhead.

IOPro is included with `Anaconda Workgroup and Anaconda Enterprise
subscriptions <https://www.continuum.io/content/anaconda-subscriptions>`_.

To start a 30-day free trial just download and install the IOPro package.

If you already have `Anaconda <http://continuum.io/downloads.html>`_ (free
Python platform) or `Miniconda <http://conda.pydata.org/miniconda.html>`_
installed::

    conda update conda
    conda install iopro

If you do not have Anaconda installed, you can `download it
<http://continuum.io/downloads.html>`_.

IOPro can also be installed into your own (non-Anaconda) Python environment.
For more information about IOPro please contact `sales@continuum.io
<mailto:sales@continuum.io>`_.


Getting started
***REMOVED***--

Some of the basic usage patterns look like these.  Create TextAdapter object
for data source::

***REMOVED***
    >>> adapter = iopro.text_adapter('data.csv', parser='csv')

Define field dtypes (example: set field 0 to unsigned int and field 4 to
float)::

    >>> adapter.set_field_types({0: 'u4', 4:'f4'})

Parse text and store records in NumPy array using slicing notation::

    >>> # read all records
***REMOVED***

    >>> # read first ten records
    >>> array = adapter[0:10]

    >>> # read last record
 ***REMOVED***

    >>> # read every other record
***REMOVED***

User guide
***REMOVED***

.. toctree::
    :maxdepth: 1

    install
    textadapter_examples
    pyodbc_firststeps
    pyodbc_enhancedcapabilities
    pyodbc_cancel

Reference guide
***REMOVED***--

.. toctree::
    :maxdepth: 1

    TextAdapter
    pyodbc
    MongoAdapter
    AccumuloAdapter
    PostgresAdapter
    loadtxt
    genfromtxt

Requirements
***REMOVED***

* python 2.7, or 3.4+
* numpy 1.10+

Python modules (optional):

* boto (for S3 support)
* Pandas (to use DataFrames)

What's new in version 1.9?
***REMOVED***-

The documentation has been substantially updated for version 1.9.0. 
Numba has been removed and the code has been cleaned up, but no other 
features were added or removed. Some refactoring was done that didn't 
change functionality. We recommend that users not use older versions.
See Release notes for additional detail.

Release notes
***REMOVED***

.. toctree::
    :maxdepth: 1

    release-notes

License Agreement
***REMOVED***----

.. toctree::
    :maxdepth: 1

    eula

Previous Versions
***REMOVED***----

This documentation is provided for the use of our customers who have not yet upgraded 
to the current version. 

NOTE: We recommend that users not use older versions of IOPro.

.. toctree::
   :maxdepth: 1

   IOPro 1.8.0 <1.8.0/index>
