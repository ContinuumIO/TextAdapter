-----
TextAdapter
-----

TextAdapter loads NumPy arrays (and Pandas DataFrames) directly from files,
including ones with millions of rows--without creating millions of
temporary, intermediate Python objects, or requiring expensive array
resizing operations.

TextAdapter provides a drop-in replacement for the NumPy functions
:code:`loadtxt()` and :code:`genfromtxt()`, but dramatically improves
performance and reduces memory overhead.

If you already have `Anaconda <http://continuum.io/downloads.html>`_ (free
Python platform) or `Miniconda <http://conda.pydata.org/miniconda.html>`_
installed::

    conda update conda
    conda install TextAdapter

If you do not have Anaconda installed, you can `download it
<http://continuum.io/downloads.html>`_.


Getting started
---------------

Some of the basic usage patterns look like these.  Create TextAdapter object
for data source::

***REMOVED***
    >>> adapter = TextAdapter.text_adapter('data.csv', parser='csv')

Define field dtypes (example: set field 0 to unsigned int and field 4 to
float)::

    >>> adapter.set_field_types({0: 'u4', 4:'f4'})

Parse text and store records in NumPy array using slicing notation::

    >>> # read all records
    >>> array = adapter[:]

    >>> # read first ten records
    >>> array = adapter[0:10]

    >>> # read last record
    >>> array = adapter[-1]

    >>> # read every other record
    >>> array = adapter[::2]

User guide
----------

.. toctree::
    :maxdepth: 1

    install
    textadapter_examples

Reference guide
---------------

.. toctree::
    :maxdepth: 1

    TextAdapter
    loadtxt
    genfromtxt

Requirements
------------

* python 2.7, or 3.5+
* numpy 1.10+

Python modules (optional):

* boto (for S3 support)
* Pandas (to use DataFrames)

What's new in version 2.0?
--------------------------

The documentation has been substantially updated for version IOPro
1.9.0 and TextAdapter 2.0.

Numba has been removed and the code has been cleaned up, but no other
features were added or removed.  Some refactoring was done that didn't
change functionality.  We recommend that users not use older versions.
See Release notes for additional detail.

Release notes
-------------

.. toctree::
    :maxdepth: 1

    release-notes

