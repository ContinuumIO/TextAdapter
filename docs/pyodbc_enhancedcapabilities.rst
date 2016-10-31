----------------------------------
iopro.pyodbc Enhanced Capabilities
----------------------------------


Demo code showing the enhanced capabilities of iopro.pyodbc submodule
---------------------------------------------------------------------


This demo shows the basic capabilities for the iopro.pyodbc module.  It first will connect with the database of your choice by ODBC, create and fill a new table (market) and then retrieve data with different methods (fetchall(), fetchdictarray() and fetchsarray()).

Author: Francesc Alted, Continuum Analytics


::

    >>> import iopro.pyodbc as pyodbc
    >>> # Open the database (use the most appropriate for you)
    >>> connect_string = 'DSN=odbcsqlite;DATABASE=market.sqlite'  # SQLite
    >>> #connect_string = 'Driver={SQL Server};SERVER=MyWinBox;DATABASE=Test;USER=Devel;PWD=XXX'  # SQL Server
    >>> #connect_string = 'DSN=myodbc3;UID=devel;PWD=XXX;DATABASE=test'  # MySQL
    >>> #connect_string = 'DSN=PSQL;UID=devel;PWD=XXX;DATABASE=test'   # PostgreSQL
    >>> connection = pyodbc.connect(connect_string)
    >>> cursor = connection.cursor()







Create the test table (optional if already done)
------------------------------------------------


::

    >>> try:
    ...     cursor.execute('drop table market')
    ... except:
    ...     pass
    >>> cursor.execute('create table market (symbol_ varchar(5), open_ float, low_ float, high_ float, close_ float, volume_ int)')







Fill the test table (optional if already done)
----------------------------------------------


::

    >>> from time import time
    >>> t0 = time()
    >>> N = 1000*1000
    >>> for i in xrange(N):
    ...     cursor.execute(
    ...         "insert into market(symbol_, open_, low_, high_, close_, volume_)"
    ...         " values (?, ?, ?, ?, ?, ?)",
    ...         (str(i), float(i), float(2*i), None, float(4*i), i))
    >>> cursor.execute("commit")             # not supported by SQLite
    >>> t1 = time() - t0
    >>> print "Stored %d rows in %.3fs" % (N, t1)







Do the query in the traditional way
-----------------------------------


::

    >>> # Query of the full table using the traditional fetchall
    >>> query = "select * from market"
    >>> cursor.execute(query)
    >>> %time all = cursor.fetchall()
    CPU times: user 5.23 s, sys: 0.56 s, total: 5.79 s
    Wall time: 7.09 s








Do the query and get a dictionary of NumPy arrays
-------------------------------------------------


::

    >>> # Query of the full table using the fetchdictarray (retrieve a dictionary of arrays)
    >>> cursor.execute(query)
    >>> %time dictarray = cursor.fetchdictarray()
    CPU times: user 0.92 s, sys: 0.10 s, total: 1.02 s
    Wall time: 1.44 s








Peek into the retrieved data
----------------------------


::

    >>> dictarray.keys()
    ['high_', 'close_', 'open_', 'low_', 'volume_', 'symbol_']
    >>> dictarray['high_']
    array([ nan,  nan,  nan, ...,  nan,  nan,  nan])
    >>> dictarray['symbol_']
    array(['0', '1', '2', ..., '99999', '99999', '99999'], dtype='|S6')







Do the query and get a NumPy structured array
---------------------------------------------


::

    >>> # Query of the full table using the fetchsarray (retrieve a structured array)
    >>> cursor.execute(query)
    >>> %time sarray = cursor.fetchsarray()
    CPU times: user 1.08 s, sys: 0.11 s, total: 1.20 s
    Wall time: 1.99 s








Peek into retrieved data
------------------------


::

    >>> sarray.dtype
    dtype([('symbol_', 'S6'), ('open_', '&lt;f8'), ('low_', '&lt;f8'), ('high_', '&lt;f8'), ('close_', '&lt;f8'), ('volume_', '&lt;i4')])
    >>> sarray[0:10]
    array([('0', 0.0, 0.0, nan, 0.0, 0), ('1', 1.0, 2.0, nan, 4.0, 1),
           ('2', 2.0, 4.0, nan, 8.0, 2), ('3', 3.0, 6.0, nan, 12.0, 3),
           ('4', 4.0, 8.0, nan, 16.0, 4), ('5', 5.0, 10.0, nan, 20.0, 5),
           ('6', 6.0, 12.0, nan, 24.0, 6), ('7', 7.0, 14.0, nan, 28.0, 7),
           ('8', 8.0, 16.0, nan, 32.0, 8), ('9', 9.0, 18.0, nan, 36.0, 9)], 
          dtype=[('symbol_', 'S6'), ('open_', '&lt;f8'), ('low_', '&lt;f8'), ('high_', '&lt;f8'), ('close_', '&lt;f8'), ('volume_', '&lt;i4')])
    >>> sarray['symbol_']
    array(['0', '1', '2', ..., '99999', '99999', '99999'], dtype='|S6')








