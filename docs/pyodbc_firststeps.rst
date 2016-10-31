-------------------------
iopro.pyodbc First Steps
-------------------------

iopro.pyodbc extends pyodbc with methods that allow data to be fetched directly into numpy containers. These functions are faster than regular fetch calls in pyodbc, providing also the convenience of being returned in a container appropriate to fast analysis.


This notebook is intended to be a tutorial on iopro.pyodbc. Most of the material is applicable to pyodbc (and based on pyodbc tutorials). There will be some examples specific to iopro.pyodbc. When that's the case, it will be noted.


Concepts
--------



In pyodbc there are two main classes to understand:
 * connection
 * cursor
 
A connection is, as its name says, a connection to a datasource. A datasource is your database. It may be a database handled by a DBMS or just a plain file.
A cursor allows you to interface with statements. Interaction with queries and other commands is performed through a cursor. A cursor is associated to a connection and commands over a cursor are performed over that connection to the datasource.
In order to use iopro.pyodbc you must import it::

    >>> import iopro.pyodbc as pyodbc

Connection to a datasource
--------------------------


In order to operate with pyodbc you need to connect to a datasource. Typically this will be a database. This is done by creating a connection object.
To create a connection object you need a connection string. This string describes the datasource to use as well as some extra parameters. You can learn more about connection strings here.::

    >>> connection_string = '''DSN=SQLServerTest;DATABASE=Test'''
    >>> connection = pyodbc.connect(connection_string)

pyodbc.connect supports a keyword parameter autocommit. This controls the way the connection is handle. The default value (False) means that the commands that modify the database statements need to be committed explicitly. All commands between commits will form a single transaction. If autocommit is enabled every command will be issued and committed.
It is also possible to change autocommit status after the connection is established.::

    >>> connection.autocommit = True #enable autocommit
    >>> connection.autocommit = False # disable autocommit

When not in autocommit mode, you can end a transaction by either commiting it or rolling it back.::

    In[6]: connection.commit() # commit the transaction
    In[7]: connection.rollback() # rollback the transaction

Note that commit/rollback is always performed at the connection level. pyodbc provides a commit/rollback method in the cursor objects, but they will act on the associated connection.



Working with cursors
--------------------


Command execution in pyodbc is handled through cursors. You can create a cursor from a connection using the cursor() method. The first step is creating a cursor::

    In[8]: cursor = connection.cursor()

With a cursor created, we can start issuing SQL commands using the execute method.



Creating a sample table
-----------------------



First, create a sample table in the database. The following code will create a sample table with three columns of different types.::

    >>> def create_test_table(cursor):
    ...    try:
    ...        cursor.execute('drop table test_table')
    ...    except:
    ...        pass
    ...    cursor.execute('''create table test_table (
    ...                                    name varchar(10),
    ...                                    fval float(24),
    ...                                    ival int)''')
    ...    cursor.commit()
        
    >>> create_test_table(cursor)

Filling the sample table with sample data
-----------------------------------------



After creating the table, rows can be inserted by executing insert into the table. Note you can pass parameters by placing a ? into the SQL statement. The parameters will be taken in order for the sequence appears in the next parameter.::



    >>> cursor.execute('''insert into test_table values (?,?,?)''', ('foo', 3.0, 2))
    >>> cursor.rowcount
    1






Using executemany a sequence of parameters to the SQL statement can be passed and the statement will be executed many times, each time with a different parameter set. This allows us to easily insert several rows into the database so that we have a small test set:::



    >>> cursor.executemany('''insert into test_table values (?,?,?)''', [
    ...                        ('several', 2.1, 3),
    ...                        ('tuples', -1.0, 2),
    ...                        ('can', 3.0, 1),
    ...                        ('be', 12.0, -3),
    ...                        ('inserted', 0.0, -2),
    ...                        ('at', 33.0, 0),
    ...                        ('once', 0.0, 0)
    ...                        ])






Remember that if autocommit is turned off the changes won't be visible to any other connection unless we commit.::



    >>> cursor.commit() # remember this is a shortcut to connection.commit() method







Querying the sample data from the sample table
----------------------------------------------



Having populated our sample database, we can retrieve the inserted data by executing select statements:::



    >>> cursor.execute('''select * from test_table''')
    <pyodbc.Cursor at 0x6803510>






After calling execute with the select statement we need to retrieve the data. This can be achieved by calling fetch methods in the cursor
fetchone fetches the next row in the cursor, returning it in a tuple::



    >>> cursor.fetchone()
    ('foo', 3.0, 2)






fetchmany retrieves several rows at a time in a list of tuples::



    >>> cursor.fetchmany(3)
    [('several', 2.0999999046325684, 3), ('tuples', -1.0, 2), ('can', 3.0, 1)]






fetchall retrieves all the remaining rows in a list of tuples::



    >>> cursor.fetchall()
    [('be', 12.0, -3), ('inserted', 0.0, -2), ('at', 33.0, 0), ('once', 0.0, 0)]






All the calls to any kind of fetch advances the cursor, so the next fetch starts in the row after the last row fetched.
execute returns the cursor object. This is handy to retrieve the full query by chaining fetchall. This results in a one-liner:::



    >>> cursor.execute('''select * from test_table''').fetchall()
    [('foo', 3.0, 2),
     ('several', 2.0999999046325684, 3),
     ('tuples', -1.0, 2),
     ('can', 3.0, 1),
     ('be', 12.0, -3),
     ('inserted', 0.0, -2),
     ('at', 33.0, 0),
     ('once', 0.0, 0)]







iopro.pyodbc extensions
-----------------------



When using iopro.pyodbc it is possible to retrieve the results from queries directly into numpy containers. This is accomplished by using the new cursor methods fetchdictarray and fetchsarray.



fetchdictarray
--------------



fetchdictarray fetches the results of a query in a dictionary. By default fetchdictarray fetches all remaining rows in the cursor.::



    >>> cursor.execute('''select * from test_table''')
    >>> dictarray = cursor.fetchdictarray()
    >>> type(dictarray)
    dict






The keys in the dictionary are the column names:::

    >>> dictarray.keys()
    ['ival', 'name', 'fval']






Each column name is mapped to a numpy array (ndarray) as its value:::



    >>> ', '.join([type(dictarray[i]).__name__ for i in dictarray.keys()])
    'ndarray, ndarray, ndarray'






The types of the numpy arrays are infered from the database column information. So for our columns we get an appropriate numpy type. Note that in the case of name the type is a string of 11 characters even if in test_table is defined as varchar(10). The extra parameter is there to null-terminate the string:::

    >>> ', '.join([repr(dictarray[i].dtype) for i in dictarray.keys()])
    "dtype('int32'), dtype('|S11'), dtype('float32')"






The numpy arrays will have a shape containing a single dimension with the number of rows fetched:::



    >>> ', '.join([repr(dictarray[i].shape) for i in dictarray.keys()])
    '(8L,), (8L,), (8L,)'






The values in the different column arrays are index coherent. So in order to get the values associated to a given row it suffices to access each column using the appropriate index. The following snippet shows this correspondence:::



    >>> print '\n'.join(
    ... [', '.join(
    ...     [repr(dictarray[i][j]) for i in dictarray.keys()]) 
    ...         for j in range(dictarray['name'].shape[0])])
    2, 'foo', 3.0
    3, 'several', 2.0999999
    2, 'tuples', -1.0
    1, 'can', 3.0
    -3, 'be', 12.0
    -2, 'inserted', 0.0
    0, 'at', 33.0
    0, 'once', 0.0







Having the results in numpy containers makes it easy to use numpy to analyze the data:::



    >>> import numpy as np
    >>> np.mean(dictarray['fval'])
    6.5124998092651367






fetchdictarray accepts an optional parameter that places an upper bound to the number of rows to fetch. If there are not enough elements left to be fetched in the cursor the arrays resulting will be sized accordingly. This way it is possible to work with big tables in chunks of rows.::



    >>> cursor.execute('''select * from test_table''')
    >>> dictarray = cursor.fetchdictarray(6)
    >>> print dictarray['name'].shape
    (6L,)
    >>> dictarray = cursor.fetchdictarray(6)
    >>> print dictarray['name'].shape
    (2L,)

fetchsarray
-----------



fetchsarray fetches the result of a query in a numpy structured array.::



    >>> cursor.execute('''select * from test_table''')
    >>> sarray = cursor.fetchsarray()
    >>> print sarray
    [('foo', 3.0, 2) ('several', 2.0999999046325684, 3) ('tuples', -1.0, 2)
     ('can', 3.0, 1) ('be', 12.0, -3) ('inserted', 0.0, -2) ('at', 33.0, 0)
     ('once', 0.0, 0)]



The type of the result is a numpy array (ndarray):::



    >>> type(sarray)
    numpy.ndarray






The dtype of the numpy array contains the description of the columns and their types:::



    >>> sarray.dtype
    dtype([('name', '|S11'), ('fval', '&lt;f4'), ('ival', '&lt;i4')])






The shape of the array will be one-dimensional, with cardinality equal to the number of rows fetched:::



    >>> sarray.shape
    (8L,)






It is also possible to get the shape of a column. In this way it will look similar to the code needed when using dictarrays::



    >>> sarray['name'].shape
    (8L,)






In a structured array it is as easy to access data by row or by column:::



    >>> sarray['name']
    array(['foo', 'several', 'tuples', 'can', 'be', 'inserted', 'at', 'once'], 
          dtype='|S11')







    >>> sarray[0]
    ('foo', 3.0, 2)






It is also very easy and efficient to feed data into numpy functions:::



    >>> np.mean(sarray['fval'])
    6.5124998092651367







fetchdictarray vs fetchsarray
-----------------------------



Both methods provide ways to input data from a database into a numpy-friendly container. The structured array version provides more flexibility extracting rows in an easier way. The main difference is in the memory layout of the resulting object. An in-depth analysis of this is beyond the scope of this notebook. Suffice it to say that you can view the dictarray laid out in memory as an structure of arrays  (in fact, a dictionary or arrays), while the structured array would be laid out in memory like an array of structures. This can make a lot of difference performance-wise when working with large chunks of data.



