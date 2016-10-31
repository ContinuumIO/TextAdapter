----------------------------------------------
PostgresAdapter, PostGIS, and GreenPlum
----------------------------------------------

.. contents::

The PostgresAdapter module reads data from PostresSQL based databases and produces
a NumPy array or a Pandas Dataframe containing the parsed data. The PostgresAdapter
can be used to access data from PostgresSQL and GreenPlum, and has enhancements
to support PostGIS points, lines, multilines, polygons, and multipolygons. The
following features are currently implemented:

* The PostgresAdapter engine is written in C to ensure returned data is parsed
  as fast as data can be read from the source. Data is read and parsed in small
  chunks instead of reading entire data into memory at once.

* Python slicing notation can be used to specify a subset of records to be
  read from the data source.

* A subset of columns can be specified to be returned instead of returning all
  columns for the records.

Adapter Methods
---------------
PostgreSQL Adapter Constructor:

**PostgresAdapter** (connection_uri, table=None, query=None, field_filter=None, dataframe=False, field_names=None, field_types=None, field_shapes=None):

    | Create an adaptor for connecting to a PostgreSQL based database.

    | connection_uri: string URI describing how to connect to database
    | table: string, name of table to read records from. Only table
    | parameter or query parameter can be set, but not both.
    | query: string, custom query to use for reading records. Only query
      parameter or table parameter can be set, but not both.
      field_filter parameter cannot be set when query parameter is
      set (since it is trivial to specify fields in query string).
    | field_filter: names of fields include in query (only valid when table
      parameter is specified)
    | dataframe: bool, return results as dataframe instead of array
    | field_names: list, names of fields in output array or dataframe.
      Defaults to database table column names.
    | field_types: list, NumPy dtype for each field in output array
      or dataframe. Defaults to database table column types.
    | field_shapes: list, shape of each field value for geometry field
      types with variable length data. For example, for a
      'path' database column with 2d points, the points of
      the path will be stored in a list object by default.
      If a field shape of '10' is specified, the points will
      be stored in a 2x10 float subarray (2 floats per point* 10 points max).
      A field shape of the form (x,y) should be specifed for types like
      multipolygon where x is the max number of polygons and y is the max
      length of each polygon (the size of the point is inferred).

**close** ()
    | Close connection to the database.

The PostgresAdapter object supports array slicing:

    | Read all records:
      adapter[:]

    | Read first 100 records:
      adapter[0:100]

    | Read last record:
      adapter[-1]

    | Read every other record:
      adapter[::2]

Adapter Properties
------------------
**num_records** (readonly)
    | Get number of records that will be returned from table or custom query.

**num_fields** (readonly)
    | Get number of fields in records that will be returned from table
      or custom query

**field_names**
    | Get/set names of fields in final array or dataframe. Field names can be
      set by specifying a list of names, or dict mapping of field number to
      field name. If names is a list, the length of list must match the number
      of fields in data set. If names is a dict, the field name from the database
      will be used if no name in dict is specified for that field.

**field_types**
    | Get/set field types in final array or dataframe. Field types can be set
      by specifying a list of NumPy dtypes, or a dict mapping of field number
      or name to field type. If types is a list, the length of list must match
      the number of fields in data set. If types is a dict, the field type from
      the database will be used if type is not specified in dict.

**field_shapes**
    | Get/set field shapes for variable length fields. Field shapes can be set
      by specifying a list of shape tuples (or a single integer if shape has
      one dimension), or a dict mapping of field number or name to field shape.
      If shapes is a list, the length of the list must match the number of fields
      in data set. A value of None or zero for a field, or an unspecified shape,
      means that an infinite length value will be allowed for field, and value will be
      stored in Python list object if field is PostgreSQL geometry type, or
      as Well Known Text string objects if field is PostGIS type.

Basic Usage
-----------

Create PostgresAdapter object for data source::

    >>> import iopro
    >>> adapter = iopro.PostgresAdapter('host=localhost dbname=db_name user=user table=table_name')

IOPro adapters use slicing to retrieve data. To retrieve records from the table
or query, the standard NumPy slicing notation can be used:

    >>> # read all records
    >>> array = adapter[:]

    >>> # read first ten records
    >>> array = adapter[0:10]

    >>> # read last record
    >>> array = adapter[-1]

    >>> # read every other record
    >>> array = adapter[::2]

The PostgreSQL adapter has a few properties that we can use to find out
more about our data. To get the number of records in our dataset:

    >>> adapter.num_records
    5

or the number of fields:

    >>> adapter.num_fields
    5

To find the names of each field:

    >>> adapter.field_names
    ['field1', 'real', 'name', 'point2d', 'multipoint3d']

These names come from the names of the columns in the database and are used by
default for the field names in the NumPy array result. These names can be changed
by setting the field names property using a list of field names:

    >>> adapter.field_names = ['field1', 'field2', 'field3', 'field4', 'field5']
    >>> adapter[:].dtype
    dtype([('field1', '<i4'), ('field2', '<f4'), ('field3', '<U10'), ('field4', '<f8', (2,)), ('field5', 'O')])

Individual fields can also be set by using a dict, where the key is the field
number and the value is the field name we want:

    >>> adapter.field_names = {1: 'AAA'}
    >>> adapter[:].dtype
    dtype([('integer', '<i4'), ('AAA', '<f4'), ('string', '<U10'), ('point2d', '<f8', (2,)), ('multipoint3d', 'O')])

To find out the NumPy dtype of each field:

    >>> adapter.field_types
    ['i4', 'f4', 'U10', 'f8', 'O']

Similar to the field names property, the types property can be set using a list
or dict to force a field to be cast to a specific type:

    >>> adapter.field_types = {0: 'f4', 1: 'i4', 2: 'U3', 4: 'O'}

To filter the fields returned by passing a list of field names to the constructor:

    >>> adapter = iopro.PostgresAdapter('host=localhost dbname=db_name user=user',
                                         table='data',
                                         field_filter=['field1', 'field2'])

For fields like path or multipoint3d with a variable length, the adapter will return
values as a list of tuples containing the float components of each point (if a
PostgreSQL geometric type) or as string objects in Well Known Text format (if a
PostGIS type). For improved performance, a field shape can be specified which
will set the max dimensions of the field values. For example, a multipoint3d
field can be set to have a maximum of two points so that each set of 3d points
will be stored in a 2x3 subarray of floats:

    >>> adapter = iopro.PostgresAdapter('host=localhost dbname=db_name user=user'
                                        table='data',
                                        field_filter=['multipoint3d'],
                                        field_shapes={'multipoint3d': 2})
    >>> adapter[:]
    array([([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],),
           ([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],),
           ([[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],),
           ([[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],),
           ([[24.0, 25.0, 26.0], [27.0, 28.0, 29.0]],)],
          dtype=[('multipoint3d', '<f8', (2, 3))])

For more advanced queries, a custom select query can be passed to the constructor.
Either a table name or a custom query can be passed to the constructor, but not
both.

    >>> adapter = iopro.PostgresAdapter('host=localhost dbname=db_name user=user',
                                         query='select integer, string from data where data.integer > 2')
    >>> adapter[:]

Data can also be returned as a pandas dataframe using the adapter constructor's
dataframe' argument:

    >>> adapter = iopro.PostgresAdapter('host=localhost dbname=iopro_tutorial user=jayvius',
                                        table='data',
                                        dataframe=True)

To retrieve some PostGIS data that falls within a given bounding box:

    >>> adapter = iopro.PostgresAdapter('host=localhost dbname=db_name user=user',
                                        query='select integer, point2d from data '
                                              'where data.point2d @ ST_MakeEnvelope(0, 0, 4, 4)')

Close database connection:
    >>> adapter.close()
