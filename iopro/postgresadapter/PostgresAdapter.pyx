from __future__ import division
import numpy
cimport numpy
from cpython.ref cimport Py_INCREF, Py_DECREF, PyObject
import sys
from libc.string cimport memcpy, memset
from libc.stdlib cimport calloc, free
from libc.stdio cimport printf
import math
import pandas as pd
from collections import OrderedDict

numpy.import_array()

# Numpy unicode strings should always be 4 bytes per character,
# but check just in case
numpy_char_width = numpy.array(['x'], dtype='U1').itemsize

cdef extern from '_stdint.h':
    # Actual type lengths are defined in _stdint.h
    # Sizes here are just place holders
    ctypedef unsigned long long uint64_t
    ctypedef unsigned int uint32_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned char uint8_t
    ctypedef long long int64_t
    ctypedef int int32_t
    ctypedef short int16_t
    ctypedef char int8_t

cdef extern from 'postgres_adapter.h':
    ctypedef struct postgres_adapter_t:
        const char *client_encoding
        PGresult *result
        char *result_error_msg
        int postgis_geometry_oid
    ctypedef enum AdapterError:
        ADAPTER_SUCCESS
        ADAPTER_ERROR_INVALID_QUERY
        ADAPTER_ERROR_INVALID_QUERY_VALUE
        ADAPTER_ERROR_INVALID_TYPE
        ADAPTER_ERROR_INVALID_GIS_TYPE
        ADAPTER_ERROR_PARSE_GIS
    cdef int BOOLOID
    cdef int CHAROID
    cdef int INT8OID
    cdef int INT2OID
    cdef int INT4OID
    cdef int TEXTOID
    cdef int FLOAT4OID
    cdef int FLOAT8OID
    cdef int BPCHAROID
    cdef int VARCHAROID
    cdef int TEXTOID
    #cdef int NUMERICOID
    cdef int POINTOID
    cdef int LINEOID
    cdef int LSEGOID
    cdef int BOXOID
    cdef int PATHOID
    cdef int POLYGONOID
    cdef int CIRCLEOID

    postgres_adapter_t* open_postgres_adapter(const char *connection_uri)
    void close_postgres_adapter(postgres_adapter_t *adapter)
    AdapterError exec_query(postgres_adapter_t *adapter, const char *query)
    void clear_query(postgres_adapter_t *adapter)
    int get_num_records(postgres_adapter_t *adapter)
    int get_num_fields(postgres_adapter_t *adapter)
    void set_field_dim_size(postgres_adapter_t *adapter, int field, int dim, int size)
    void clear_field_size(postgres_adapter_t *adapter, int field)
    AdapterError read_records(postgres_adapter_t *adapter,
                              int start,
                              int stop,
                              int step,
                              char **output,
                              int *num_records_found,
                              int cast_types,
                              void *src,
                              void *dst,
                              int dataframe)

    cdef int CAST_BUFFER_SIZE
    int get_field_type(PGresult *result, int field)
    int get_field_length(PGresult *result, int field)
    char * get_field_data(PGresult *result, int row, int field)
    int get_postgis_geometry_oid(postgres_adapter_t *adapter)

cdef extern from 'postgis_fields.h':
    uint32_t get_gis_type(char *data, int *has_srid)
    int get_gis_point_size(uint32_t gis_type)
    cdef int GIS_POINT2D
    cdef int GIS_POINT3D
    cdef int GIS_POINT4D
    cdef int GIS_LINE2D
    cdef int GIS_LINE3D
    cdef int GIS_LINE4D
    cdef int GIS_POLYGON2D
    cdef int GIS_POLYGON3D
    cdef int GIS_POLYGON4D
    cdef int GIS_MULTIPOINT2D
    cdef int GIS_MULTIPOINT3D
    cdef int GIS_MULTIPOINT4D
    cdef int GIS_MULTILINE2D
    cdef int GIS_MULTILINE3D
    cdef int GIS_MULTILINE4D
    cdef int GIS_MULTIPOLYGON2D
    cdef int GIS_MULTIPOLYGON3D
    cdef int GIS_MULTIPOLYGON4D

cdef extern from 'libpq-fe.h':
   ctypedef struct PGresult
   char *PQfname(PGresult *res, int field_num) 

# Mapping that specifies which numpy dtype to
# convert each postgres column type to
pg_to_numpy_mapping = {BOOLOID: 'B',
                       CHAROID: 'S1',
                       INT8OID: 'i8',
                       INT2OID: 'i2',
                       INT4OID: 'i4',
                       FLOAT4OID: 'f4',
                       FLOAT8OID: 'f8',
                       BPCHAROID: 'U',
                       VARCHAROID: 'U',
                       TEXTOID: 'O',
                       #NUMERICOID: 'O',
                       POINTOID: 'f8',
                       LINEOID: 'f8',
                       LSEGOID: 'f8',
                       BOXOID: 'f8',
                       PATHOID: 'f8',
                       POLYGONOID: 'f8',
                       CIRCLEOID: 'f8'}

exceptions = {ADAPTER_ERROR_INVALID_QUERY: RuntimeError('Invalid PostgreSQL query'),
              ADAPTER_ERROR_INVALID_QUERY_VALUE: RuntimeError('Invalid PostgreSQL query result'),
              ADAPTER_ERROR_INVALID_TYPE: RuntimeError('PostgreSQL type not supported'),
              ADAPTER_ERROR_INVALID_GIS_TYPE: RuntimeError('PostGIS type not supported'),
              ADAPTER_ERROR_PARSE_GIS: RuntimeError('PostGIS field data could not be parsed')}

cdef public PyObject* create_list(void **output):
    """
    Create new list and store in output array or dataframe. Caller is responsible
    for appending values to list.
    Inputs:
        output: pointer to output pointer (output pointer will be incremented
                after list pointer is stored in it)
    Outputs:
        pointer to list
    """
    cdef object py_list = <list>[]
    cdef PyObject ***object_ptr
    # Increment ref count since this object will be referenced by final numpy array
    Py_INCREF(py_list)
    if output != NULL:
        object_ptr = <PyObject***>(output)
        object_ptr[0][0] = <PyObject*>py_list
        object_ptr[0] += 1
    return <PyObject*>py_list

cdef public void create_string(void **output, const char *value):
    cdef object py_string = value.decode('utf8')
    cdef PyObject ***object_ptr
    Py_INCREF(py_string)
    if output != NULL:
        object_ptr = <PyObject***>output
        object_ptr[0][0] = <PyObject*>py_string
        object_ptr[0] += 1

cdef public void add_to_list(void *py_list_ptr, double *items, int num_items):
    """
    Append values to list stored in output array or dataframe
    Inputs:
        py_list_ptr: pointer to python list object
        items: array of double values
        num_items: length of items array
    """
    cdef object py_list = <object>py_list_ptr
    cdef object py_sublist = <list>[]
    if num_items > 1:
        for i in range(num_items):
            py_sublist.append(items[i])
        py_list.append(tuple(py_sublist))
    else:
        py_list.append(items[0])

cdef public void append_list(void *py_list_ptr, void *py_sublist_ptr):
    cdef object py_list = <object>py_list_ptr
    cdef object py_sublist = <object>py_sublist_ptr
    py_list.append(py_sublist)

cdef public AdapterError convert_str2object(const char *input_str, const char *encoding, void **output):
    """
    Convert c string to Python unicode object
    Inputs:
        input_str: pointer to input c string
        encoding: encoding of input c string
        output: pointer to pointer of numpy array memory chunk. Pointer to memory
                chunk will be incremented by size of Python object pointer.
    """
    cdef object unicode_obj = input_str.decode(encoding)
    # Increment ref count since this object will be referenced by final numpy array
    Py_INCREF(unicode_obj)
    cdef PyObject ***object_ptr = <PyObject***>output
    object_ptr[0][0] = <PyObject*>unicode_obj
    object_ptr[0] += 1
    return ADAPTER_SUCCESS

cdef public AdapterError convert_str2str(const char *input_str, const char *encoding,
        void **output, int max_chars):
    """
    Convert c string to numpy UCS2 or UCS4 unicode value
    Inputs:
        input_str: pointer to input c string
        encoding: encoding of input c string
        output: pointer to pointer of numpy array memory chunk. Pointer to memory
                chunk will be incremented by size of Python object pointer.
    """
    cdef object unicode_obj = input_str.decode(encoding)

    utf_bytes = unicode_obj.encode('utf32')
    cdef char *utf_bytes_ptr = utf_bytes

    str_len = len(unicode_obj)
    if str_len > max_chars:
        str_len = max_chars

    cdef char **output_ptr = <char**>output
    if str_len == 0:
        memset(output_ptr[0], 0, max_chars * numpy_char_width)
    else:
        memcpy(output_ptr[0], utf_bytes_ptr + <int>numpy_char_width, str_len * numpy_char_width)
    output_ptr[0] += <int>numpy_char_width * <int>max_chars

cdef public void cast_arrays(numpy.ndarray dst, numpy.ndarray src, int offset, int num):
    """
    Copy values from src array (buffer) to dst array (final array), forcing a
    cast.
    Inputs:
        dst: final array set to user specified dtypes
        src: buffer array set to database table/query dtypes
        offset: first record in final array to copy to
        num: number of records to copy
    """
    dst[offset:offset+num] = src[0:num]

cdef public void cast_dataframes(dst, src, int offset, int num, int num_fields):
    """
    Copy values from src array (buffer) to dst array (final array), forcing a
    cast.
    Inputs:
        dst: final array set to user specified dtypes
        src: buffer array set to database table/query dtypes
        offset: first record in final array to copy to
        num: number of records to copy
    """
    for i in range(num_fields):
        dst[dst.columns[i]].values[offset:offset+num] = src[src.columns[i]].values[0:num]


cdef public void* convert_list_to_array(void *py_list_ptr):
    """
    Convert Python string to NumPy array (infer type)
    Inputs:
        py_list_ptr: pointer to Python list
        output: pointer to a NumPy array
    """
    cdef object py_list = <object>py_list_ptr

    # create a temporary reference to the array
    array = numpy.array(py_list)
    cdef object py_array = <object>array

    # Increment ref count on the array, decrement ref count on the list
    Py_INCREF(py_array)
    Py_DECREF(py_list)
    return <void*>py_array


cdef class PostgresAdapter:
    """
    PostgreSQL adapter for reading data from PostgreSQL database into
    NumPy array or Pandas dataframe.

    Constructor Inputs:
        connection_uri: string URI describing how to connect to database
        table: string, name of table to read records from. Only table
               parameter or query parameter can be set, but not both.
        query: string, custom query to use for reading records. Only query
               parameter or table parameter can be set, but not both.
               field_filter parameter cannot be set when query parameter is
               set (since it is trivial to specify fields in query string).
        field_filter: names of fields include in query (only valid when table
                      parameter is specified)
        dataframe: bool, return results as dataframe instead of array
        field_names: list, names of fields in output array or dataframe.
                     Defaults to database table column names.
        field_types: list, NumPy dtype for each field in output array
                     or dataframe. Defaults to database table column types.
        field_shapes: list, shape of each field value for geometry field
                      types with variable length data. For example, for a
                      'path' database column with 2d points, the points of
                      the path will be stored in a list object by default.
                      If a field shape of '10' is specified, the points will
                      be stored in a 2x10 float subarray (2 floats per point
                      * 10 points max). A field shape of the form (x,y) should
                      be specifed for types like multipolygon where x is the
                      max number of polygons and y is the max length of each
                      polygon (the size of the point is inferred).
    """
    cdef postgres_adapter_t *_adapter
    cdef object _field_names
    cdef object _field_types
    cdef object _field_shapes
    cdef object _dataframe_result

    def __cinit__(self, connection_uri,
                  table=None,
                  query=None,
                  field_filter=None,
                  dataframe=False,
                  field_names=None,
                  field_types=None,
                  field_shapes=None):
        """
        PostgreSQL adapter constructor.
        See above for constructor argument descriptions.
        """
        if table is None and query is None:
            raise ValueError('Either table or query must be set')
        if table is not None and query is not None:
            raise ValueError('Table and query cannot both be set')

        if field_filter is not None and len(field_filter) == 0:
            field_filter = None
        if table is None and field_filter is not None:
            raise ValueError('Field filter cannot be set for query')

        self._adapter = open_postgres_adapter(connection_uri.encode('utf8'))
        if not self._adapter:
            raise IOError("PostgresAdapter: Unable to connect to %s" % connection_uri)

        pg_to_numpy_mapping[self._adapter.postgis_geometry_oid] = 'f8'

        if table is not None:
            if field_filter is None:
                columns = '*'
            else:
                columns = ','.join(field_filter)
            query = 'select {0} from {1}'.format(columns, table)
            result = exec_query(self._adapter, query.encode('utf8'))
            if result != ADAPTER_SUCCESS:
                msg = 'Could not perform query on table "{0}"'.format(table)
                if columns != '*':
                    msg += ' with columns "{0}"'.format(columns)
                msg += '.'
                if self._adapter.result_error_msg != NULL:
                    msg += ' PostgreSQL {0}'.format(self._adapter.result_error_msg.decode('utf8'))
                raise ValueError(msg)
        else:
            result = exec_query(self._adapter, query.encode('utf8'))
            msg = 'Could not perform query "{0}".'.format(query)
            if self._adapter.result_error_msg != NULL:
                msg += ' PostgreSQL {0}'.format(self._adapter.result_error_msg.decode('utf8'))
            if result != ADAPTER_SUCCESS:
                raise ValueError(msg)

        self.set_field_names(field_names)
        self.set_field_types(field_types)
        self.set_field_shapes(field_shapes)
        self._dataframe_result = dataframe

    def close(self):
        """
        Close PostgreSQL connection
        """
        if self._adapter != NULL:
            close_postgres_adapter(self._adapter)
            self._adapter = NULL

    def __dealloc__(self):
        """
        PostgreSQL adapter destructor
        """
        self.close()

    def _check_connection(self):
        if self._adapter == NULL:
            raise RuntimeError('Connection already closed. '
                               'Please create a new adapter.')

    @property
    def num_records(self):
        """
        Get number of records that will be returned from table or custom query
        """
        self._check_connection()
        return get_num_records(self._adapter)

    @property
    def num_fields(self):
        """
        Get number of fields in records that will be returned from table
        or custom query
        """
        self._check_connection()
        return get_num_fields(self._adapter)

    def get_field_names(self):
        """
        Get names of fields in final array or dataframe
        """
        self._check_connection()
        return self._field_names

    def set_field_names(self, names):
        """
        Set names to assign to fields in final array or dataframe
        Inputs:
            names: list of names to assign to fields in result, or dict mapping
                   field number to field name. If names is a list, the length
                   of list must match the number of fields in data set.
                   If names is a dict, the field name from the database will
                   be used if no name in dict is specified for that field.
        """
        self._check_connection()
        if names is not None and len(names) == 0:
            names = None

        def make_unique_name(field_name):
            suffix = 1
            while field_name + str(suffix) in self._field_names:
                suffix += 1
            return field_name + str(suffix)

        if names is None:
            self._field_names = []
            for i in range(self.num_fields):
                field_name = PQfname(self._adapter.result, i).decode(self._adapter.client_encoding.decode())
                # Since libpq returns 'st_astext' as the column name for every
                # PostGIS column retrieved as text, we need to number multiple
                # st_astext columns to make them unique.
                if field_name in self._field_names:
                    field_name = make_unique_name(field_name)
                self._field_names.append(field_name)
        elif isinstance(names, dict):
            self._field_names = []
            for field_num in names.keys():
                if field_num >= self.num_fields:
                    raise ValueError('Invalid field number {0}'.format(field_num))
            for i in range(self.num_fields):
                if i in names.keys():
                    self._field_names.append(names[i])
                else:
                    field_name = PQfname(self._adapter.result, i).decode(self._adapter.client_encoding.decode())
                    # Since libpq returns 'st_astext' as the column name for every
                    # PostGIS column retrieved as text, we need to number multiple
                    # st_astext columns to make them unique.
                    if field_name in self._field_names:
                        field_name = make_unique_name(field_name)
                    self._field_names.append(field_name)
        else:
            if len(names) != self.num_fields:
                raise ValueError('Number of field names does not match number of fields')
            self._field_names = names

    field_names = property(get_field_names, set_field_names)

    def _get_default_types(self):
        """
        Get list of types based on types in database table/query. These will be
        the default types of final array or dataframe if user does not specify
        any types.
        """
        cdef int has_srid
        self._check_connection()
        types = []
        dtype_len = []
        for i in range(self.num_fields):
            pg_type = get_field_type(self._adapter.result, i)
            field_length = get_field_length(self._adapter.result, i)
            gis_field_type = None
            if pg_type == self._adapter.postgis_geometry_oid:
                data = get_field_data(self._adapter.result, 0, i)
                gis_field_type = get_gis_type(data, &has_srid)

            if pg_type not in pg_to_numpy_mapping:
                raise RuntimeError('postgresql type {0} not supported'.format(pg_type))
            numpy_type = pg_to_numpy_mapping[pg_type]

            if pg_type in [BPCHAROID, VARCHAROID] and not self._dataframe_result:
                numpy_type += str(get_field_length(self._adapter.result, i))
                types.append(numpy_type)
                dtype_len.append(1)
            elif pg_type == TEXTOID or (pg_type in [BPCHAROID, VARCHAROID] and self._dataframe_result):
                types.append('O')
                dtype_len.append(1)
            elif (pg_type == self._adapter.postgis_geometry_oid and 
                    (gis_field_type == <uint32_t>GIS_POINT2D or
                     gis_field_type == <uint32_t>GIS_POINT3D or
                     gis_field_type == <uint32_t>GIS_POINT4D)):
                if self._field_types[i] in [None, 'O']:
                    types.append('O')
                    dtype_len.append(1)
                else:
                    types.append('f8')
                    dtype_len.append(field_length)
            elif field_length < 0 or pg_type == self._adapter.postgis_geometry_oid:
                if (self.field_shapes is None or self.field_names[i] not in self.field_shapes or self._dataframe_result):
                    types.append('O')
                    dtype_len.append(1)
                else:
                    # For variable length fields that are composed of things
                    # like points, the user specified max field size is in terms
                    # of points. We need to get the max number of items, which
                    # for points is two items per point (each point contains an
                    # x an y coordinate).
                    shape = self.field_shapes[self.field_names[i]]
                    types.append(numpy_type)
                    dtype_len.append(shape)
            elif self._dataframe_result and field_length > 1:
                types.append('O')
                dtype_len.append(1)
            else:
                types.append(numpy_type)
                dtype_len.append(field_length)
        return types, dtype_len

    def get_field_types(self):
        """
        Get field types in final array or dataframe
        """

        # Reporting the field types may depend on whether the user has specified
        # any field shapes using the field_shapes property. Because of this, we
        # need to recompute the field types list on the fly in the field types
        # getter, rather than compute once in the field types setter.
        self._check_connection()
        default_types, dtype_len = self._get_default_types()

        field_types = []
        for i, field_type in enumerate(self._field_types):
            if field_type is None:
                field_types.append(default_types[i])
            else:
                if field_type in ['U', 'S']:
                    field_types.append(field_type + str(get_field_length(self._adapter.result, i)))
                else:
                    field_types.append(field_type)
        return field_types

    def set_field_types(self, types):
        """
        Set types to assign to fields in final array or dataframe
        Inputs:
            names: list of types to assign to fields in result, or dict mapping
                   field number or name to field type. If types is a list, the
                   length of list must match the number of fields in data set.
                   If types is a dict, the field type from the database will be
                   used if type is not specified in dict.
        """
        self._check_connection()

        self._field_types = [None] * self.num_fields
        if types is not None and len(types) == 0:
            return
        if types is None:
            return

        if isinstance(types, dict):
            for key, value in types.items():
                if isinstance(key, (int, long)):
                    self._field_types[key] = value
                elif key in self.field_names:
                    self._field_types[self.field_names.index(key)] = value
                else:
                    raise ValueError('Invalid field name "{0}"'.format(key))
        else:
            if len(types) != self.num_fields:
                raise ValueError('length of types list does not match the number of fields')
            self._field_types = types
    
    field_types = property(get_field_types, set_field_types)

    def get_field_shapes(self):
        """
        Get field shapes for variable length fields. If a field does not have a
        defined shape, then the field values will be stored in a list object.
        """
        self._check_connection()
        if self._field_shapes is None:
            return self._field_shapes
        field_shapes = {}
        for i, shape in enumerate(self._field_shapes):
            if shape is not None:
                name = self.field_names[i]
                field_shapes[name] = shape
        return field_shapes

    def set_field_shapes(self, shapes):
        """
        Set field shapes. A value of None or zero for a field means that
        an infinite length value will be allowed for field, and value will be
        stored in Python list object if field is postgresql geometry types, or
        WKT string objects if field is postgis type.
        Inputs:
            shapes: dict, mapping of field name or number to field shape
        """
        cdef int has_srid
        if self._dataframe_result:
            raise RuntimeError('Field shape cannot be set when outputing to dataframe')
        self._check_connection()

        self._field_shapes = [None] * self.num_fields
        if shapes is not None and len(shapes) == 0:
            return
        if shapes is None:
            return

        # Convert list of shapes to dict with field number to shape mapping
        new_shapes = {}
        if not isinstance(shapes, dict):
            if len(shapes) != self.num_fields:
                raise ValueError('number of shapes must equal number of fields')
            shapes = dict(zip(range(len(shapes)), shapes))

        for key, value in shapes.items():
            if value is not None:
                if isinstance(key, (int, long)):
                    index = key
                elif key in self.field_names:
                    index = self.field_names.index(key)
                else:
                    raise ValueError('Invalid field name "{0}"'.format(key))

                if isinstance(value, (int, long)):
                    if value < 1:
                        raise ValueError('Field shape must be greater than 0')
                elif isinstance(value, (list, tuple)):
                    for x in value:
                        if x < 1:
                            raise ValueError('Field shape values must be greater than 0')
                else:
                    raise ValueError('Field shape must be int, list, or tuple')

                # field shape for geometric types that contain points will always
                # have a fixed point size, so user only has to specify number of
                # points, lines, etc
                field_type = get_field_type(self._adapter.result, index)
                if field_type in [PATHOID, POLYGONOID]:
                    if not isinstance(value, int):
                        raise ValueError('Field shape for path or polygon must be int')
                    self._field_shapes[index] = (value, 2)
                elif field_type == self._adapter.postgis_geometry_oid:
                    data = get_field_data(self._adapter.result, 0, index)
                    gis_field_type = get_gis_type(data, &has_srid)
                    gis_point_size = get_gis_point_size(gis_field_type)
                    if gis_point_size == -1:
                        raise ValueError('Unknown PostGIS point size')
                    if gis_field_type in [GIS_LINE2D, GIS_LINE3D, GIS_LINE4D]:
                        if not isinstance(value, int):
                            raise ValueError('Field shape for PostGIS line must be int')
                        self._field_shapes[index] = (value, gis_point_size)
                    elif gis_field_type in [GIS_POLYGON2D, GIS_POLYGON3D, GIS_POLYGON4D]:
                        if not isinstance(value, (list, tuple)) or len(value) != 2:
                            raise ValueError('Field shape for PostGIS polygon must have 2 dimensions')
                        self._field_shapes[index] = (value[0], value[1], gis_point_size)
                    elif gis_field_type in [GIS_MULTIPOINT2D, GIS_MULTIPOINT3D, GIS_MULTIPOINT4D]:
                        if not isinstance(value, int):
                            raise ValueError('Field shape for PostGIS multipoint must be int')
                        self._field_shapes[index] = (value, gis_point_size)
                    elif gis_field_type in [GIS_MULTILINE2D, GIS_MULTILINE3D, GIS_MULTILINE4D]:
                        if not isinstance(value, (list, tuple)) or len(value) != 2:
                            raise ValueError('Field shape for PostGIS multiline must have 2 dimensions')
                        self._field_shapes[index] = (value[0], value[1], gis_point_size)
                    elif gis_field_type in [GIS_MULTIPOLYGON2D, GIS_MULTIPOLYGON3D, GIS_MULTIPOLYGON4D]:
                        if not isinstance(value, (list, tuple)) or len(value) != 3:
                            raise ValueError('Field shape for GIS multipolygon must have 3 dimensions')
                        self._field_shapes[index] = (value[0], value[1], value[2], gis_point_size)
                    else:
                        raise ValueError('Setting shape for field "{0}" not allowed.'.format(index))
                else:
                    raise ValueError('Setting shape for field "{0}" not allowed.'.format(index))

    field_shapes = property(get_field_shapes, set_field_shapes)

    def _parse_slice(self, start, stop, step):
        """
        Adjust stop slice parameter if needed, get number of records to read.
        Stop is set to total number of records if None.
        Inputs:
            start: int, start record
            stop: int, stop record
            step: int, step value
        Outputs:
            Adjusted stop and number of records
        """
        if stop is None:
            stop = self.num_records
        if step > 0:
            num_records = stop - start
        else:
            num_records = start - stop
        num_records = math.ceil(abs(num_records / step))
        return stop, num_records

    def _to_array(self, start=0, stop=None, step=1):
        """
        Read records from table or custom query into NumPy record array,
        and return array. Custom query takes precedent over table.
        Inputs:
            start: first record to read
            stop: last record to read. If None, read all records.
            step: number of records to skip between each read
        """
        cdef numpy.ndarray carray
        cdef numpy.ndarray final_carray
        cdef char **data_arrays
        cdef int num_records_found
        cdef int has_srid
        self._check_connection()

        stop, num_records = self._parse_slice(start, stop, step)

        # Check to see if we need to cast database values to user specified dtype.
        # Default types are based on database table/query column types.
        default_types, dtype_len = self._get_default_types()
        if default_types == self.field_types:
            cast_types = False
        else:
            cast_types = True

        dtype = numpy.dtype([(str(name), type_, size) for name, type_, size
                             in zip(self.field_names, default_types, dtype_len)])
        if cast_types:
            cast_dtype = numpy.dtype([(str(name), type_, size) for name, type_, size
                                      in zip(self.field_names, self.field_types, dtype_len)])

        for i in range(self.num_fields):
            if self._field_shapes[i] is not None:
                for dim, size in enumerate(self._field_shapes[i]):
                    set_field_dim_size(self._adapter, i, dim, size)
            elif get_field_type(self._adapter.result, i) == self._adapter.postgis_geometry_oid:
                data = get_field_data(self._adapter.result, 0, i)
                gis_field_type = get_gis_type(data, &has_srid)
                if gis_field_type in [GIS_POINT2D, GIS_POINT3D, GIS_POINT4D] and default_types[i] == 'f8':
                    field_length = get_field_length(self._adapter.result, i)
                    set_field_dim_size(self._adapter, i, 0, field_length)
                else:
                    clear_field_size(self._adapter, i)
            else:
                clear_field_size(self._adapter, i)

        if ((step > 0 and start >= stop)
                or (step < 0 and start <= stop)):
            if cast_types:
                return numpy.array([], dtype=cast_dtype)
            else:
                return numpy.array([], dtype=dtype)

        # Allocate and prepare result array. An array of data pointers needs to
        # be passed to read_records. Since the output is going to be a record
        # array, pass a single data pointer to read_records so that data from
        # all source fields will be stored in array sequentially.
        # If we need to cast values from database to user specified dtype,
        # temporarily store values from db in a buffer array. As buffer array
        # is filled, call Cython function to copy values to final array set
        # to user specified dtype. During the copy, numpy will perform the
        # correct cast under the hood. This isn't super fast, but it's easy
        # and will ensure casting follows numpy casting
        data_arrays = <char**>calloc(1, sizeof(char*))
        if cast_types:
            carray = numpy.ndarray(CAST_BUFFER_SIZE, numpy.dtype(dtype))
            final_carray = numpy.ndarray(num_records, numpy.dtype(cast_dtype))
            data_arrays[0] = carray.data
            result = read_records(self._adapter, start, stop, step, data_arrays,
                &num_records_found, 1, <void*>carray, <void*>final_carray, 0)
            free(data_arrays)
            if result != ADAPTER_SUCCESS:
                raise exceptions[result]
            return numpy.asarray(final_carray)
        else:
            carray = numpy.ndarray(num_records, dtype)
            data_arrays[0] = carray.data
            result = read_records(self._adapter, start, stop, step, data_arrays,
                &num_records_found, 0, NULL, NULL, 0)
            free(data_arrays)
            if result != ADAPTER_SUCCESS:
                raise exceptions[result]
            return numpy.asarray(carray)

    def _to_dataframe(self, start=0, stop=None, step=1):
        """
        Read records from table or custom query into Pandas dataframe.
        and return dataframe. Custom query takes precedent over table.
        Inputs:
            start: first record to read
            stop: last record to read. If None, read all records.
            step: number of records to skip between each read
        """
        cdef char **data_arrays
        cdef numpy.ndarray temp_array
        cdef int num_records_found
        self._check_connection()

        stop, num_records = self._parse_slice(start, stop, step)

        # Check to see if we need to cast database values to user specified dtype.
        # Default types are based on database table/query column types.
        default_types, dtype_len = self._get_default_types()
        if default_types == self.field_types:
            cast_types = False
        else:
            cast_types = True

        if ((step > 0 and start >= stop)
                or (step < 0 and start <= stop)):
            data = OrderedDict()
            if cast_types:
                for i in range(self.num_fields):
                    dtype = self.field_types[i]
                    data[self.field_names[i]] = numpy.array([], dtype=dtype)
            else:
                for i in range(self.num_fields):
                    dtype = self.field_types[i]
                    data[self.field_names[i]] = numpy.array([], dtype=dtype)
            return pd.DataFrame(data=data)
        
        # Allocate and prepare result dataframe. An array of data pointers needs to
        # be passed to read_records, one for each dataframe field. Each source
        # field's data will be stored in corresponding data pointer's memory chunk.
        data_arrays = <char**>calloc(self.num_fields, sizeof(char*))
        data = OrderedDict()
        final_data = OrderedDict()
        for i in range(self.num_fields):
            dtype = default_types[i]
            data[self.field_names[i]] = numpy.array([0], dtype=default_types[i])
            final_data[self.field_names[i]] = numpy.array([0], dtype=self.field_types[i])

        # If we need to cast values from database to user specified dtype,
        # temporarily store values from db in a buffer dataframe. As buffer
        # dataframe is filled, call Cython function to copy values to final
        # dataframe set to user specified dtype. During the copy, pandas will
        # perform the correct cast under the hood. This isn't super fast,
        # but it's easy and will ensure casting follows pandas casting.
        if cast_types:
            df_result = pd.DataFrame(index=numpy.arange(CAST_BUFFER_SIZE, dtype='u8'), data=data)
            final_df_result = pd.DataFrame(index=numpy.arange(num_records, dtype='u8'), data=final_data)
        else:
            df_result = pd.DataFrame(index=numpy.arange(num_records, dtype='u8'), data=data)

        # Get data pointers from dataframe columns
        for i in range(self.num_fields):
            temp_array = df_result[self.field_names[i]].values
            data_arrays[i] = temp_array.data

        if cast_types:
            result = read_records(self._adapter, start, stop, step, data_arrays,
                &num_records_found, 1, <void*>df_result, <void*>final_df_result, 1)
            free(data_arrays)
            if result != ADAPTER_SUCCESS:
                raise exceptions[result]
            return final_df_result
        else:
            result = read_records(self._adapter, start, stop, step, data_arrays,
                &num_records_found, 0, NULL, NULL, 1)
            free(data_arrays)
            if result != ADAPTER_SUCCESS:
                raise exceptions[result]
            return df_result

    def __getitem__(self, index):
        """
        Read records by slice
        Inputs:
            index: slice object or int index
        """
        self._check_connection()

        if isinstance(index, (int, long)):
            if index > self.num_records - 1:
                raise ValueError('index {0} is out of bounds'.format(index))
            elif index < 0:
                if self.num_records + index < 0:
                    raise ValueError('index {0} is out of bounds'.format(index))
                index = self.num_records + index
            if self._dataframe_result:
                return self._to_dataframe(index, index + 1)
            else:
                return self._to_array(index, index + 1)
        elif isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step

            if step is None:
                step = 1

            if start is None:
                if step > 0:
                    start = 0
                else:
                    start = self.num_records - 1
            elif start > self.num_records:
                start = self.num_records
            elif start < 0:
                if self.num_records + start < 0:
                    start = 0 - self.num_records
                start = self.num_records + start

            if stop is None:
                if step > 0:
                    stop = self.num_records
                else:
                    stop = -1
            elif stop > self.num_records:
                stop = self.num_records
            elif stop < 0:
                if self.num_records + stop < 0:
                    stop = 0 - self.num_records
                stop = self.num_records + stop

            if self._dataframe_result:
                return self._to_dataframe(start, stop, step)
            else:
                return self._to_array(start, stop, step)

        else:
            raise IndexError('index must be slice or int')
