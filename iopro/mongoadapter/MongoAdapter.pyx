import numpy
cimport numpy
import ctypes
from cpython.ref cimport Py_INCREF
from cpython cimport PyTypeObject

cdef extern from "bytesobject.h":
    object PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)
    object PyUnicode_FromFormat(const char *format, ...)

cdef extern from "numpy/arrayobject.h":
    cdef enum requirements:
        NPY_WRITEABLE
    
    object PyArray_NewFromDescr(PyTypeObject *subtype, numpy.dtype descr,
                                int nd, numpy.npy_intp* dims, numpy.npy_intp* strides,
                                void* data, int flags, object obj)
    struct PyArray_Descr:
        int type_num, elsize
        char type

    cdef enum:
        NPY_MAXDIMS


# enable C api
numpy.import_array()


dtype_converter_mapping = {}
dtype_converter_mapping[numpy.bool] = <long>&mongo2uint_converter
dtype_converter_mapping[numpy.uint8] = <long>&mongo2uint_converter
dtype_converter_mapping[numpy.uint16] = <long>&mongo2uint_converter
dtype_converter_mapping[numpy.uint32] = <long>&mongo2uint_converter
dtype_converter_mapping[numpy.uint64] = <long>&mongo2uint_converter
dtype_converter_mapping[numpy.int8] = <long>&mongo2int_converter
dtype_converter_mapping[numpy.int16] = <long>&mongo2int_converter
dtype_converter_mapping[numpy.int32] = <long>&mongo2int_converter
dtype_converter_mapping[numpy.int64] = <long>&mongo2int_converter
dtype_converter_mapping[numpy.float32] = <long>&mongo2float_converter
dtype_converter_mapping[numpy.float64] = <long>&mongo2float_converter


cdef class ArrayDealloc:
   cdef void* data

   def __cinit__(self):
       self.data = NULL

   def __dealloc__(self):
       if self.data:
           free(self.data)

cdef class MongoAdapter:
    """
    Mongo adapter for reading data from Mongo database into
    NumPy array or Pandas dataframe.

    Constructor Inputs:
        host: Mongo db host name
        port: Mongo db port 
        database: Mongo db database name
        collection: name of collection within database
    """

    cdef mongo_adapter_t *adapter
    cdef object mapping
    cdef object pythonConverters

    cdef object field_names

    def __cinit__(self, host, port, database, collection):
        """
        Mongo adapter constructor.
        See above for constructor argument descriptions.
        """

        # open_mongo_adapter requires bytes, so encode if needed
        b_host = host if isinstance(host, bytes) else host.encode()
        b_db = database if isinstance(database, bytes) else database.encode()
        b_clxn = collection if isinstance(collection, bytes) else collection.encode()

        self.adapter = open_mongo_adapter(b_host, port, b_db, b_clxn)
        self.adapter.default_converters[<unsigned int>STRING_OBJECT_CONVERTER_FUNC] = <converter_func_ptr>&mongo2str_object_converter;

        self.field_names = None

    def __dealloc__(self):
        close_mongo_adapter(self.adapter)


    @property
    def size(self):
        """
        Return number of records in mongo db collection.
        """
        return get_num_records(self.adapter)


    '''def set_missing_values(self, missing_values):
        pass

    def set_fill_values(self, fill_values, loose=False):
        pass

    def to_array(self):
        pass'''


    def __get_dtype(self):
        """ Set field/dtype dict from converter functions set by type inference engine """
        cdef converter_func_ptr converter
        field_dtypes = []

        for field in range(self.adapter.fields.num_fields):
            raw_name = self.adapter.fields.field_info[field].name
            field_name = raw_name if isinstance(raw_name, basestring) else raw_name.decode()
            converter = self.adapter.fields.field_info[field].converter
            if converter == &mongo2uint_converter:
                dtype_string = 'u'+str(self.adapter.fields.field_info[field].output_field_size)
                field_dtypes.append((field_name, dtype_string))
            elif converter == &mongo2int_converter:
                dtype_string = 'i'+str(self.adapter.fields.field_info[field].output_field_size)
                field_dtypes.append((field_name, dtype_string))
            elif converter == &mongo2float_converter:
                dtype_string = 'f'+str(self.adapter.fields.field_info[field].output_field_size)
                field_dtypes.append((field_name, dtype_string))
            elif converter == &mongo2str_object_converter:
                dtype_string = 'O'
                field_dtypes.append((field_name, dtype_string))

        return numpy.dtype(field_dtypes)



    def __read_slice(self, start_rec, stop_rec, step_rec):

        cdef uint32_t recs_read = 0
        cdef char *data = NULL

        # calculate record size in bytes
        rec_size = 0
        for i in range(self.adapter.fields.num_fields):
            rec_size = rec_size + get_field_size(self.adapter.fields, NULL, i)

        num_recs = stop_rec - start_rec
        if num_recs > self.size:
            num_recs = self.size

        data = <char*>calloc(num_recs / step_rec, rec_size)
        result = read_records(self.adapter, start_rec, num_recs, step_rec, <char*>data, &recs_read)
        if result != MONGO_ADAPTER_SUCCESS:
            if result == MONGO_ADAPTER_ERROR_TYPE_CHANGED:
                free(data)
                rec_size = 0
                for i in range(self.adapter.fields.num_fields):
                    rec_size = rec_size + get_field_size(self.adapter.fields, NULL, i)
                data = <char*>calloc(num_recs / step_rec, rec_size)
                result = read_records(self.adapter, start_rec, num_recs, step_rec, <char*>data, &recs_read)
                if result != MONGO_ADAPTER_SUCCESS:
                    free(data)
                    raise IOError('read error - result was ' + str(result))
            else:
                free(data)
                raise IOError('read error - result was ' + str(result))
        array = self.__create_array(data, self.__get_dtype(), recs_read)

        return array


    cdef __create_array(self, char *data, object dtype, uint32_t num_recs):
        """ Create numpy array out of pre allocated data filled in by call
            to read_records() """
        cdef numpy.npy_intp dims[1]
        cdef ArrayDealloc array_dealloc
        cdef numpy.ndarray carray

        dims[0] = num_recs
        carray = PyArray_NewFromDescr(<PyTypeObject*>numpy.ndarray, dtype, 1, dims, NULL, data, NPY_WRITEABLE, <object>NULL)
        Py_INCREF(dtype)

        # Use ArrayDealloc object to make sure array data is properly deallocted when array is destroyed
        array_dealloc = ArrayDealloc.__new__(ArrayDealloc)
        array_dealloc.data = data
        Py_INCREF(array_dealloc)

        carray.base = <PyObject*>array_dealloc
        return numpy.asarray(carray)

    def __getitem__(self, index):
        """ Read records by record number or slice """

        start = 0
        stop = 0
        step = 1

        if isinstance(index, basestring):
            set_num_fields(self.adapter.fields, 1)
            set_converter(self.adapter.fields, 0, index.encode(), 8, &mongo2uint_converter, NULL)
            init_infer_types(self.adapter.fields)
            return MongoAdapterFields(self, index)

        elif isinstance(index, (tuple, list)):
            if isinstance(index[0], basestring):
                set_num_fields(self.adapter.fields, <uint32_t>len(index))
                for i, field_name in enumerate(index):
                    set_converter(self.adapter.fields, i, field_name.encode(), 8, &mongo2uint_converter, NULL)
                init_infer_types(self.adapter.fields)
                return MongoAdapterFields(self, index)
            else:
                raise TypeError('Invalid index type')

        elif isinstance(index, int):
            if self.adapter.fields.num_fields == 0:
                raise ValueError('No fields selected. Use set_field_names(fields) or [fields]')
            start = index
            if start < 0:
                start = self.size + start
            stop = start + 1
            step = 1

        elif isinstance(index, slice):
            if self.adapter.fields.num_fields == 0:
                raise ValueError('No fields selected. Use set_field_names(fields) or [fields]')
            start = index.start
            stop = index.stop
            step = index.step
            
            if start is None:
                start = 0
            else:
                if start < 0:
                    start = self.size + start

            if stop is None:
                stop = self.size
            else:
                # Check for greater than max uint32
                if stop > 0xffffffff:
                    stop = 0xffffffff
                if stop < 0:
                    stop = self.size + stop
            if step is None:
                step = 1
            else:
                if step < 0:
                    raise NotImplementedError('reverse stepping not implemented yet')

        else:
            raise TypeError('Invalid index type')
        return self.__read_slice(<uint32_t>start, <uint32_t>stop, <uint32_t>step)


    def set_field_names(self, index):
        """ Select field(s) by name. Argument might be a tuple, list or naked string """
        if isinstance(index, basestring):
            set_num_fields(self.adapter.fields, 1)
            set_converter(self.adapter.fields, 0, index.encode(), 8, &mongo2uint_converter, NULL)
            init_infer_types(self.adapter.fields)

        elif isinstance(index, (tuple, list)):
            if isinstance(index[0], basestring):
                set_num_fields(self.adapter.fields, <uint32_t>len(index))
                for i, field_name in enumerate(index):
                    set_converter(self.adapter.fields, i, field_name.encode(), 8, &mongo2uint_converter, NULL)
                init_infer_types(self.adapter.fields)
            else:
                raise TypeError('Invalid index type')


    def get_field_names(self):
        """ Returns: List of field names 
        if fields have been set by __getitem__ or set_field_names, else None """
        if self.adapter.fields.num_fields == 0:
            return None

        self.field_names = []
        for i in range(self.adapter.fields.num_fields):
            fieldname = self.adapter.fields.field_info[i].name
            fieldname = fieldname if isinstance(fieldname, basestring) else fieldname.decode() # Python3 support
            if isinstance(fieldname, basestring):
                self.field_names.append(fieldname)
            else:
                self.field_names.append('f'+str(i))
                print('WARNING: No name associated with field_info '+ str(i))

        return self.field_names

    def set_field_types(self, types=None):
        """ Set types of selected fields by specifying a dictionary of types. 

        Each key in types dict must be either an int index or a str field name.
        Each value is an array-protocol type string, eg 'i8', 'u4', 'O'.
        Note that the type 'O' represents a string object.
        """
        self.get_field_names()

        if isinstance(types, dict):
            for field, dtype in types.items():
                if isinstance(field, basestring):
                    field_idx = self.field_names.index(field)
                    field_name = field
                elif isinstance(field, int):
                    field_idx = field
                    field_name = self.field_names[field_idx]
                else:
                    raise TypeError('Each key in types dict must be either an int index or a str field name')
                numpy_dtype = numpy.dtype(dtype)
                dtype_kind = numpy_dtype.kind
                if numpy.version.version[0:3] == '1.6' and \
                        (dtype_kind == 'M' or dtype_kind == 'm'):
                    raise TypeError('NumPy 1.6 datetime/timedelta not supported')
                
                # Set field's infer_type flag to false
                self.adapter.fields.field_info[field_idx].infer_type = 0
                # Set field's converter to the appropriate function
                itemsize =  numpy_dtype.itemsize
                if dtype_kind in ('b', 'u'):
                    set_converter(self.adapter.fields, field_idx, field_name.encode(), itemsize, 
                        self.adapter.default_converters[<unsigned int>UINT_CONVERTER_FUNC], NULL)
                elif dtype_kind == 'i':
                    set_converter(self.adapter.fields, field_idx, field_name.encode(), itemsize, 
                        self.adapter.default_converters[<unsigned int>INT_CONVERTER_FUNC], NULL)
                elif dtype_kind == 'f':
                    set_converter(self.adapter.fields, field_idx, field_name.encode(), itemsize, 
                        self.adapter.default_converters[<unsigned int>FLOAT_CONVERTER_FUNC], NULL)
                else:
                    # CSC: Unsure about output_len for str objects. Chose the same value as done in try_converter()
                    set_converter(self.adapter.fields, field_idx, field_name.encode(), 8, 
                        self.adapter.default_converters[<unsigned int>STRING_OBJECT_CONVERTER_FUNC], NULL)

        else:
            raise TypeError('types must be dict of fields/dtypes')

    def get_field_types(self):
        """ Returns: numpy.dtype object of field types """
        if self.adapter.fields.num_fields == 0:
            raise ValueError('Adapter has no fields specified. Use set_field_names(fields) or [fields]')
        return self.__get_dtype()


    def get_fixed_fields(self):
        """ Returns: List of fields that have been user-selected """
        if self.adapter.fields.num_fields == 0:
            raise ValueError('Adapter has no fields specified. Use set_field_names(fields) or [fields]')
        fixed_fields = []
        for i in range(self.adapter.fields.num_fields):
            field = self.adapter.fields.field_info[i]
            fieldname = self.adapter.fields.field_info[i].name
            if isinstance(fieldname, basestring):
                if field.infer_type == 0:
                    fixed_fields.append(fieldname)
        return fixed_fields

    def is_field_type_set(self, field):
        """ Returns: True if user has selected given field (str name of int index), else False """
        if self.adapter.fields.num_fields == 0:
            return False
        if self.field_names == None:
            return False

        if isinstance(field, basestring):
            field_idx = self.field_names.index(field)
        elif isinstance(field, int):
            field_idx = field
        else:
            raise TypeError('field arg must be either an int index or a str field name')
        
        return self.adapter.fields.field_info[field_idx].infer_type == 0

class MongoAdapterFields:
    
    def __init__(self, adapter, fields):
        self.adapter_object = adapter
        self.fields = fields

    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, slice):
            return self.adapter_object[index]
        elif isinstance(index, (tuple, list)):
            if isinstance(index[0], int):
                return self.adapter_object[index]
            else:
                raise TypeError('Invalid index type.')
        else:
            raise TypeError('Invalid index type')

cdef ConvertError mongo2str_object_converter(void *input, uint32_t input_len, int input_type, void *output, uint32_t output_len, void *arg):
    """
    Wrapper function for calling string object converter function
    from low level C api. This is used to convert c strings to python
    string objects.

    Arguments:
    void *input - pointer to value to convert
    uint32_t input_len - length in bytes of input value
    void *output - pointer to memory chunk to store converted value
    uint32_t output_len - size of output memory chunk
    void *arg - pointer to python callable object which does the actual converting

    Returns:
    converted value as a python string object
    """
    cdef ConvertError result = CONVERT_ERROR_OBJECT_CONVERTER
    cdef PyObject **object_ptr
    object_ptr = <PyObject**>output
    cdef int ret
    cdef object temp = None
    cdef int32_t *int_input
    cdef double *double_input

    # Convert c type to Python string object and store in output array
    try:
        if input_type == 2:
            temp = PyUnicode_FromStringAndSize(<char*>input, input_len)
        elif input_type == 16:
            int_input = <int32_t*>input
            #temp = PyUnicode_FromFormat("%d", int_input[0])
            temp = str(int_input[0])
        elif input_type == 1:
            double_input = <double*>input
            #temp = PyUnicode_FromFormat("%f", double_input[0])
            temp = str(double_input[0])
        else:
            result = CONVERT_ERROR_INPUT_TYPE

        if object_ptr != NULL and temp is not None:
            object_ptr[0] = <PyObject*>temp
            Py_INCREF(<object>object_ptr[0])
            result = CONVERT_SUCCESS
        elif temp is not None:
            result = CONVERT_SUCCESS

    except Exception as e:
        result = CONVERT_ERROR_OBJECT_CONVERTER
 
    return result
