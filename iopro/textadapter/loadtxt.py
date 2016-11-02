import numpy
import operator
import iopro

from numpy.compat import (
    asstr, bytes, basestring, unicode
    )


def loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0):
    """
    Load data from a text file.

    Each row in the text file must have the same number of values.

    Parameters
<<<<<<< HEAD
    ***REMOVED***
=======
    ----------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    fname : file or str
        File, filename, or generator to read.  If the filename extension is
        ``.gz`` or ``.bz2``, the file is first decompressed. Note that
        generators should return byte strings for Python 3k.
    dtype : data-type, optional
        Data-type of the resulting array; default: float.  If this is a
        record data-type, the resulting array will be 1-dimensional, and
        each row will be interpreted as an element of the array.  In this
        case, the number of columns used must match the number of fields in
        the data-type.
    comments : str, optional
        The character used to indicate the start of a comment;
        default: '#'.
    delimiter : str, optional
        The string used to separate values.  By default, this is any
        whitespace.
    converters : dict, optional
        A dictionary mapping column number to a function that will convert
        that column to a float.  E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``.  Converters can also be used to
        provide a default value for missing data (but see also `genfromtxt`):
        ``converters = {3: lambda s: float(s.strip() or 0)}``.  Default: None.
    skiprows : int, optional
        Skip the first `skiprows` lines; default: 0.
    usecols : sequence, optional
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        The default, None, results in all columns being read.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``.  When used with a record
        data-type, arrays are returned for each field.  Default is False.
    ndmin : int, optional
        The returned array will have at least `ndmin` dimensions.
        Otherwise mono-dimensional axes will be squeezed.
        Legal values: 0 (default), 1 or 2.
        .. versionadded:: 1.6.0

    Returns
    -------
    out : ndarray
        Data read from the text file.

    See Also
    --------
    load, fromstring, fromregex
    genfromtxt : Load data with missing values handled as specified.
    scipy.io.loadmat : reads MATLAB data files

    Notes
    -----
    This function aims to be a fast reader for simply formatted files.  The
    `genfromtxt` function provides more sophisticated handling of, e.g.,
    lines with missing values.

    Examples
    --------
    >>> from StringIO import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\\n2 3")
    >>> np.loadtxt(c)
    array([[ 0.,  1.],
           [ 2.,  3.]])

    >>> d = StringIO("M 21 72\\nF 35 58")
    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    array([('M', 21, 72.0), ('F', 35, 58.0)],
          dtype=[('gender', '|S1'), ('age', '<i4'), ('weight', '<f4')])

    >>> c = StringIO("1,0,2\\n3,0,4")
    >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
    >>> x
    array([ 1.,  3.])
    >>> y
    array([ 2.,  4.])

    """

    user_converters = converters

    whitespace_delims = False
    if delimiter is None:
        whitespace_delims = True

    compression = None
    if isinstance(fname, basestring) and fname[-3:] == '.gz':
        compression = 'gzip'

    try:
        adapter = iopro.text_adapter(fname, parser='csv', delimiter=delimiter,
            comment=comments, header=skiprows, compression=compression, whitespace_delims=whitespace_delims,
            field_names=False, infer_types=False)
    except EOFError:
        array = numpy.array([], dtype=numpy.int64, ndmin=ndmin)
        if ndmin == 2:
            array = array.T
        return array

    if usecols is None:
        usecols = [x for x in range(0, adapter.field_count)]
    elif isinstance(usecols, numpy.ndarray):
        usecols = usecols.tolist()
    else:
        usecols = list(usecols)

    # create valid dtype object
    if isinstance(dtype, (list, tuple)):
        dtype = [dt if isinstance(dt, tuple) else ('', dt) for dt in dtype]
    dtype = numpy.dtype(dtype)
    
    # create list of dtypes to send to TextAdapter
    if dtype.names is None:
        # create list of homogenous scalar dtypes from single scalar dtype
        numFields = len(usecols)
        dtypes = [dtype]*numFields
        fieldNames = None
    else:
        # create list of scalar dtypes from struct dtype
        dtypes, fieldNames = unpack_dtype(dtype)

    if fieldNames is not None:
        list_names = ['' for x in range(adapter.field_count)]
        for i, col in enumerate(usecols):
            list_names[col] = fieldNames[i]
        adapter.field_names = list_names

    adapter.set_field_types(types=dict(zip(usecols, dtypes)))

    if converters is not None:
        for field, converter in converters.items():
            adapter.set_converter(field, converter)

    array = adapter[usecols][:]
      
    if dtype.fields is not None and numpy.object_ not in [dt[0] for dt in array.dtype.fields.values()]:
        array.dtype = dtype
    elif dtype.fields is None:
        array.dtype = dtype
    if dtype.names is None:
        if adapter.field_count == 0:
            array.shape = (adapter.size,)
        else:
            array.shape = (adapter.size, len(usecols))
 
    # Multicolumn data are returned with shape (1, N, M), i.e.
    # (1, 1, M) for a single row - remove the singleton dimension there
    if array.ndim == 3 and array.shape[:2] == (1, 1):
        array.shape = (1, -1)

    # Verify that the array has at least dimensions `ndmin`.
    # Check correctness of the values of `ndmin`
    if not ndmin in [0, 1, 2]:
        raise ValueError('Illegal value of ndmin keyword: %s' % ndmin)

    # Tweak the size and shape of the arrays - remove extraneous dimensions
    if array.ndim > ndmin:
        array = numpy.squeeze(array)

    # and ensure we have the minimum number of dimensions asked for
    # - has to be in this order for the odd case ndmin=1, array.squeeze().ndim=0
    if array.ndim < ndmin:
        if ndmin == 1:
            array = numpy.atleast_1d(array)
        elif ndmin == 2:
            array = numpy.atleast_2d(array).T

    if unpack:
        if len(dtype) > 1:
            # For structured arrays, return an array for each field.
            return [array[field] for field in dtype.names]
        else:
            return array.T
    else:
        return array


def unpack_dtype(dtype):
    dtypes = []
    names = []
    for name in dtype.names:
        if dtype.fields[name][0].names is None:
            count = 1
            shape = dtype.fields[name][0].shape
            if len(shape) > 0:
                count = 1
                for s in shape:
                    count = count * s
            if count == 0 or count == 1:
                dtypes.append(dtype.fields[name][0].base)
                names.append(name)
            else:
                for x in range(count):
                    dtypes.append(dtype.fields[name][0].base)
                    names.append('')
        else:
            nested_dtypes, nested_names = unpack_dtype(dtype.fields[name][0])
            for dt in nested_dtypes:
                dtypes.append(dt)
            for n in nested_names:
                names.append(n)
    return dtypes, names


