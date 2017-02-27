<<<<<<< HEAD
<<<<<<< HEAD
***REMOVED***--
TextAdapter.loadtxt
***REMOVED***--
=======
-------------
TextAdapter.loadtxt
-------------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
-------------
TextAdapter.loadtxt
-------------
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c

Load data from a text file.

Each row in the text file must have the same number of values.

Parameters
<<<<<<< HEAD
<<<<<<< HEAD
***REMOVED***
=======
----------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
----------
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c
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
    provide a default value for missing data (but see also `TextAdapter.genfromtxt`):
    ``converters = {3: lambda s: float(s.strip() or 0)}``.  Default: None.
skiprows : int, optional
    Skip the first `skiprows` lines; default: 0.
usecols : sequence, optional
    Which columns to read, with 0 being the first.  For example,
    ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
    The default, None, results in all columns being read.
unpack : bool, optional
    If True, the returned array is transposed, so that arguments may be
    unpacked using ``x, y, z = TextAdapter.loadtxt(...)``.  When used with a record
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
TextAdapter.genfromtxt : Load data with missing values handled as specified.

Examples
--------

simple parse of StringIO object data
    >>> import TextAdapter
    >>> from io import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\\n2 3")
    >>> TextAdapter.loadtxt(c)
    >>> array([[ 0.,  1.],
           [ 2.,  3.]])

set dtype of output array
    >>> d = StringIO("M 21 72\\nF 35 58")
    >>> TextAdapter.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    >>> array([('M', 21, 72.0), ('F', 35, 58.0)],
          dtype=[('gender', '|S1'), ('age', '<i4'), ('weight', '<f4')])

set delimiter and columns to parse
    >>> c = StringIO("1,0,2\\n3,0,4")
    >>> x, y = TextAdapter.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
    >>> x
    >>> array([ 1.,  3.])
    >>> y
    >>> array([ 2.,  4.])


