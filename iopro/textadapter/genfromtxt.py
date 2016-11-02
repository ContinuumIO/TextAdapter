import numpy as np
from numpy.compat import asbytes, asbytes_nested
from numpy.lib._iotools import LineSplitter, NameValidator, easy_dtype, _is_string_like
import warnings
import operator
import iopro
import sys

from numpy.compat import (
    asbytes, asstr, asbytes_nested, bytes, basestring, unicode
    )

if sys.version > '3':
    int_types = int
else:
    int_types = (int, long)

def genfromtxt(fname, dtype=float, comments='#', delimiter=None,
               skiprows=0, skip_header=0, skip_footer=0, converters=None,
               missing='', missing_values=None, filling_values=None,
               usecols=None, names=None,
               excludelist=None, deletechars=None, replace_space='_',
               autostrip=False, case_sensitive=True, defaultfmt="f%i",
               unpack=None, usemask=False, loose=True, invalid_raise=True):
    """
    Load data from a text file, with missing values handled as specified.

    Each line past the first `skip_header` lines is split at the `delimiter`
    character, and characters following the `comments` character are discarded.

    Parameters
<<<<<<< HEAD
    ***REMOVED***
=======
    ----------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    fname : file or str
        File, filename, or generator to read.  If the filename extension is
        `.gz` or `.bz2`, the file is first decompressed. Note that
        generators must return byte strings in Python 3k.
    dtype : dtype, optional
        Data type of the resulting array.
        If None, the dtypes will be determined by the contents of each
        column, individually.
    comments : str, optional
        The character used to indicate the start of a comment.
        All the characters occurring on a line after a comment are discarded
    delimiter : str, int, or sequence, optional
        The string used to separate values.  By default, any consecutive
        whitespaces act as delimiter.  An integer or sequence of integers
        can also be provided as width(s) of each field.
    skip_header : int, optional
        The numbers of lines to skip at the beginning of the file.
    skip_footer : int, optional
        The numbers of lines to skip at the end of the file
    converters : variable, optional
        The set of functions that convert the data of a column to a value.
        The converters can also be used to provide a default value
        for missing data: ``converters = {3: lambda s: float(s or 0)}``.
    missing_values : variable, optional
<<<<<<< HEAD
        The set of strings corresponding to ***REMOVED***
=======
        The set of strings corresponding to missing data.
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    filling_values : variable, optional
        The set of values to be used as default when the data are missing.
    usecols : sequence, optional
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1, 4, 5)`` will extract the 2nd, 5th and 6th columns.
    names : {None, True, str, sequence}, optional
        If `names` is True, the field names are read from the first valid line
        after the first `skip_header` lines.
        If `names` is a sequence or a single-string of comma-separated names,
        the names will be used to define the field names in a structured dtype.
        If `names` is None, the names of the dtype fields will be used, if any.
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default list
        ['return','file','print']. Excluded names are appended an underscore:
        for example, `file` would become `file_`.
    deletechars : str, optional
        A string combining invalid characters that must be deleted from the
        names.
    defaultfmt : str, optional
        A format used to define default field names, such as "f%i" or "f_%02i".
    autostrip : bool, optional
        Whether to automatically strip white spaces from the variables.
    replace_space : char, optional
        Character(s) used in replacement of white spaces in the variables
        names. By default, use a '_'.
    case_sensitive : {True, False, 'upper', 'lower'}, optional
        If True, field names are case sensitive.
        If False or 'upper', field names are converted to upper case.
        If 'lower', field names are converted to lower case.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``
    usemask : bool, optional
        If True, return a masked array.
        If False, return a regular array.
    invalid_raise : bool, optional
        If True, an exception is raised if an inconsistency is detected in the
        number of columns.
        If False, a warning is emitted and the offending lines are skipped.

    Returns
    -------
    out : ndarray
        Data read from the text file. If `usemask` is True, this is a
        masked array.

    See Also
    --------
    numpy.loadtxt : equivalent function when no data is missing.

    Notes
    -----
    * When spaces are used as delimiters, or when no delimiter has been given
      as input, there should not be any missing data between two fields.
    * When the variables are named (either by a flexible dtype or with `names`,
      there must not be any header in the file (else a ValueError
      exception is raised).
    * Individual values are not stripped of spaces by default.
      When using a custom converter, make sure the function does remove spaces.

    References
<<<<<<< HEAD
    ***REMOVED***
=======
    ----------
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    .. [1] Numpy User Guide, section `I/O with Numpy
           <http://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html>`_.

    Examples
    ---------
    >>> from StringIO import StringIO
    >>> import numpy as np

    Comma delimited file with mixed dtype

    >>> s = StringIO("1,1.3,abcde")
    >>> data = np.genfromtxt(s, dtype=[('myint','i8'),('myfloat','f8'),
    ... ('mystring','S5')], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '|S5')])

    Using dtype = None

    >>> s.seek(0) # needed for StringIO example only
    >>> data = np.genfromtxt(s, dtype=None,
    ... names = ['myint','myfloat','mystring'], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '|S5')])

    Specifying dtype and names

    >>> s.seek(0)
    >>> data = np.genfromtxt(s, dtype="i8,f8,S5",
    ... names=['myint','myfloat','mystring'], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '|S5')])

    An example with fixed-width columns

    >>> s = StringIO("11.3abcde")
    >>> data = np.genfromtxt(s, dtype=None, names=['intvar','fltvar','strvar'],
    ...     delimiter=[1,3,5])
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('intvar', '<i8'), ('fltvar', '<f8'), ('strvar', '|S5')])

    """
    # Py3 data conversions to bytes, for convenience
    #comments = asbytes(comments)
    #if isinstance(delimiter, unicode):
    #    delimiter = asbytes(delimiter)
    #if isinstance(missing, unicode):
    #    missing = asbytes(missing)
    #if isinstance(missing_values, (unicode, list, tuple)):
    #    missing_values = asbytes_nested(missing_values)

    if usemask:
        from numpy.ma import MaskedArray, make_mask_descr

    # Check the input dictionary of converters
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        errmsg = "The input argument 'converter' should be a valid dictionary "\
            "(got '%s' instead)"
        raise TypeError(errmsg % type(user_converters))

    # Initialize the filehandle, the LineSplitter and the NameValidator
    own_fhd = False
    try:
        if isinstance(fname, basestring):
            fhd = iter(np.lib._datasource.open(fname, 'rbU'))
            own_fhd = True
        else:
            fhd = iter(fname)
    except TypeError:
        raise TypeError("fname mustbe a string, filehandle, or generator. "\
                        "(got %s instead)" % type(fname))

    validate_names = NameValidator(excludelist=excludelist,
                                   deletechars=deletechars,
                                   case_sensitive=case_sensitive,
                                   replace_space=replace_space)

    # Get the first valid lines after the first skiprows ones ..
    if skiprows:
        warnings.warn(\
            "The use of `skiprows` is deprecated, it will be removed in numpy 2.0.\n" \
            "Please use `skip_header` instead.",
            DeprecationWarning)
        skip_header = skiprows

    set_names = False
    if names is True:
        set_names = True

    infer_types = False
    if dtype is None:
        infer_types = True

    whitespace_delims = False
    if delimiter is None:
        delimiter = ' '
        whitespace_delims = True

    compression = None
    if isinstance(fname, basestring) and fname[-3:] == '.gz':
        compression = 'gzip'

    try:
        if isinstance(delimiter, basestring):
            adapter = iopro.text_adapter(fname, parser='csv', delimiter=delimiter,
                comment=comments, header=skip_header, footer=skip_footer,
                compression=compression, field_names=set_names, infer_types=True,
                whitespace_delims=whitespace_delims)
        elif isinstance(delimiter, int) or isinstance(delimiter, (list, tuple)):
            adapter = iopro.text_adapter(fname, parser='fixed_width',
            field_widths=delimiter, comment=comments, header=skip_header,
            footer=skip_footer, field_names=set_names, infer_types=True)
    except EOFError:
        return np.array([])

    field_names = None
    if isinstance(names, basestring):
        field_names = [name.strip() for name in names.split(',')]
    elif isinstance(names, tuple):
        field_names = list(names)
    elif isinstance(names, list):
        field_names = names
    elif set_names is True:
        field_names = adapter.field_names

    if usecols is None:
        usecols = [x for x in range(0, adapter.field_count)]
    elif isinstance(usecols, basestring) and field_names is None:
        raise ValueError('usecols contains unknown field names')
    elif isinstance(usecols, basestring):
        if field_names is None:
            raise ValueError('usecols contains unknown field names')
        else:
            usecols = [field_names.index(name.strip()) for name in usecols.split(',')]
    elif isinstance(usecols, (list, tuple)):
        if len(usecols) == 0:
            raise ValueError('usecols must contain at least one col')
        tempCols = []
        for col in usecols:
            if isinstance(col, basestring) and field_names is None:
                raise ValueError('usecols contains unknown field names')
            elif isinstance(col, basestring):
                tempCols.append(field_names.index(col))
            elif isinstance(col, int):
                tempCols.append(col)
            else:
                raise ValueError('usecols must contain either field numbers or field names')
        usecols = tempCols
    elif isinstance(usecols, int_types):
        usecols = [usecols]
    elif isinstance(usecols, numpy.ndarray):
        usecols = usecols.tolist()
    else:
        usecols = list(usecols)

    # adjust negative indices
    for i, col in enumerate(usecols):
        if col < 0:
            usecols[i] = adapter.field_count + col

    if isinstance(field_names, (list, dict)) and len(field_names) < len(usecols):
        field_names.extend([''] * (len(usecols) - len(field_names)))

    # Process the deprecated `missing`
    if missing != asbytes(''):
        warnings.warn(\
            "The use of `missing` is deprecated, it will be removed in Numpy 2.0.\n" \
            "Please use `missing_values` instead.",
            DeprecationWarning)
        missing_values = missing

    # Initialize the output lists ...
    # ... rows
    rows = []
    append_to_rows = rows.append
    # ... masks
    if usemask:
        masks = []
        append_to_masks = masks.append
    # ... invalid
    invalid = []
    append_to_invalid = invalid.append

    # create valid dtype object
    if isinstance(dtype, (list, tuple)):
        dtype = [dt if isinstance(dt, tuple) else ('', dt) for dt in dtype]
    dtype = np.dtype(dtype)

    # create list of dtypes to send to TextAdapter
    if dtype.names is None:
        # create list of homogenous scalar dtypes from single scalar dtype
        numFields = len(usecols)
        dtypes = [dtype]*numFields
    else:
        # create list of scalar dtypes from struct dtype
        dtypes, dtype_field_names = unpack_dtype(dtype)
        if field_names is None:
            field_names = dtype_field_names
        else:
            s = set(field_names)
            # if all entries in field_names are empty, use dtype field names
            if len(s) == 1 and s.pop() == '':
                field_names = dtype_field_names

    # use field names from dtype if field names were not specified by user
    # and not read from first line in file
    #if names is None and dtype.names is not None:
    #    field_names = dtype.names

    if field_names is not None:
        adapter.field_names = field_names

    if infer_types is False:
        adapter.set_field_types(types=dict(zip(usecols, dtypes)))

    if converters is not None:
        for field, converter in converters.items():
            adapter.set_converter(field, converter)

    if isinstance(missing_values, basestring):
        values = missing_values.split(',')
        missing_values = {}
        for i in range(adapter.field_count):
            missing_values[i] = values
    if missing_values is not None:
        adapter.set_missing_values(missing_values)

    if isinstance(filling_values, basestring):
        filling_values = filling_values.split(',')
    if filling_values is not None:
        filling_values = dict([(key,value) for key, value in filling_values.items() if key in usecols])
        adapter.set_fill_values(filling_values, loose)

    try:
        array = adapter[usecols][:]
    except DataTypeError:
        raise ValueError

    if own_fhd:
        fhd.close()

    # Adapter returns an array with struct dtype.
    # If no field names were specified or read from file,
    # and specified dtype is scalar, reset array to scalar dtype.
    if dtype.fields is None and field_names is None and set_names is False:
        # Can't set final dtype to scalar if struct dtype includes objects
        if array.dtype.fields is not None \
                and np.object_ not in [x[0] for x in array.dtype.fields.values()]:
            array.dtype = dtype
            # If no fields were read, we want to keep this a 1-d empty array.
            # Otherwise, set the proper shape.
            if adapter.field_count > 0:
                array.shape = (adapter.size, len(usecols))

    #elif dtype.names is None and isinstance(names, list):
    #    array.dtype = zip([names[i] for i in usecols], [dtype]*len(usecols))

    # Construct the final array
    #if usemask:
    #    array = array.view(MaskedArray)
    #    array._mask = outputmask

    if unpack:
        return array.squeeze().T
    return array.squeeze()


def unpack_dtype(dtype):
    dtypes = []
    names = []
    for name in dtype.names:
        if dtype.fields[name][0].names is None:
            count = 1
            shape = dtype.fields[name][0].shape
            if len(shape) > 0:
                count = reduce(operator.mul, shape)
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
