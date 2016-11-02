***REMOVED***---
TextAdapter.genfromtxt
***REMOVED***---

Load data from a text file, with missing values handled as specified.

Each line past the first `skip_header` lines is split at the `delimiter`
character, and characters following the `comments` character are discarded.

Parameters
***REMOVED***
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
    The set of strings corresponding to ***REMOVED***
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
TextAdapter.loadtxt : equivalent function when no data is missing.

Notes
-----
* When spaces are used as delimiters, or when no delimiter has been given
  as input, there should not be any missing data between two fields.
* When the variables are named (either by a flexible dtype or with `names`,
  there must not be any header in the file (else a ValueError
  exception is raised).
* Individual values are not stripped of spaces by default.
  When using a custom converter, make sure the function does remove spaces.

Examples
---------
***REMOVED***
    >>> from io import StringIO

Comma delimited file with mixed dtype

    >>> s = StringIO("1,1.3,abcde")
    >>> data = TextAdapter.genfromtxt(s, dtype=[('myint','i8'),('myfloat','f8'),
    ... ('mystring','S5')], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '|S5')])

Using dtype = None

    >>> s.seek(0) # needed for StringIO example only
    >>> data = TextAdapter.genfromtxt(s, dtype=None,
    ... names = ['myint','myfloat','mystring'], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '|S5')])

Specifying dtype and names

    >>> s.seek(0)
    >>> data = TextAdapter.genfromtxt(s, dtype="i8,f8,S5",
    ... names=['myint','myfloat','mystring'], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '|S5')])

An example with fixed-width columns

    >>> s = StringIO("11.3abcde")
    >>> data = TextAdapter.genfromtxt(s, dtype=None, names=['intvar','fltvar','strvar'],
    ...     delimiter=[1,3,5])
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('intvar', '<i8'), ('fltvar', '<f8'), ('strvar', '|S5')])


