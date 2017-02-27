Notes on the Development and Design of the TextAdapter Module
=============================================================

The TextAdapter module was the first and most complicated IOPro data
adapter.  The rest of the data adapters loosely follow the design of the
TextAdapter module described below.

Key Ideas
---------

The TextAdapter module supports parsing tab delimited text, text with fixed
width fields, JSON text, and text whose fields can be desribed with regular
expressions.

The guts of the TextAdapter module are written in C, with a Python interface
implemented in Cython.

The IOPro interface for the data adapters (TextAdapter, MongoAdapter,
PostgresAdapter, and AccumuloAdapter) are designed to be numpy array-like in
that slicing on the adapter is used to retrieve subsets of data.  When the
adapter object is first created, no data is actually read (except for a few
records at the beginning of the input data to determine field types, number
of fields, etc).

IOPro is generally optimized for memory usage over speed, although speed is
definitely a primary goal too.  Data copying is kept to a minimum so that as
much data as possible can be read into a numpy array.

A TextAdapter object contains an array of function pointers, one for each
field, that point to conversion functions that are responsible for
converting input text data to the final output value.

A TextAdapter object also contains a set of function pointers to IO related
functions (open, seek, read, and close) reponsible for reading data from the
data source.  Compressed data seek and read functions can also be set if
source data is compressed.  By combining normal IO function pointers with
compressed data seek/read function pointers, the TextAdapter module can
easily handle any supported data source that is also compressed with one of
the supported compression schemes (currently only gzip).

A TextAdapter object also contains a function pointer that points to a
tokenizer function appropriate for the input text type.  The tokenizer
function is responsible for parsing the input text data and calling
process_token to convert text data into the final output data type for the
current field.  Each text type has a tokenizer function.  Tokenizer
functions are also implemented for parsing lines and records as single
string values (a record can be multiple lines).

Key Low Level C Data Structures
=======

TextAdapter (textadapter/core/text_adapter.h):

  Core struct for text parser.  Contains attributes for input text such as
  delimiter character, comment character, etc.  tokenize field is a function
  pointer to the tokenize function for parsing specific type of text (tab
  delimited, fixed width, etc).  Also contains pointers to InputDatastruct
  and TextAdapterBuffer described below.

InputData (textadapter/core/text_adapter.h):

  Contains function pointers for IO functions (open, read, seek, close) and
  for compressed data read and seek functions.  Also contains a void *input
  field for storing a data structure specific to each data source (C FILE
  pointer, S3 bucket info, etc).

TextAdapterBuffer (textadapter/core/text_adapter.h):

  Main buffer for storing text data to be parsed.

Ideas for Future Optimizations
=======

- The biggest performance gains could be had by incorporating some parallel
  processing goodness.  The most natural way to split it up (this should
  work for all the adapters) might be to have one thread/process that reads
  the input data into the main buffer, and a second thread/process do the
  actual parsing and converting of the data, and storing of the converted
  data in the final numpy array.

- Another idea for a potential speedup might be to refactor the parsing
  backend so that offsets for all the tokens for a field in the buffer are
  returned, and then have separate loops for different field types, that
  would power through all the tokens for a field and call the appropriate
  conversion function (the key would be to decide outside of the loops which
  loop+conversion function to execute, so that the conversion function would
  be inlined inside each loop.  This is essentially how the Pandas CSV
  reader works, but it would increase memory usage.  For example (in
  python-like pseudocode but implemented at the C level:

```
  if field_type is integers:
    for i in range(num_records):
        convert_and_store_ints(field_token_offsets[i])
  elif field_type is floats:
    for i in range(num_records):
        convert_and_store_floats(field_token_offsets[i])
```
