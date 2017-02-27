-----------------------
TextAdapter First Steps
-----------------------

Basic Usage
-----------

IOPro works by attaching to a data source, such as a local CSV file. Before we
get started, let's create a sample CSV file to work with::

    from random import random, randint, shuffle
    import string

    NUMROWS = 10
    with open('data/table.csv','w') as data:
        # Header
        for n in range(1,5):
            print("f%d" % n, end=",", file=data)
        print("comment", file=data)

        # Body
        letters = list(string.ascii_letters)
        for n in range(NUMROWS):
            shuffle(letters)
            s = "".join(letters[:randint(5,20)])
            vals = (n, randint(1000,2000), random(), random()*100, s)
            print("%d,%d,%f,%f,%s" % vals, file=data)

Let's read in the local CSV file created here. Because this small file
easily fits in memory it would work to use the :code:`csv` or :code:`pandas`
modules, but we will demonstrate the interfaces and capabilities that will
apply to much larger data.

    >>> import iopro
    >>> adapter = iopro.text_adapter('data/table.csv', parser='csv')
    >>> adapter.get_field_names()
    ['f1', 'f2', 'f3', 'f4', 'comment']

We can specify the data types for values in the columns of the CSV file being
read;  but first we look at the ability of IOPro's TextAdapter to
auto-discover the data types used.

We can ask IOPro's TextAdapter to parse text and return records in NumPy
arrays from selected portions of the CSV file using slicing notation:

    >>> # the inferred datatypes
    >>> array = adapter[:]
    >>> array.dtype
    dtype([('f1', '<u8'), ('f2', '<u8'), ('f3', '<f8'), ('f4', '<f8'),
           ('comment', 'O')])

Let's define field dtypes (example: set field 0 to a 16-bit unsigned int and
field 3 to a 32-bit float).

Massage the datatypes:

    >>> adapter.set_field_types({0: 'u2', 3:'f4'})
    >>> array = adapter[:]
    >>> array.dtype
    dtype([('f1', '<u2'), ('f2', '<u8'), ('f3', '<f8'), ('f4', '<f4'),
           ('comment', 'O')])

The first five records:

    >>> array = adapter[0:5]
    >>> print(array)
    [(0, 1222, 0.926116, 84.44437408447266, 'MlzvBRyquns')
     (1, 1350, 0.553585, 81.03726959228516, 'ikgEauJeTZvd')
     (2, 1932, 0.710919, 31.59865951538086, 'uUQmHJFZhnirecAvx')
     (3, 1494, 0.622391, 57.90607452392578, 'iWQBAZodkfHODtI')
     (4, 1981, 0.820246, 40.848018646240234, 'igxeXdBpqE')]

Read last five records:

    >>> array = adapter[-5:]
    >>> print(array)
    [(5, 1267, 0.694631, 6.999039173126221, 'bRSrwitHeY')
     (6, 1166, 0.37465, 38.7022705078125, 'qzbMgVThXtHpfDNrd')
     (7, 1229, 0.390566, 55.338134765625, 'hyarmvWi')
     (8, 1816, 0.201106, 59.74718475341797, 'DcHymelRusO')
     (9, 1416, 0.725697, 42.50992965698242, 'QMUGRAwe')]

Read every other record:

    >>> array = adapter[::2]
    >>> print(array)
    [(0, 1222, 0.926116, 84.44437408447266, 'MlzvBRyquns')
     (2, 1932, 0.710919, 31.59865951538086, 'uUQmHJFZhnirecAvx')
     (4, 1981, 0.820246, 40.848018646240234, 'igxeXdBpqE')
     (6, 1166, 0.37465, 38.7022705078125, 'qzbMgVThXtHpfDNrd')
     (8, 1816, 0.201106, 59.74718475341797, 'DcHymelRusO')]

Read first and second, third fields only:

    >>> array = adapter[[0,1,2]][:]
    >>> list(array)
    [(0, 1222, 0.926116),
     (1, 1350, 0.553585),
     (2, 1932, 0.710919),
     (3, 1494, 0.622391),
     (4, 1981, 0.820246),
     (5, 1267, 0.694631),
     (6, 1166, 0.37465),
     (7, 1229, 0.390566),
     (8, 1816, 0.201106),
     (9, 1416, 0.725697)]

Read fields named 'f2' and 'comment' only:

    >>> array = adapter[['f2','comment']][:]
    >>> list(array)
    [(1222, 'MlzvBRyquns'),
     (1350, 'ikgEauJeTZvd'),
     (1932, 'uUQmHJFZhnirecAvx'),
     (1494, 'iWQBAZodkfHODtI'),
     (1981, 'igxeXdBpqE'),
     (1267, 'bRSrwitHeY'),
     (1166, 'qzbMgVThXtHpfDNrd'),
     (1229, 'hyarmvWi'),
     (1816, 'DcHymelRusO'),
     (1416, 'QMUGRAwe')]


JSON Support
------------

Text data in JSON format can be parsed by specifying 'json' for the
parser argument:

Content of file :code:`data/one.json`:

.. parsed-literal::

    {"id":123, "name":"xxx"}

Single JSON object:

    >>> adapter = iopro.text_adapter('data/one.json', parser='json')
    >>> adapter[:]
    array([(123, 'xxx')],
          dtype=[('id', '<u8'), ('name', 'O')])

Currently, each JSON object at the root level is interpreted as a single
NumPy record. Each JSON object can be part of an array, or separated by
a newline. Examples of valid JSON documents that can be parsed by IOPro,
with the NumPy array result:

Content of file :code:`data/two.json`:

.. parsed-literal::

    [{"id":123, "name":"xxx"}, {"id":456, "name":"yyy"}]

Array of two JSON objects:

    >>> iopro.text_adapter('data/two.json', parser='json')[:]
    array([(123, 'xxx'), (456, 'yyy')],
          dtype=[('id', '<u8'), ('name', 'O')])

Content of file :code:`data/three.json`:

.. parsed-literal::

    {"id":123, "name":"xxx"}
    {"id":456, "name":"yyy"}

Two JSON objects separated by newline:

    >>> iopro.text_adapter('data/three.json', parser='json')[:]
    array([(123, 'xxx'), (456, 'yyy')],
          dtype=[('id', '<u8'), ('name', 'O')])


Massaging data in the adapter
-----------------------------

A custom function can be used to modify values as they are read.

    >>> import iopro, io, math
    >>> stream = io.StringIO('3,abc,3.3\n7,xxx,9.9\n4,,')
    >>> adapter = iopro.text_adapter(stream, parser='csv', field_names=False)

Override default converter for first field:

    >>> adapter.set_converter(0, lambda x: math.factorial(int(x)))
    >>> adapter[:]
    array([(6, 'abc', 3.3), (5040, 'xxx', 9.9), (24, '', nan)],
          dtype=[('f0', '<u8'), ('f1', 'O'), ('f2', '<f8')])

We can also force data types and set fill values for missing data.

Apply data types to columns:

    >>> stream = io.StringIO('3,abc,3.3\n7,xxx,9.9\n4,,')
    >>> adapter = iopro.text_adapter(stream, parser='csv', field_names=False)
    >>> adapter.set_field_types({1:'S3', 2:'f4'})
    >>> adapter[:]
    array([(3, b'abc', 3.299999952316284), (7, b'xxx', 9.899999618530273),
           (4, b'', nan)],
          dtype=[('f0', '<u8'), ('f1', 'S3'), ('f2', '<f4')])

Set fill value for missing values in each field:

    >>> adapter.set_fill_values({1:'ZZZ', 2:999.999})
    >>> adapter[:]
    array([(3, b'abc', 3.299999952316284), (7, b'xxx', 9.899999618530273),
           (4, b'ZZZ', 999.9990234375)],
          dtype=[('f0', '<u8'), ('f1', 'S3'), ('f2', '<f4')])


Combining regular expressions and typecasting
---------------------------------------------

A later section discusses regular expressions in more detail.  This example
is a quick peek into using them with IOPro.

Content of the file :code:`data/transactions.csv`:

.. parsed-literal::

    $2.56, 50%, September 20 1978
    $1.23, 23%, April 5 1981

Combining features:

    >>> import iopro
    >>> regex_string = '\$(\d)\.(\d{2}),\s*([0-9]+)\%,\s*([A-Za-z]+)'
    >>> adapter = iopro.text_adapter('data/transactions.csv',
    ...                              parser='regex',
    ...                              regex_string=regex_string,
    ...                              field_names=False,
    ...                              infer_types=False)

Set dtype of fields and their names:

    >>> adapter.set_field_types({0:'i2', 1:'u2', 2:'f4', 3:'S10'})
    >>> adapter.set_field_names(['dollars', 'cents', 'percentage', 'month'])
    >>> adapter[:]
    array([(2, 56, 50.0, b'September'), (1, 23, 23.0, b'April')],
          dtype=[('dollars', '<i2'), ('cents', '<u2'),
                 ('percentage', '<f4'), ('month', 'S10')])


--------------------
Advanced TextAdapter
--------------------

``iopro.loadtext()`` versus ``iopro.genfromtxt()``
--------------------------------------------------

Within IOPro there are two closely related functions. ``loadtext()``,
which we have been looking at, makes a more optimistic assumption that
your data is well-formatted. ``genfromtxt()`` has a number of arguments
for handling messier data, and special behaviors for dealing with
missing data.

``loadtext()`` is already highly configurable for dealing with data
under many CSV and other delimited formats. ``genfromtxt()`` contains
a superset of these arguments.


Gzip Support
------------

IOPro can decompress gzip'd data on the fly, simply by indicating a
``compression`` keyword argument.

   >>> adapter = iopro.text_adapter('data.gz', parser='csv', compression='gzip')
   >>> array = adapter[:]

As well as being able to store and work with your compressed data without
having to decompress it first, you also do not need to sacrifice any
performance in doing so. For example, with one test 419 MB CSV file of
numerical data, and a 105 MB file of the same data compressed with gzip, the
following are run times on a test machine for loading the entire contents of
each file into a NumPy array.  Exact performance will vary between
machines, especially between machines with HDD and SSD architecture.::

-  uncompressed: 13.38 sec
-  gzip compressed: 14.54 sec

In the test, the compressed file takes slightly longer, but consider having to
uncompress the file to disk before loading with IOPro:

-  uncompressed: 13.38 sec
-  gzip compressed: 14.54 sec
-  gzip compressed (decompress to disk, then load): 21.56 sec


Indexing CSV Data
-----------------

One of the most useful features of IOPro is the ability to index data to
allow for fast random lookup.

For example, to retrieve the last record of the compressed 109 MB
dataset we used above::

   >>> adapter = iopro.text_adapter('data.gz', parser='csv', compression='gzip')
   >>> array = adapter[-1]

Retrieving the last record into a NumPy array takes 14.82 sec. This is
about the same as the time to read the entire array, because the entire
dataset has to be parsed to get to the last record.

To make seeking faster, we can build an index:

   >>> adapter.create_index('index_file')

The above method creates an index in memory and saves it to disk, taking
9.48 sec. Now when seeking to and reading the last record again, it
takes a mere 0.02 sec.

Reloading the index only takes 0.18 sec. If you build an index once, you get
near instant random access to your data forever (assuming the data remains
static)::

   >>> adapter = iopro.text_adapter('data.gz', parser='csv',
   ...                              compression='gzip',
   ...                              index_name='index_file')

Let's try it with a moderate sized example.  You can download this data from
the `Exoplanets Data Explorer <http://exoplanets.org/csv>`_ site.

   >>> adapter = iopro.text_adapter('data/exoplanets.csv.gz',
   ...                              parser='csv', compression='gzip')
   >>> print(len(adapter[:]), "rows")
   >>> print(', '.join(adapter.field_names[:3]),
   ...       '...%d more...\n   ' % (adapter.field_count-6),
   ...       ', '.join(adapter.field_names[-3:]))
   2042 rows
   name, mass, mass_error_min ...73 more...
       star_teff, star_detected_disc, star_magnetic_field

   >>> adapter.field_types
   {0: dtype('O'),
    1: dtype('float64'),
    2: dtype('float64'),
    3: dtype('O'),
    4: dtype('float64'),
    5: dtype('float64'),
    6: dtype('float64'),
    7: dtype('float64'),
    8: dtype('O'),
    9: dtype('float64'),
    [... more fields ...]
    69: dtype('float64'),
    70: dtype('float64'),
    71: dtype('float64'),
    72: dtype('float64'),
    73: dtype('float64'),
    74: dtype('O'),
    75: dtype('float64'),
    76: dtype('float64'),
    77: dtype('O'),
    78: dtype('uint64')}

Do some timing (using an IPython magic):

   >>> %time row = adapter[-1]
   CPU times: user 35 ms, sys: 471 µs, total: 35.5 ms
   Wall time: 35.5 ms

   >>> %time adapter.create_index('data/exoplanets.index')
   CPU times: user 15.7 ms, sys: 3.35 ms, total: 19.1 ms
   Wall time: 18.6 ms

   >>> %time row = adapter[-1]
   CPU times: user 18.3 ms, sys: 1.96 ms, total: 20.3 ms
   Wall time: 20.1 ms

   >>> new_adapter = iopro.text_adapter('data/exoplanets.csv.gz', parser='csv',
   ...                                  compression='gzip',
   ...                                  index_name='data/exoplanets.index')

   >>> %time row = new_adapter[-1]
   CPU times: user 17.3 ms, sys: 2.12 ms, total: 19.4 ms
   Wall time: 19.4 ms


Regular Expressions
-------------------

   Some people, when confronted with a problem, think "I know, I'll use
   regular expressions." Now they have two problems. --Jamie Zawinski

IOPro supports using regular expressions to help parse messy data. Take
for example the following snippet of actual NASDAQ stock data found on
the Internet:

The content of the file :code:`data/stocks.csv`:

.. parsed-literal::

   Name,Symbol,Exchange,Range
   Apple,AAPL,NasdaqNM,363.32 - 705.07
   Google,GOOG,NasdaqNM,523.20 - 774.38
   Microsoft,MSFT,NasdaqNM,24.30 - 32.95

The first three fields are easy enough: name, symbol, and exchange. The
fourth field presents a bit of a problem. Let's try IOPro's regular
expression based parser:

    >>> regex_string = '([A-Za-z]+),([A-Z]{1,4}),([A-Za-z]+),'\
    ...                '(\d+.\.\d{2})\s*\-\s*(\d+.\.\d{2})'
    >>> adapter = iopro.text_adapter('data/stocks.csv', parser='regex',
    ...                              regex_string=regex_string)

    >>> # Notice that header does not now match the regex
    >>> print(adapter.field_names)
    ['Name,Symbol,Exchange,Range', '', '', '', '']

    >>> # We can massage the headers to reflect our match pattern
    >>> info = adapter.field_names[0].split(',')[:3]
    >>> adapter.field_names =  info + ["Low", "High"]
    >>> adapter[:]
    array([('Apple', 'AAPL', 'NasdaqNM', 363.32, 705.07),
           ('Google', 'GOOG', 'NasdaqNM', 523.2, 774.38),
           ('Microsoft', 'MSFT', 'NasdaqNM', 24.3, 32.95)],
           dtype=[('Name', 'O'), ('Symbol', 'O'),
                  ('Exchange', 'O'), ('Low', '<f8'), ('High', '<f8')])

Regular expressions are compact and often difficult to read, but they
are also very powerful. By using the above regular expression with the
grouping operators '(' and ')', we can define exactly how each record
should be parsed into fields. Let's break it down into individual
fields:

-  ``([A-Za-z]+)`` defines the first field (stock name) in our output array
-  ``([A-Z]{1-4})`` defines the second (stock symbol)
-  ``([A-Za-z]+)`` defines the third (exchange name)
-  ``(\d+.\.\d{2})`` defines the fourth field (low price)
-  ``\s*\-\s*`` is skipped because it is not part of a group
-  ``(\d+.\.\d{2})`` defines the fifth field (high price)


The output array contains five fields: three string fields and two float
fields. Exactly what we want.


S3 Support
----------

IOPro can parse CSV data stored in Amazon's S3 cloud storage service. In
order to access S3 files, you need to specify some credentials along
with the resource you are accessing.

The first two parameters are your AWS access key and secret key,
followed by the S3 bucket name and key name. The S3 CSV data is
downloaded in 128K chunks and parsed directly from memory, bypassing the
need to save the entire S3 data set to local disk.

Let's take a look at what we have stored from the Health Insurance Marketplace
data.  There's a little bit of code with BeautifulSoup just to prettify the
raw XML query results.

    >>> import urllib.request
    >>> url = 'http://s3.amazonaws.com/product-training/'
    >>> xml = urllib.request.urlopen(url).read()

    >>> import bs4, re
    >>> r = re.compile(r'^(\s*)', re.MULTILINE)
    >>> def display(bs, encoding=None, formatter="minimal", indent=4):
    ...     print(r.sub(r'\1' * indent, bs.prettify(encoding, formatter)))
    >>> display(bs4.BeautifulSoup(xml, "xml"))
    <?xml version="1.0" encoding="utf-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
        <Name>
            product-training
        </Name>
        <Prefix/>
        <Marker/>
        <MaxKeys>
            1000
        </MaxKeys>
        <IsTruncated>
            false
        </IsTruncated>
        <Contents>
            <Key>
                BusinessRules.csv
            </Key>
            <LastModified>
                2016-06-25T00:03:20.000Z
            </LastModified>
            <ETag>
                "a565ebede6a7e6e060cd4526a7ae4345"
            </ETag>
            <Size>
                8262590
            </Size>
            <StorageClass>
                STANDARD
            </StorageClass>
        </Contents>
        <Contents>
            [... more files ...]
        </Contents>
    </ListBucketResult>

In simple form, we see details about some S3 resources.  Let's access one of
them. Note that you will need to fill in your actual AWS access key and secret key.

    >>> user_name = "class1"
    >>> aws_access_key = "ABCD"
    >>> aws_secret_key = "EFGH/IJK"
    >>> bucket = 'product-training'
    >>> key_name = 'BusinessRules.csv' # 21k lines, 8MB
    >>> # key_name = 'PlanAttributes.csv' # 77k lines, 95MB
    >>> # key_name = 'Rate.csv.gzip' # 13M lines, 2GB raw, 110MB compressed
    >>> adapter = iopro.s3_text_adapter(aws_access_key, aws_secret_key,
    ...                                 bucket, key_name)
    >>> # Don't try with the really large datasets, works with the default one
    >>> df = adapter.to_dataframe()
    >>> df.iloc[:6,:6]

.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>BusinessYear</th>
          <th>StateCode</th>
          <th>IssuerId</th>
          <th>SourceName</th>
          <th>VersionNum</th>
          <th>ImportDate</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2014</td>
          <td>AL</td>
          <td>82285</td>
          <td>HIOS</td>
          <td>7</td>
          <td>2014-01-21 08:29:49</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2014</td>
          <td>AL</td>
          <td>82285</td>
          <td>HIOS</td>
          <td>7</td>
          <td>2014-01-21 08:29:49</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2014</td>
          <td>AL</td>
          <td>82285</td>
          <td>HIOS</td>
          <td>7</td>
          <td>2014-01-21 08:29:49</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2014</td>
          <td>AL</td>
          <td>82285</td>
          <td>HIOS</td>
          <td>7</td>
          <td>2014-01-21 08:29:49</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2014</td>
          <td>AL</td>
          <td>82285</td>
          <td>HIOS</td>
          <td>7</td>
          <td>2014-01-21 08:29:49</td>
        </tr>
        <tr>
          <th>5</th>
          <td>2014</td>
          <td>AZ</td>
          <td>17100</td>
          <td>HIOS</td>
          <td>7</td>
          <td>2013-10-15 07:27:56</td>
        </tr>
      </tbody>
    </table>
    </div>

IOPro can also build an index for S3 data just as with disk based CSV
data, and use the index for fast random access lookup. If an index file
is created with IOPro and stored with the S3 dataset in the cloud, IOPro
can use this remote index to download and parse just the subset of
records requested. This allows you to generate an index file once and
share it on the cloud along with the data set, and does not require
others to download the entire index file to use it.
