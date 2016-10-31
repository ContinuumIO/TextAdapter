***REMOVED***
TextAdapter
***REMOVED***

***REMOVED***

    <div class="contents topic" id="contents">
    <p class="topic-title first">Contents</p>
    <ul class="simple">
    <li><a class="reference internal" href="#textadapter" id="id1">TextAdapter</a><ul>
    <li><a class="reference internal" href="#methods" id="id2">Methods</a></li>
    <li><a class="reference internal" href="#basic-usage" id="id3">***REMOVED***</a></li>
    <li><a class="reference internal" href="#advanced-usage" id="id4">Advanced Usage</a></li>
    </ul>
    </li>
    </ul>
***REMOVED***
    <p>The TextAdapter module reads csv data and produces a NumPy array containing the
    parsed data. The following features are currently implemented:</p>
    <ul class="simple">
    <li>The TextAdapter engine is written
    in C to ensure text is parsed as fast as data can be read from the source.
    Text is read and parsed in small chunks instead of reading entire data into
    memory at once, which enables very large files to be read and parsed without
    running out of memory.</li>
    <li>Python slicing notation can be used to specify a subset of records to be
    read from the data source, as well as a subset of fields.</li>
    <li>In additional to specifying a delimiter character, fields can
    be specified by fixed field widths as well as a regular expression. This enables
    a larger variety of csv-like and other types of text files to be parsed.</li>
    <li>A gzipped file can be parsed without having to uncompress it first. Parsing speed
    is about the same as an uncompressed version of same file.</li>
    <li>An index of record offsets in a file can be built to allow fast random access to
    records. This index can be saved to disk and loaded again later.</li>
    <li>Converter functions can be specified for converting parsed text to proper dtype
    for storing in NumPy array. If Numba is installed, converter functions will
    be compiled to llvm bytecode on the fly for faster execution.</li>
    <li>The TextAdapter engine has automatic type inference so the user does not have to
    specify dtypes of the output array. The user can still specify dtypes manually if
    desired.</li>
    <li>Remote data stored in Amazon S3 can be read. An index can be built and stored
    with S3 data. Index can be read remotely, allowing for random access to S3 data.</li>
    </ul>
    <div class="section" id="methods">
    <h2><a class="toc-backref" href="#id2">Methods</a><a class="headerlink" href="#methods" title="Permalink to this headline">¶</a></h2>
    <p>The TextAdapter module contains the following factory methods for creating TextAdapter objects:</p>
    <dl class="docutils">
    <dt><strong>text_adapter</strong> (source, parser=&#8217;csv&#8217;, compression=None, comment=&#8217;#&#8217;,</dt>
    <dd><blockquote class="first">
***REMOVED***quote=&#8217;&#8221;&#8217;, num_records=0, header=0, field_names=True,
    indexing=False, index_name=None, encoding=&#8217;utf-8&#8217;)</div></blockquote>
    <div class="line-block">
    <div class="line">Create a text adapter for reading CSV, JSON, or fixed width</div>
    <div class="line">text files, or a text file defined by regular expressions.</div>
***REMOVED***
    <div class="last line-block">
    <div class="line">source - filename, file object, StringIO object, BytesIO object, S3 key,
    http url, or python generator</div>
    <div class="line">parser - Type of parser for parsing text. Valid parser types are &#8216;csv&#8217;, &#8216;fixed width&#8217;, &#8216;regex&#8217;, and &#8216;json&#8217;.</div>
    <div class="line">encoding - type of character encoding (current ascii and utf8 are supported)</div>
    <div class="line">compression - type of data compression (currently only gzip is supported)</div>
    <div class="line">comment - character used to indicate comment line</div>
    <div class="line">quote - character used to quote fields</div>
    <div class="line">num_records - limits parsing to specified number of records; defaults
    to all records</div>
    <div class="line">header - number of lines in file header; these lines are skipped when parsing</div>
    <div class="line">footer - number of lines in file footer; these lines are skipped when parsing</div>
    <div class="line">indexing - create record index on the fly as characters are read</div>
    <div class="line">index_name - name of file to write index to</div>
    <div class="line">output - type of output object (numpy array or pandas dataframe)</div>
***REMOVED***
    </dd>
    <dt>If parser is set to &#8216;csv&#8217;, additional parameters include:</dt>
    <dd><div class="first last line-block">
    <div class="line">delimiter - Delimiter character used to define fields in data source. Default is &#8216;,&#8217;.</div>
***REMOVED***
    </dd>
    <dt>If parser is set to &#8216;fixed_width&#8217;, additional parameters include:</dt>
    <dd><div class="first last line-block">
    <div class="line">field_widths - List of field widths</div>
***REMOVED***
    </dd>
    <dt>If parser is set to &#8216;regex&#8217;, additional parameters include:</dt>
    <dd><div class="first last line-block">
    <div class="line">regex - Regular expression used to define records and fields in data source.
    See the regular expression example in the Advanced Usage section.</div>
***REMOVED***
    </dd>
    <dt><strong>s3_text_adapter</strong> (access_key, secret_key, bucket_name, key_name, remote_s3_index=False)</dt>
    <dd><blockquote class="first">
***REMOVED***parser=&#8217;csv&#8217;, compression=None, comment=&#8217;#&#8217;,
    quote=&#8217;&#8221;&#8217;, num_records=0, header=0, field_names=True,
    indexing=False, index_name=None, encoding=&#8217;utf-8&#8217;)</div></blockquote>
    <div class="last line-block">
    <div class="line">Create a text adapter for reading a text file from S3. Text file can be</div>
    <div class="line">CSV, JSON, fixed width, or defined by regular expressions</div>
***REMOVED***
    </dd>
    </dl>
    <p>In addition to the arguments described for the text_adapter function above,
    the s3_text_adapter function also has the following parameters:</p>
    <blockquote>
***REMOVED***<div class="line-block">
    <div class="line">access_key - AWS access key</div>
    <div class="line">secret_key - AWS secret key</div>
    <div class="line">bucket_name - name of S3 bucket</div>
    <div class="line">key_name - name of key in S3 bucket</div>
    <div class="line">remote_s3_index - use remote S3 index (index name must be key name + &#8216;.idx&#8217; extension)</div>
***REMOVED***
***REMOVED***</blockquote>
    <p>The TextAdapter object returned by the text_adapter factory method contains the following methods:</p>
    <dl class="docutils">
    <dt><strong>set_converter</strong> (field, converter, use_numba=True)</dt>
    <dd><div class="first line-block">
    <div class="line">Set converter function for field</div>
***REMOVED***
    <div class="last line-block">
    <div class="line">field - field to apply converter function</div>
    <div class="line">converter - python function object</div>
    <div class="line">use_numba - If true, numba will be used to compile function.
    Otherwise the function will be executed as a normal Python
    function, resulting in slower performance.</div>
***REMOVED***
    </dd>
    <dt><strong>set_missing_values</strong> (missing_values)</dt>
    <dd><div class="first line-block">
    <div class="line">Set strings for each field that represents a missing value</div>
***REMOVED***
    <div class="line-block">
    <div class="line">missing_values - dict of field name or number,
    and list of missing value strings</div>
***REMOVED***
    <p class="last">Default missing values: &#8216;NA&#8217;, &#8216;NaN&#8217;, &#8216;inf&#8217;, &#8216;-inf&#8217;, &#8216;None&#8217;, &#8216;none&#8217;, &#8216;&#8217;</p>
    </dd>
    <dt><strong>set_fill_values</strong> (fill_values, loose=False)</dt>
    <dd><div class="first line-block">
    <div class="line">Set fill values for each field</div>
***REMOVED***
    <div class="line-block">
    <div class="line">fill_values - dict of field name or number, and fill value</div>
    <div class="line">loose - If value cannot be converted, and value does not match
    any of the missing values, replace with fill value anyway.</div>
***REMOVED***
    <p class="last">Default fill values for each data type:
    | int - 0
    | float - numpy.nan
    | char - 0
    | bool - False
    | object - numpy.nan
    | string - numpy.nan</p>
    </dd>
    <dt><strong>create_index</strong> (index_name=None, density=1)</dt>
    <dd><div class="first line-block">
    <div class="line">Create an index of record offsets in file</div>
***REMOVED***
    <div class="last line-block">
    <div class="line">index_name - Name of file on disk used to store index. If None, index
    will be created in memory but not saved.</div>
    <div class="line">density - density of index. Value of 1 will index every record, value of
    2 will index every other record, etc.</div>
***REMOVED***
    </dd>
    <dt><strong>to_array</strong> ()</dt>
    <dd><div class="first last line-block">
    <div class="line">Parses entire data source and returns data as NumPy array object</div>
***REMOVED***
    </dd>
    <dt><strong>to_dataframe</strong> ()</dt>
    <dd><div class="first last line-block">
    <div class="line">Parses entire data source and returns data as Pandas DataFrame object</div>
***REMOVED***
    </dd>
    </dl>
    <p>The TextAdapter object contains the following properties:</p>
    <dl class="docutils">
    <dt><strong>size</strong> (readonly)</dt>
    <dd><div class="first last line-block">
    <div class="line">Number of records in data source. This value is only set if entire data
    source has been read or indexed, or number of recods was specified in
    text_adapter factory method when creating object.</div>
***REMOVED***
    </dd>
    <dt><strong>field_count</strong> (readonly)</dt>
    <dd><div class="first last line-block">
    <div class="line">Number of fields in each record</div>
***REMOVED***
    </dd>
    <dt><strong>field_names</strong></dt>
    <dd><div class="first last line-block">
    <div class="line">Field names to use when creating output NumPy array. Field names can be
    set here before reading data or in text_adapter function with
    field_names parameter.</div>
***REMOVED***
    </dd>
    <dt><strong>field_types</strong></dt>
    <dd><div class="first last line-block">
    <div class="line">NumPy dtypes for each field, specified as a dict of fields and associated
    dtype. (Example: {0:&#8217;u4&#8217;, 1:&#8217;f8&#8217;, 2:&#8217;S10&#8217;})</div>
***REMOVED***
    </dd>
    <dt><strong>field_filter</strong></dt>
    <dd><div class="first line-block">
    <div class="line">Fields in data source to parse, specified as a list of field numbers
    or names (Examples: [0, 1, 2] or [&#8216;f1&#8217;, &#8216;f3&#8217;, &#8216;f5&#8217;]). This filter stays
    in effect until it is reset to empty list, or is overridden with array
    slicing (Example: adapter[[0, 1, 3, 4]][:]).</div>
***REMOVED***
    <dl class="last docutils">
    <dt>See the NumPy data types documentation for more details:</dt>
    <dd><a class="reference external" href="http://docs.continuum.io/anaconda/numpy/reference/arrays.dtypes.html">http://docs.continuum.io/anaconda/numpy/reference/arrays.dtypes.html</a></dd>
    </dl>
    </dd>
    </dl>
    <p>The TextAdapter object supports array slicing:</p>
    <blockquote>
***REMOVED***<div class="line-block">
    <div class="line">Read all records:
    adapter[:]</div>
***REMOVED***
    <div class="line-block">
    <div class="line">Read first 100 records:
    adapter[0:100]</div>
***REMOVED***
    <div class="line-block">
    <div class="line">Read last record (only if data has been indexed or entire dataset
    has been read once before):
    adapter[-1]</div>
***REMOVED***
    <div class="line-block">
    <div class="line">Read first field in all records by specifying field number:
    adapter[0][:]</div>
***REMOVED***
    <div class="line-block">
    <div class="line">Read first field in all records by specifying field name:
    adapter[&#8216;f0&#8217;][:]</div>
***REMOVED***
    <div class="line-block">
    <div class="line">Read first and third fields in all records:
    adapter[[0, 2]][:]</div>
***REMOVED***
***REMOVED***</blockquote>
***REMOVED***
    <div class="section" id="basic-usage">
    <h2><a class="toc-backref" href="#id3">***REMOVED***</a><a class="headerlink" href="#basic-usage" title="Permalink to this headline">¶</a></h2>
    <p>Create TextAdapter object for data source:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.csv&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">)</span>
    </pre></div>
***REMOVED***
    <p>Parse text and store records in NumPy array using slicing notation:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># read all records</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[:]</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># read first ten records</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">10</span><span class="p">]</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># read last record</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># read every other record</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[::</span><span class="mi">2</span><span class="p">]</span>
    </pre></div>
***REMOVED***
***REMOVED***
    <div class="section" id="advanced-usage">
    <h2><a class="toc-backref" href="#id4">Advanced Usage</a><a class="headerlink" href="#advanced-usage" title="Permalink to this headline">¶</a></h2>
    <p>user defined converter function for field 0:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">io</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="s1">&#39;1, abc, 3.3</span><span class="se">\n</span><span class="s1">2, xxx, 9.9&#39;</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="n">io</span><span class="o">.</span><span class="n">StringIO</span><span class="p">(</span><span class="n">data</span><span class="p">),</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">,</span> <span class="n">field_names</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Override default converter for first field</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">set_converter</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="nb">int</span><span class="p">(</span><span class="n">x</span><span class="p">)</span><span class="o">*</span><span class="mi">2</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="p">[:]</span>
    <span class="go">array([(2L, &#39; abc&#39;, 3.3), (4L, &#39; xxx&#39;, 9.9)],</span>
    <span class="go">          dtype=[(&#39;f0&#39;, &#39;&lt;u8&#39;), (&#39;f1&#39;, &#39;S4&#39;), (&#39;f2&#39;, &#39;&lt;f8&#39;)])</span>
    </pre></div>
***REMOVED***
    <p>overriding default missing and fill values:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">io</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="s1">&#39;1,abc,inf</span><span class="se">\n</span><span class="s1">2,NA,9.9&#39;</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="n">io</span><span class="o">.</span><span class="n">StringIO</span><span class="p">(</span><span class="n">data</span><span class="p">),</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">,</span> <span class="n">field_names</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Define field dtypes (example: set field 1 to string object and field 2 to float)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">field_types</span> <span class="o">=</span> <span class="p">{</span><span class="mi">1</span><span class="p">:</span><span class="s1">&#39;O&#39;</span><span class="p">,</span> <span class="mi">2</span><span class="p">:</span><span class="s1">&#39;f4&#39;</span><span class="p">}</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Define list of strings for each field that represent missing values</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">set_missing_values</span><span class="p">({</span><span class="mi">1</span><span class="p">:[</span><span class="s1">&#39;NA&#39;</span><span class="p">],</span> <span class="mi">2</span><span class="p">:[</span><span class="s1">&#39;inf&#39;</span><span class="p">]})</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Set fill value for missing values in each field</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">set_fill_values</span><span class="p">({</span><span class="mi">1</span><span class="p">:</span><span class="s1">&#39;xxx&#39;</span><span class="p">,</span> <span class="mi">2</span><span class="p">:</span><span class="mf">999.999</span><span class="p">})</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="p">[:]</span>
    <span class="go">array([(&#39; abc&#39;, 999.9990234375), (&#39;xxx&#39;, 9.899999618530273)],</span>
    <span class="go">          dtype=[(&#39;f0&#39;, &#39;O&#39;), (&#39;f1&#39;, &#39;&lt;f4&#39;)])</span>
    </pre></div>
***REMOVED***
    <p>creating and saving tuple of index arrays for gzip file, and reloading indices:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.gz&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">,</span> <span class="n">compression</span><span class="o">=</span><span class="s1">&#39;gzip&#39;</span><span class="p">)</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Build index of records and save index to disk.</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">create_index</span><span class="p">(</span><span class="n">index_name</span><span class="o">=</span><span class="s1">&#39;index_file&#39;</span><span class="p">)</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Create new adapter object and load index from disk.</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.gz&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">,</span> <span class="n">compression</span><span class="o">=</span><span class="s1">&#39;gzip&#39;</span><span class="p">,</span> <span class="n">indexing</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">index_name</span><span class="o">=</span><span class="s1">&#39;index_file&#39;</span><span class="p">)</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Read last record</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
    <span class="go">array([(100, 101, 102)],dtype=[(&#39;f0&#39;, &#39;&lt;u4&#39;), (&#39;f1&#39;, &#39;&lt;u4&#39;), (&#39;f2&#39;, &#39;&lt;u4&#39;)])</span>
    </pre></div>
***REMOVED***
    <p>Use regular expression for finer control of extracting data:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">io</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Define regular expression to extract dollar amount, percentage, and month.</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Each set of parentheses defines a field.</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="s1">&#39;$2.56, 50%, September 20 1978</span><span class="se">\n</span><span class="s1">$1.23, 23%, April 5 1981&#39;</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">regex_string</span> <span class="o">=</span> <span class="s1">&#39;([0-9]\.[0-9][0-9]+)\,\s ([0-9]+)\%\,\s ([A-Za-z]+)&#39;</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="n">io</span><span class="o">.</span><span class="n">StringIO</span><span class="p">(</span><span class="n">data</span><span class="p">),</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;regex&#39;</span><span class="p">,</span> <span class="n">regex_string</span><span class="o">=</span><span class="n">regex_string</span><span class="p">,</span> <span class="n">field_names</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">infer_types</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># set dtype of field to float</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">field_types</span> <span class="o">=</span> <span class="p">{</span><span class="mi">0</span><span class="p">:</span><span class="s1">&#39;f4&#39;</span><span class="p">,</span> <span class="mi">1</span><span class="p">:</span><span class="s1">&#39;u4&#39;</span><span class="p">,</span> <span class="mi">2</span><span class="p">:</span><span class="s1">&#39;S10&#39;</span><span class="p">}</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="p">[:]</span>
    <span class="go">array([(2.56, 50L, &#39;September&#39;), (1.23, 23L, &#39;April&#39;)],</span>
    <span class="go">    dtype=[(&#39;f0&#39;, &#39;&lt;f8&#39;), (&#39;f1&#39;, &#39;&lt;u8&#39;), (&#39;f2&#39;, &#39;S9&#39;)])</span>
    </pre></div>
***REMOVED***
***REMOVED***
