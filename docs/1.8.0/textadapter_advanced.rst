Advanced TextAdapter
====================

.. raw:: html

    <div class="section" id="gzip-support">
    <h2>Gzip Support<a class="headerlink" href="#gzip-support" title="Permalink to this headline">¶</a></h2>
    <p>IOPro can decompress gzip data on the fly, like so:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.gz&#39;</span><span class="p">,</span> <span class="n">compression</span><span class="o">=</span><span class="s1">&#39;gzip&#39;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[:]</span>
    </pre></div>
    </div>
    <p>Aside from the obvious advantage of being able to store and work with your
    compressed data without having to decompress first, you also don&#8217;t need to
    sacrifice any performance in doing so. For example, with a 419 MB csv file
    of numerical data, and a 105 MB file of the same data compressed with gzip,
    the following are the “best of three” run times for loading the entire
    contents of each file into a NumPy array:</p>
    <p>uncompressed: 13.38 sec gzip compressed: 14.54 sec</p>
    <p>The compressed file takes slightly longer, but consider having to uncompress
    the file to disk before loading with IOPro:</p>
    <p>uncompressed: 13.38 sec gzip compressed: 14.54 sec gzip compressed (decompress to disk, then load): 21.56 sec</p>
    </div>
    <div class="section" id="indexing-csv-data">
    <h2>Indexing CSV Data<a class="headerlink" href="#indexing-csv-data" title="Permalink to this headline">¶</a></h2>
    <p>One of the most useful features of IOPro is the ability to index data to allow
    for fast random lookup.</p>
    <p>For example, to retrieve the last record of the compressed 109 MB dataset we used above:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.gz&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">,</span> <span class="n">compression</span><span class="o">=</span><span class="s1">&#39;gzip&#39;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
    </pre></div>
    </div>
    <p>Retrieving the last record into a NumPy array takes 14.82 sec. This is about the same as the time
    to read the entire record, because the entire dataset has to be parsed to get to the last record.</p>
    <p>To make seeking faster, we can build an index:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">create_index</span><span class="p">(</span><span class="s1">&#39;index_file&#39;</span><span class="p">)</span>
    </pre></div>
    </div>
    <p>The above method creates an index in memory and saves it to disk, taking 9.48 sec.
    Now when seeking to and reading the last record again, it takes a mere 0.02 sec.</p>
    <p>Reloading the index only takes 0.18 sec. Build an index once, and get near instant random access
    to your data forever:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.gz&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">,</span> <span class="n">compression</span><span class="o">=</span><span class="s1">&#39;gzip&#39;</span><span class="p">,</span> <span class="n">index_name</span><span class="o">=</span><span class="s1">&#39;index_file&#39;</span><span class="p">)</span>
    </pre></div>
    </div>
    </div>
    <div class="section" id="advanced-regular-expressions">
    <h2>Advanced Regular Expressions<a class="headerlink" href="#advanced-regular-expressions" title="Permalink to this headline">¶</a></h2>
    <p>IOPro supports using regular expressions to help parse messy data.
    Take for example the following snippet of actual NASDAQ stock data found on the Internet:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="n">Apple</span><span class="p">,</span><span class="n">AAPL</span><span class="p">,</span><span class="n">NasdaqNM</span><span class="p">,</span><span class="mf">363.32</span> <span class="o">-</span> <span class="mf">705.07</span>
    <span class="n">Google</span><span class="p">,</span><span class="n">GOOG</span><span class="p">,</span><span class="n">NasdaqNM</span><span class="p">,</span><span class="mf">523.20</span> <span class="o">-</span> <span class="mf">774.38</span>
    <span class="n">Microsoft</span><span class="p">,</span><span class="n">MSFT</span><span class="p">,</span><span class="n">NasdaqNM</span><span class="p">,</span><span class="mf">24.30</span> <span class="o">-</span> <span class="mf">32.95</span>
    </pre></div>
    </div>
    <p>The first three fields are easy enough: name, symbol, and exchange. The fourth field presents a bit of a problem.
    Let&#8217;s try IOPro&#8217;s regular expression based parser:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">regex_string</span> <span class="o">=</span> <span class="s1">&#39;([A-Za-z]+),([A-Z]{1-4}),([A-Za-z]+),([0-9]+\.[0-9]</span><span class="si">{2}</span><span class="s1">)\s*-\s*([0-9]+\.[0-9]</span><span class="si">{2}</span><span class="s1">)&#39;</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.csv&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;regex&#39;</span><span class="p">,</span> <span class="n">regex_string</span><span class="o">=</span><span class="n">regex_string</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[:]</span>
    </pre></div>
    </div>
    <p>Regular expressions can admittedly get pretty ugly, but they can also be very powerful.
    By using the above regular expression with the grouping operators &#8216;(&#8216; and &#8216;)&#8217;, we can define
    exactly how each record should be parsed into fields. Let&#8217;s break it down into individual fields:</p>
    <p>([A-Za-z]+) defines the first field (stock name) in our output array,</p>
    <p>([A-Z]{1-4}) defines the second (stock symbol),</p>
    <p>([A-Za-z]+) defines the third (company name),</p>
    <p>([0-9]+.[0-9]{2}) defines the fourth field (low price), and</p>
    <p>([0-9]+.[0-9]{2}) defines the fifth field (high price)</p>
    <p>The output array contains five fields: three string fields and two float fields. Exactly what we want.</p>
    </div>
    <div class="section" id="numba-integration">
    <h2>Numba Integration<a class="headerlink" href="#numba-integration" title="Permalink to this headline">¶</a></h2>
    <p>IOPro comes with experimental integration with NumbaPro, the amazing NumPy aware Python compiler
    also available in Anaconda. Previously when parsing messy csv data, you had to use either a very slow
    custom Python converter function to convert the string data to the target data type, or use a complex
    regular expression to define the fields in each record string. Using the regular expression feature of
    IOPro will certainly still be a useful and valid option for certain types of data, but it would be nice
    if custom Python converter functions weren&#8217;t so slow as to be almost unusable. Numba solves this problem
    by compiling your converter functions on the fly without any action on your part. Simply set the converter
    function with a call to set_converter_function() as before, and IOPro + NumbaPro will handle the rest.
    To illustrate, I&#8217;ll show a trivial example using the sdss data set again. Take the following converter
    function which converts the input string to a floating point value and rounds to the nearest integer,
    returning the integer value:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="k">def</span> <span class="nf">convert_value</span><span class="p">(</span><span class="n">input_str</span><span class="p">):</span>
    <span class="gp">... </span>    <span class="n">float_value</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span><span class="n">input_str</span><span class="p">)</span>
    <span class="gp">... </span>    <span class="k">return</span> <span class="nb">int</span><span class="p">(</span><span class="nb">round</span><span class="p">(</span><span class="n">float_value</span><span class="p">))</span>
    </pre></div>
    </div>
    <p>We&#8217;ll use it to convert field 1 from the sdss dataset to an integer. By calling the set_converter method
    with the use_numba parameter set to either True or False (the default is True), we can test the converter
    function being called as both interpreted Python and as Numba compiled llvm bytecode. In this case,
    compiling the converter function with NumbaPro gives us a 5x improvement in run time performance. To put
    that in perspective, the Numba compiled converter function takes about the same time as converting field 1
    to a float value using IOPro&#8217;s built in C compiled float converter function. That isn&#8217;t quite an
    “apples to apples” comparison, but it does show that NumbaPro enables user defined python converter
    functions to achieve speeds in the same league as compiled C code.</p>
    </div>
    <div class="section" id="s3-support">
    <h2>S3 Support<a class="headerlink" href="#s3-support" title="Permalink to this headline">¶</a></h2>
    <p>Also in IOPro is the ability to parse csv data stored in Amazon&#8217;s S3 cloud storage service.
    The S3 text adapter constructor looks slightly different than the normal text adapter constructor:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">s3_text_adapter</span><span class="p">(</span><span class="n">aws_access_key</span><span class="p">,</span> <span class="n">aws_secret_key</span><span class="p">,</span> <span class="s1">&#39;dev-wakari-public&#39;</span><span class="p">,</span> <span class="s1">&#39;FEC/FEC_ALL.csv&#39;</span><span class="p">)</span>
    </pre></div>
    </div>
    <p>The first two parameters are your AWS access key and secret key, followed by the S3 bucket name and key name.
    The S3 csv data is downloaded in 128K chunks and parsed directly from memory, bypassing the need to save the
    entire S3 data set to disk first. IOPro can also build an index for S3 data just as with disk based csv data,
    and use the index for fast random access lookup. If an index file is created with IOPro and stored with the S3
    dataset in the cloud, IOPro can use this remote index to download and parse just the subset of records requested.
    This allows you to generate an index file once and share it on the cloud along with the data set, and does not
    require others to download the entire index file to use it.</p>
    </div>
