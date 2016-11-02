<<<<<<< HEAD
***REMOVED***
=======================

***REMOVED***

    <div class="section" id="basic-usage">
    <h2>***REMOVED***<a class="headerlink" href="#basic-usage" title="Permalink to this headline">¶</a></h2>
=======
TextAdapter First Steps
=======================

.. raw:: html

    <div class="section" id="basic-usage">
    <h2>Basic Usage<a class="headerlink" href="#basic-usage" title="Permalink to this headline">¶</a></h2>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    <p>Create TextAdapter object for data source:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.csv&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">)</span>
    </pre></div>
<<<<<<< HEAD
***REMOVED***
    <p>Define field dtypes (example: set field 0 to unsigned int and field 4 to float):</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">set_field_types</span><span class="p">({</span><span class="mi">0</span><span class="p">:</span> <span class="s1">&#39;u4&#39;</span><span class="p">,</span> <span class="mi">4</span><span class="p">:</span><span class="s1">&#39;f4&#39;</span><span class="p">})</span>
    </pre></div>
***REMOVED***
=======
    </div>
    <p>Define field dtypes (example: set field 0 to unsigned int and field 4 to float):</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">set_field_types</span><span class="p">({</span><span class="mi">0</span><span class="p">:</span> <span class="s1">&#39;u4&#39;</span><span class="p">,</span> <span class="mi">4</span><span class="p">:</span><span class="s1">&#39;f4&#39;</span><span class="p">})</span>
    </pre></div>
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
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
<<<<<<< HEAD
***REMOVED***
***REMOVED***
    <div class="section" id="json-support">
    <h2>***REMOVED***<a class="headerlink" href="#json-support" title="Permalink to this headline">¶</a></h2>
    <p>Text data in JSON format can be parsed by specifying &#8216;json&#8217; for the ***REMOVED***</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.json&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;json&#39;</span><span class="p">)</span>
    </pre></div>
***REMOVED***
    <p>***REMOVED*** NumPy
=======
    </div>
    </div>
    <div class="section" id="json-support">
    <h2>JSON Support<a class="headerlink" href="#json-support" title="Permalink to this headline">¶</a></h2>
    <p>Text data in JSON format can be parsed by specifying &#8216;json&#8217; for the parser argument:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.json&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;json&#39;</span><span class="p">)</span>
    </pre></div>
    </div>
    <p>Currently, each JSON object at the root level is interpreted as a single NumPy
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    record. Each JSON object can be part of an array, or separated by a newline.
    Examples of valid JSON documents that can be parsed by IOPro, with the NumPy
    array result:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># Single JSON object</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="n">StringIO</span><span class="p">(</span><span class="s1">&#39;{&quot;id&quot;:123, &quot;name&quot;:&quot;xxx&quot;}&#39;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;json&#39;</span><span class="p">)[:]</span>
    <span class="go">array([(123L, &#39;xxx&#39;)],</span>
    <span class="go">      dtype=[(&#39;f0&#39;, &#39;u8&#39;), (&#39;f1&#39;, &#39;O&#39;)])</span>
    </pre></div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># Array of two JSON objects</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="n">StringIO</span><span class="p">(</span><span class="s1">&#39;[{&quot;id&quot;:123, &quot;name&quot;:&quot;xxx&quot;}, {&quot;id&quot;:456, &quot;name&quot;:&quot;yyy&quot;}]&#39;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;json&#39;</span><span class="p">)[:]</span>
    <span class="go">array([(123L, &#39;xxx&#39;), (456L, &#39;yyy&#39;)],</span>
    <span class="go">      dtype=[(&#39;f0&#39;, &#39;u8&#39;), (&#39;f1&#39;, &#39;O&#39;)])</span>
    </pre></div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># Two JSON objects separated by newline</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="n">StringIO</span><span class="p">(</span><span class="s1">&#39;{&quot;id&quot;:123, &quot;name&quot;:&quot;xxx&quot;}</span><span class="se">\n</span><span class="s1">{&quot;id&quot;:456, &quot;name&quot;:&quot;yyy&quot;}&#39;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;json&#39;</span><span class="p">)[:]</span>
    <span class="go">array([(123L, &#39;xxx&#39;), (456L, &#39;yyy&#39;)],</span>
    <span class="go">      dtype=[(&#39;f0&#39;, &#39;u8&#39;), (&#39;f1&#39;, &#39;O&#39;)])</span>
    </pre></div>
<<<<<<< HEAD
***REMOVED***
    <p>Future versions of IOPro will have support for selecting specific JSON fields,
    using a query language similar to XPath for XML.</p>
***REMOVED***
=======
    </div>
    <p>Future versions of IOPro will have support for selecting specific JSON fields,
    using a query language similar to XPath for XML.</p>
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    <div class="section" id="advanced-usage">
    <h2>Advanced Usage<a class="headerlink" href="#advanced-usage" title="Permalink to this headline">¶</a></h2>
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
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    <p>overriding default missing and fill values:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">io</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="s1">&#39;1,abc,inf</span><span class="se">\n</span><span class="s1">2,NA,9.9&#39;</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="n">io</span><span class="o">.</span><span class="n">StringIO</span><span class="p">(</span><span class="n">data</span><span class="p">),</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">,</span> <span class="n">field_names</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">set_field_types</span><span class="p">({</span><span class="mi">1</span><span class="p">:</span><span class="s1">&#39;S3&#39;</span><span class="p">,</span> <span class="mi">2</span><span class="p">:</span><span class="s1">&#39;f4&#39;</span><span class="p">})</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Define list of strings for each field that represent missing values</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">set_missing_values</span><span class="p">({</span><span class="mi">1</span><span class="p">:[</span><span class="s1">&#39;NA&#39;</span><span class="p">],</span> <span class="mi">2</span><span class="p">:[</span><span class="s1">&#39;inf&#39;</span><span class="p">]})</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Set fill value for missing values in each field</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">set_fill_values</span><span class="p">({</span><span class="mi">1</span><span class="p">:</span><span class="s1">&#39;xxx&#39;</span><span class="p">,</span> <span class="mi">2</span><span class="p">:</span><span class="mf">999.999</span><span class="p">})</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="p">[:]</span>
    <span class="go">array([(&#39; abc&#39;, 999.9990234375), (&#39;xxx&#39;, 9.899999618530273)],</span>
    <span class="go">          dtype=[(&#39;f0&#39;, &#39;S4&#39;), (&#39;f1&#39;, &#39;&lt;f4&#39;)])</span>
    </pre></div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    <p>creating and saving tuple of index arrays for gzip file, and reloading indices:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.gz&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">,</span> <span class="n">compression</span><span class="o">=</span><span class="s1">&#39;gzip&#39;</span><span class="p">)</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># build index of records and save index to NumPy array</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">create_index</span><span class="p">(</span><span class="s1">&#39;index_file&#39;</span><span class="p">)</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># reload index</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;data.gz&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">,</span> <span class="n">compression</span><span class="o">=</span><span class="s1">&#39;gzip&#39;</span><span class="p">,</span> <span class="n">index_name</span><span class="o">=</span><span class="s1">&#39;index_file&#39;</span><span class="p">)</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Read last record</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
    <span class="go">array([(100, 101, 102)],dtype=[(&#39;f0&#39;, &#39;&lt;u4&#39;), (&#39;f1&#39;, &#39;&lt;u4&#39;), (&#39;f2&#39;, &#39;&lt;u4&#39;)])</span>
    </pre></div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    <p>Use regular expression for finer control of extracting data:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">io</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Define regular expression to extract dollar amount, percentage, and month.</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Each set of parentheses defines a field.</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="s1">&#39;$2.56, 50%, September 20 1978</span><span class="se">\n</span><span class="s1">$1.23, 23%, April 5 1981&#39;</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">regex_string</span> <span class="o">=</span> <span class="s1">&#39;([0-9]\.[0-9][0-9]+)\,\s ([0-9]+)\%\,\s ([A-Za-z]+)&#39;</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="n">io</span><span class="o">.</span><span class="n">StringIO</span><span class="p">(</span><span class="n">data</span><span class="p">),</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;regex&#39;</span><span class="p">,</span> <span class="n">regex_string</span><span class="o">=</span><span class="n">regex_string</span><span class="p">,</span> <span class="n">field_names</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">infer_types</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># set dtype of field to float</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="o">.</span><span class="n">set_field_types</span><span class="p">({</span><span class="mi">0</span><span class="p">:</span><span class="s1">&#39;f4&#39;</span><span class="p">,</span> <span class="mi">1</span><span class="p">:</span><span class="s1">&#39;u4&#39;</span><span class="p">,</span> <span class="mi">2</span><span class="p">:</span><span class="s1">&#39;S10&#39;</span><span class="p">})</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span><span class="p">[:]</span>
    <span class="go">array([(2.56, 50L, &#39;September&#39;), (1.23, 23L, &#39;April&#39;)],</span>
    <span class="go">    dtype=[(&#39;f0&#39;, &#39;&lt;f8&#39;), (&#39;f1&#39;, &#39;&lt;u8&#39;), (&#39;f2&#39;, &#39;S9&#39;)])</span>
    </pre></div>
<<<<<<< HEAD
***REMOVED***
***REMOVED***
=======
    </div>
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
