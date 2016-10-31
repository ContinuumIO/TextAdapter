----------------
iopro.genfromtxt
----------------

.. raw:: html

    <p>Load data from a text file, with missing values handled as specified.</p>
    <p>Each line past the first <cite>skip_header</cite> lines is split at the <cite>delimiter</cite>
    character, and characters following the <cite>comments</cite> character are discarded.</p>
    <div class="section" id="parameters">
    <h2>Parameters<a class="headerlink" href="#parameters" title="Permalink to this headline">¶</a></h2>
    <dl class="docutils">
    <dt>fname</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">file or str</span><dd>File, filename, or generator to read.  If the filename extension is
    <cite>.gz</cite> or <cite>.bz2</cite>, the file is first decompressed. Note that
    generators must return byte strings in Python 3k.</dd>
    <dt>dtype</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">dtype, optional</span><dd>Data type of the resulting array.
    If None, the dtypes will be determined by the contents of each
    column, individually.</dd>
    <dt>comments</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">str, optional</span><dd>The character used to indicate the start of a comment.
    All the characters occurring on a line after a comment are discarded</dd>
    <dt>delimiter</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">str, int, or sequence, optional</span><dd>The string used to separate values.  By default, any consecutive
    whitespaces act as delimiter.  An integer or sequence of integers
    can also be provided as width(s) of each field.</dd>
    <dt>skip_header</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">int, optional</span><dd>The numbers of lines to skip at the beginning of the file.</dd>
    <dt>skip_footer</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">int, optional</span><dd>The numbers of lines to skip at the end of the file</dd>
    <dt>converters</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">variable, optional</span><dd>The set of functions that convert the data of a column to a value.
    The converters can also be used to provide a default value
    for missing data: <code class="docutils literal"><span class="pre">converters</span> <span class="pre">=</span> <span class="pre">{3:</span> <span class="pre">lambda</span> <span class="pre">s:</span> <span class="pre">float(s</span> <span class="pre">or</span> <span class="pre">0)}</span></code>.</dd>
    <dt>missing_values</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">variable, optional</span><dd>The set of strings corresponding to missing data.</dd>
    <dt>filling_values</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">variable, optional</span><dd>The set of values to be used as default when the data are missing.</dd>
    <dt>usecols</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">sequence, optional</span><dd>Which columns to read, with 0 being the first.  For example,
    <code class="docutils literal"><span class="pre">usecols</span> <span class="pre">=</span> <span class="pre">(1,</span> <span class="pre">4,</span> <span class="pre">5)</span></code> will extract the 2nd, 5th and 6th columns.</dd>
    <dt>names</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">{None, True, str, sequence}, optional</span><dd>If <cite>names</cite> is True, the field names are read from the first valid line
    after the first <cite>skip_header</cite> lines.
    If <cite>names</cite> is a sequence or a single-string of comma-separated names,
    the names will be used to define the field names in a structured dtype.
    If <cite>names</cite> is None, the names of the dtype fields will be used, if any.</dd>
    <dt>excludelist</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">sequence, optional</span><dd>A list of names to exclude. This list is appended to the default list
    [&#8216;return&#8217;,&#8217;file&#8217;,&#8217;print&#8217;]. Excluded names are appended an underscore:
    for example, <cite>file</cite> would become <cite>file_</cite>.</dd>
    <dt>deletechars</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">str, optional</span><dd>A string combining invalid characters that must be deleted from the
    names.</dd>
    <dt>defaultfmt</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">str, optional</span><dd>A format used to define default field names, such as &#8220;f%i&#8221; or &#8220;f_%02i&#8221;.</dd>
    <dt>autostrip</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">bool, optional</span><dd>Whether to automatically strip white spaces from the variables.</dd>
    <dt>replace_space</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">char, optional</span><dd>Character(s) used in replacement of white spaces in the variables
    names. By default, use a &#8216;_&#8217;.</dd>
    <dt>case_sensitive</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">{True, False, &#8216;upper&#8217;, &#8216;lower&#8217;}, optional</span><dd>If True, field names are case sensitive.
    If False or &#8216;upper&#8217;, field names are converted to upper case.
    If &#8216;lower&#8217;, field names are converted to lower case.</dd>
    <dt>unpack</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">bool, optional</span><dd>If True, the returned array is transposed, so that arguments may be
    unpacked using <code class="docutils literal"><span class="pre">x,</span> <span class="pre">y,</span> <span class="pre">z</span> <span class="pre">=</span> <span class="pre">loadtxt(...)</span></code></dd>
    <dt>usemask</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">bool, optional</span><dd>If True, return a masked array.
    If False, return a regular array.</dd>
    <dt>invalid_raise</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">bool, optional</span><dd>If True, an exception is raised if an inconsistency is detected in the
    number of columns.
    If False, a warning is emitted and the offending lines are skipped.</dd>
    </dl>
    </div>
    <div class="section" id="returns">
    <h2>Returns<a class="headerlink" href="#returns" title="Permalink to this headline">¶</a></h2>
    <dl class="docutils">
    <dt>out</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">ndarray</span><dd>Data read from the text file. If <cite>usemask</cite> is True, this is a
    masked array.</dd>
    </dl>
    </div>
    <div class="section" id="see-also">
    <h2>See Also<a class="headerlink" href="#see-also" title="Permalink to this headline">¶</a></h2>
    <p>iopro.loadtxt : equivalent function when no data is missing.</p>
    </div>
    <div class="section" id="notes">
    <h2>Notes<a class="headerlink" href="#notes" title="Permalink to this headline">¶</a></h2>
    <ul class="simple">
    <li>When spaces are used as delimiters, or when no delimiter has been given
    as input, there should not be any missing data between two fields.</li>
    <li>When the variables are named (either by a flexible dtype or with <cite>names</cite>,
    there must not be any header in the file (else a ValueError
    exception is raised).</li>
    <li>Individual values are not stripped of spaces by default.
    When using a custom converter, make sure the function does remove spaces.</li>
    </ul>
    </div>
    <div class="section" id="examples">
    <h2>Examples<a class="headerlink" href="#examples" title="Permalink to this headline">¶</a></h2>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">io</span> <span class="k">import</span> <span class="n">StringIO</span>
    </pre></div>
    </div>
    <p>Comma delimited file with mixed dtype</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">StringIO</span><span class="p">(</span><span class="s2">&quot;1,1.3,abcde&quot;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">genfromtxt</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="p">[(</span><span class="s1">&#39;myint&#39;</span><span class="p">,</span><span class="s1">&#39;i8&#39;</span><span class="p">),(</span><span class="s1">&#39;myfloat&#39;</span><span class="p">,</span><span class="s1">&#39;f8&#39;</span><span class="p">),</span>
    <span class="gp">... </span><span class="p">(</span><span class="s1">&#39;mystring&#39;</span><span class="p">,</span><span class="s1">&#39;S5&#39;</span><span class="p">)],</span> <span class="n">delimiter</span><span class="o">=</span><span class="s2">&quot;,&quot;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span>
    <span class="go">array((1, 1.3, &#39;abcde&#39;),</span>
    <span class="go">      dtype=[(&#39;myint&#39;, &#39;&lt;i8&#39;), (&#39;myfloat&#39;, &#39;&lt;f8&#39;), (&#39;mystring&#39;, &#39;|S5&#39;)])</span>
    </pre></div>
    </div>
    <p>Using dtype = None</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span><span class="o">.</span><span class="n">seek</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span> <span class="c1"># needed for StringIO example only</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">genfromtxt</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>
    <span class="gp">... </span><span class="n">names</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;myint&#39;</span><span class="p">,</span><span class="s1">&#39;myfloat&#39;</span><span class="p">,</span><span class="s1">&#39;mystring&#39;</span><span class="p">],</span> <span class="n">delimiter</span><span class="o">=</span><span class="s2">&quot;,&quot;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span>
    <span class="go">array((1, 1.3, &#39;abcde&#39;),</span>
    <span class="go">      dtype=[(&#39;myint&#39;, &#39;&lt;i8&#39;), (&#39;myfloat&#39;, &#39;&lt;f8&#39;), (&#39;mystring&#39;, &#39;|S5&#39;)])</span>
    </pre></div>
    </div>
    <p>Specifying dtype and names</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span><span class="o">.</span><span class="n">seek</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">genfromtxt</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="s2">&quot;i8,f8,S5&quot;</span><span class="p">,</span>
    <span class="gp">... </span><span class="n">names</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;myint&#39;</span><span class="p">,</span><span class="s1">&#39;myfloat&#39;</span><span class="p">,</span><span class="s1">&#39;mystring&#39;</span><span class="p">],</span> <span class="n">delimiter</span><span class="o">=</span><span class="s2">&quot;,&quot;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span>
    <span class="go">array((1, 1.3, &#39;abcde&#39;),</span>
    <span class="go">      dtype=[(&#39;myint&#39;, &#39;&lt;i8&#39;), (&#39;myfloat&#39;, &#39;&lt;f8&#39;), (&#39;mystring&#39;, &#39;|S5&#39;)])</span>
    </pre></div>
    </div>
    <p>An example with fixed-width columns</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">StringIO</span><span class="p">(</span><span class="s2">&quot;11.3abcde&quot;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">genfromtxt</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">names</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;intvar&#39;</span><span class="p">,</span><span class="s1">&#39;fltvar&#39;</span><span class="p">,</span><span class="s1">&#39;strvar&#39;</span><span class="p">],</span>
    <span class="gp">... </span>    <span class="n">delimiter</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="mi">5</span><span class="p">])</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">data</span>
    <span class="go">array((1, 1.3, &#39;abcde&#39;),</span>
    <span class="go">      dtype=[(&#39;intvar&#39;, &#39;&lt;i8&#39;), (&#39;fltvar&#39;, &#39;&lt;f8&#39;), (&#39;strvar&#39;, &#39;|S5&#39;)])</span>
    </pre></div>
    </div>
