-------------
iopro.loadtxt
-------------

.. raw:: html

    <p>Load data from a text file.</p>
    <p>Each row in the text file must have the same number of values.</p>
    <div class="section" id="parameters">
    <h2>Parameters<a class="headerlink" href="#parameters" title="Permalink to this headline">¶</a></h2>
    <dl class="docutils">
    <dt>fname</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">file or str</span><dd>File, filename, or generator to read.  If the filename extension is
    <code class="docutils literal"><span class="pre">.gz</span></code> or <code class="docutils literal"><span class="pre">.bz2</span></code>, the file is first decompressed. Note that
    generators should return byte strings for Python 3k.</dd>
    <dt>dtype</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">data-type, optional</span><dd>Data-type of the resulting array; default: float.  If this is a
    record data-type, the resulting array will be 1-dimensional, and
    each row will be interpreted as an element of the array.  In this
    case, the number of columns used must match the number of fields in
    the data-type.</dd>
    <dt>comments</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">str, optional</span><dd>The character used to indicate the start of a comment;
    default: &#8216;#&#8217;.</dd>
    <dt>delimiter</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">str, optional</span><dd>The string used to separate values.  By default, this is any
    whitespace.</dd>
    <dt>converters</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">dict, optional</span><dd>A dictionary mapping column number to a function that will convert
    that column to a float.  E.g., if column 0 is a date string:
    <code class="docutils literal"><span class="pre">converters</span> <span class="pre">=</span> <span class="pre">{0:</span> <span class="pre">datestr2num}</span></code>.  Converters can also be used to
    provide a default value for missing data (but see also <cite>iopro.genfromtxt</cite>):
    <code class="docutils literal"><span class="pre">converters</span> <span class="pre">=</span> <span class="pre">{3:</span> <span class="pre">lambda</span> <span class="pre">s:</span> <span class="pre">float(s.strip()</span> <span class="pre">or</span> <span class="pre">0)}</span></code>.  Default: None.</dd>
    <dt>skiprows</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">int, optional</span><dd>Skip the first <cite>skiprows</cite> lines; default: 0.</dd>
    <dt>usecols</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">sequence, optional</span><dd>Which columns to read, with 0 being the first.  For example,
    <code class="docutils literal"><span class="pre">usecols</span> <span class="pre">=</span> <span class="pre">(1,4,5)</span></code> will extract the 2nd, 5th and 6th columns.
    The default, None, results in all columns being read.</dd>
    <dt>unpack</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">bool, optional</span><dd>If True, the returned array is transposed, so that arguments may be
    unpacked using <code class="docutils literal"><span class="pre">x,</span> <span class="pre">y,</span> <span class="pre">z</span> <span class="pre">=</span> <span class="pre">iopro.loadtxt(...)</span></code>.  When used with a record
    data-type, arrays are returned for each field.  Default is False.</dd>
    <dt>ndmin</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">int, optional</span><dd>The returned array will have at least <cite>ndmin</cite> dimensions.
    Otherwise mono-dimensional axes will be squeezed.
    Legal values: 0 (default), 1 or 2.
    .. versionadded:: 1.6.0</dd>
    </dl>
    </div>
    <div class="section" id="returns">
    <h2>Returns<a class="headerlink" href="#returns" title="Permalink to this headline">¶</a></h2>
    <dl class="docutils">
    <dt>out</dt>
     <span class="classifier-delimiter">:</span> <span class="classifier">ndarray</span><dd>Data read from the text file.</dd>
    </dl>
    </div>
    <div class="section" id="see-also">
    <h2>See Also<a class="headerlink" href="#see-also" title="Permalink to this headline">¶</a></h2>
    <p>iopro.genfromtxt : Load data with missing values handled as specified.</p>
    </div>
    <div class="section" id="examples">
    <h2>Examples<a class="headerlink" href="#examples" title="Permalink to this headline">¶</a></h2>
    <dl class="docutils">
    <dt>simple parse of StringIO object data</dt>
    <dd><div class="first last highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">io</span> <span class="k">import</span> <span class="n">StringIO</span>   <span class="c1"># StringIO behaves like a file object</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">c</span> <span class="o">=</span> <span class="n">StringIO</span><span class="p">(</span><span class="s2">&quot;0 1</span><span class="se">\\</span><span class="s2">n2 3&quot;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">iopro</span><span class="o">.</span><span class="n">loadtxt</span><span class="p">(</span><span class="n">c</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span><span class="p">([[</span> <span class="mf">0.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">],</span>
    <span class="go">       [ 2.,  3.]])</span>
    </pre></div>
    </div>
    </dd>
    <dt>set dtype of output array</dt>
    <dd><div class="first last highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">d</span> <span class="o">=</span> <span class="n">StringIO</span><span class="p">(</span><span class="s2">&quot;M 21 72</span><span class="se">\\</span><span class="s2">nF 35 58&quot;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">iopro</span><span class="o">.</span><span class="n">loadtxt</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="p">{</span><span class="s1">&#39;names&#39;</span><span class="p">:</span> <span class="p">(</span><span class="s1">&#39;gender&#39;</span><span class="p">,</span> <span class="s1">&#39;age&#39;</span><span class="p">,</span> <span class="s1">&#39;weight&#39;</span><span class="p">),</span>
    <span class="gp">... </span>                     <span class="s1">&#39;formats&#39;</span><span class="p">:</span> <span class="p">(</span><span class="s1">&#39;S1&#39;</span><span class="p">,</span> <span class="s1">&#39;i4&#39;</span><span class="p">,</span> <span class="s1">&#39;f4&#39;</span><span class="p">)})</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span><span class="p">([(</span><span class="s1">&#39;M&#39;</span><span class="p">,</span> <span class="mi">21</span><span class="p">,</span> <span class="mf">72.0</span><span class="p">),</span> <span class="p">(</span><span class="s1">&#39;F&#39;</span><span class="p">,</span> <span class="mi">35</span><span class="p">,</span> <span class="mf">58.0</span><span class="p">)],</span>
    <span class="go">      dtype=[(&#39;gender&#39;, &#39;|S1&#39;), (&#39;age&#39;, &#39;&lt;i4&#39;), (&#39;weight&#39;, &#39;&lt;f4&#39;)])</span>
    </pre></div>
    </div>
    </dd>
    <dt>set delimiter and columns to parse</dt>
    <dd><div class="first last highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">c</span> <span class="o">=</span> <span class="n">StringIO</span><span class="p">(</span><span class="s2">&quot;1,0,2</span><span class="se">\\</span><span class="s2">n3,0,4&quot;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">x</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">loadtxt</span><span class="p">(</span><span class="n">c</span><span class="p">,</span> <span class="n">delimiter</span><span class="o">=</span><span class="s1">&#39;,&#39;</span><span class="p">,</span> <span class="n">usecols</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span><span class="p">),</span> <span class="n">unpack</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">x</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span><span class="p">([</span> <span class="mf">1.</span><span class="p">,</span>  <span class="mf">3.</span><span class="p">])</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">y</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span><span class="p">([</span> <span class="mf">2.</span><span class="p">,</span>  <span class="mf">4.</span><span class="p">])</span>
    </pre></div>
    </div>
    </dd>
    </dl>
    </div>
