-----
IOPro
-----

.. raw:: html

    <p>IOPro loads NumPy arrays (and Pandas DataFrames) directly from files, SQL
    databases, and NoSQL stores, without creating millions of temporary,
    intermediate Python objects, or requiring expensive array resizing
    operations. It provides a drop-in replacement for the NumPy functions
    loadtxt() and genfromtxt(), but drastically improves performance and
    reduces the memory overhead.</p>
    <p>IOPro is included with <a class="reference external" href="https://www.continuum.io/content/anaconda-subscriptions">Anaconda Workgroup and Anaconda Enterprise subscriptions</a>.</p>
    <p>To start a 30-day free trial just download and install the IOPro package.</p>
    <p>If you already have <a class="reference external" href="http://continuum.io/downloads.html">Anaconda</a>
    (free Python distribution) installed:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="n">conda</span> <span class="n">update</span> <span class="n">conda</span>
    <span class="n">conda</span> <span class="n">install</span> <span class="n">iopro</span>
    </pre></div>
    </div>
    <p>If you do not have Anaconda installed, you can download it
    <a class="reference external" href="http://continuum.io/downloads.html">here</a>.</p>
    <p>IOPro can also be installed into your own (non-Anaconda) Python environment. For more information about IOPro please contact <a class="reference external" href="mailto:sales&#37;&#52;&#48;continuum&#46;io">sales<span>&#64;</span>continuum<span>&#46;</span>io</a>.</p>
    <div class="section" id="getting-started">
    <h2>Getting Started<a class="headerlink" href="#getting-started" title="Permalink to this headline">¶</a></h2>
    <p>Start by attaching to a data source (in this case, a local csv file):</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">text_adapter</span><span class="p">(</span><span class="s1">&#39;table.csv&#39;</span><span class="p">,</span> <span class="n">parser</span><span class="o">=</span><span class="s1">&#39;csv&#39;</span><span class="p">)</span>
    </pre></div>
    </div>
    <p>We can specify the data types for values in the columns of the csv
    file being read though here we will instead rely upon the ability of
    IOPro&#8217;s TextAdapter to auto-discover the data types used.</p>
    <p>We ask IOPro&#8217;s TextAdapter to parse text and return records in NumPy arrays
    from selected portions of the csv file using slicing notation:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># read first ten records</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">10</span><span class="p">]</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># read last five records</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[</span><span class="o">-</span><span class="mi">5</span><span class="p">:]</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># read every other record</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[::</span><span class="mi">2</span><span class="p">]</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># read first and second fields only</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">]][:]</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># read fields named &#39;f2&#39; and &#39;f4&#39; only</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[[</span><span class="s1">&#39;f2&#39;</span><span class="p">,</span><span class="s1">&#39;f4&#39;</span><span class="p">]][:]</span>
    </pre></div>
    </div>
    </div>
    <div class="section" id="user-guide">
    <h2>User Guide<a class="headerlink" href="#user-guide" title="Permalink to this headline">¶</a></h2>
    <div class="toctree-wrapper compound">
    <ul>
    <li class="toctree-l1"><a class="reference internal" href="install.html">Installation</a></li>
    <li class="toctree-l1"><a class="reference internal" href="textadapter_firststeps.html">TextAdapter First Steps</a></li>
    <li class="toctree-l1"><a class="reference internal" href="textadapter_advanced.html">Advanced TextAdapter</a></li>
    <li class="toctree-l1"><a class="reference internal" href="pyodbc_firststeps.html">iopro.pyodbc First Steps</a></li>
    <li class="toctree-l1"><a class="reference internal" href="pyodbc_enhancedcapabilities.html">iopro.pyodbc Enhanced Capabilities</a></li>
    <li class="toctree-l1"><a class="reference internal" href="pyodbc_cancel.html">iopro.pyodbc Cancelling Queries</a></li>
    </ul>
    </div>
    </div>
    <div class="section" id="reference-guide">
    <h2>Reference Guide<a class="headerlink" href="#reference-guide" title="Permalink to this headline">¶</a></h2>
    <div class="toctree-wrapper compound">
    <ul>
    <li class="toctree-l1"><a class="reference internal" href="TextAdapter.html">TextAdapter</a></li>
    <li class="toctree-l1"><a class="reference internal" href="pyodbc.html">iopro.pyodbc</a></li>
    <li class="toctree-l1"><a class="reference internal" href="MongoAdapter.html">MongoAdapter</a></li>
    <li class="toctree-l1"><a class="reference internal" href="loadtxt.html">iopro.loadtxt</a></li>
    <li class="toctree-l1"><a class="reference internal" href="genfromtxt.html">iopro.genfromtxt</a></li>
    </ul>
    </div>
    </div>
    <div class="section" id="requirements">
    <h2>Requirements<a class="headerlink" href="#requirements" title="Permalink to this headline">¶</a></h2>
    <ul class="simple">
    <li>python 2.6, 2.7, or 3.3+</li>
    <li>numpy (&gt;= 1.6)</li>
    </ul>
    <p>Python modules (optional):</p>
    <ul class="simple">
    <li>boto (&gt;= 2.8)</li>
    <li>numba (&gt;= 0.8)</li>
    </ul>
    </div>
    <div class="section" id="release-notes">
    <h2>Release Notes<a class="headerlink" href="#release-notes" title="Permalink to this headline">¶</a></h2>
    <div class="toctree-wrapper compound">
    <ul>
    <li class="toctree-l1"><a class="reference internal" href="release-notes.html">IOPro Release Notes</a></li>
    </ul>
    </div>
    </div>
    <div class="section" id="license-agreement">
    <h2>License Agreement<a class="headerlink" href="#license-agreement" title="Permalink to this headline">¶</a></h2>
    <div class="toctree-wrapper compound">
    <ul>
    <li class="toctree-l1"><a class="reference internal" href="eula.html">IOPro END USER LICENSE AGREEMENT</a></li>
    </ul>
    </div>
    </div>

.. toctree::
   :maxdepth: 1
   :hidden:

   install
   textadapter_advanced
   textadapter_firststeps
   pyodbc_firststeps
   pyodbc_enhancedcapabilities
   pyodbc_cancel
   TextAdapter
   pyodbc
   MongoAdapter
   loadtxt
   genfromtxt
   release-notes
   eula
