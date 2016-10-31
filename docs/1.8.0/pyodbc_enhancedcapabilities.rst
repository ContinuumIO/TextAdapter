***REMOVED******REMOVED***
iopro.pyodbc Enhanced Capabilities
***REMOVED******REMOVED***

***REMOVED***

    <div class="section" id="demo-code-showing-the-enhanced-capabilities-of-iopro-pyodbc-submodule">
    <h2>Demo code showing the enhanced capabilities of iopro.pyodbc submodule<a class="headerlink" href="#demo-code-showing-the-enhanced-capabilities-of-iopro-pyodbc-submodule" title="Permalink to this headline">¶</a></h2>
    <p>This demo shows the basic capabilities for the iopro.pyodbc module.  It first will connect with the database of your choice by ODBC, create and fill a new table (market) and then retrieve data with different methods (fetchall(), fetchdictarray() and fetchsarray()).</p>
    <p>Author: Francesc Alted, Continuum Analytics</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro.pyodbc</span> <span class="k">as</span> <span class="nn">pyodbc</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="c1"># Open the database (use the most appropriate for you)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">connect_string</span> <span class="o">=</span> <span class="s1">&#39;DSN=odbcsqlite;DATABASE=market.sqlite&#39;</span>  <span class="c1"># SQLite</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="c1">#connect_string = &#39;Driver={SQL Server};SERVER=MyWinBox;DATABASE=Test;USER=Devel;PWD=XXX&#39;  # SQL Server</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="c1">#connect_string = &#39;DSN=myodbc3;UID=devel;PWD=XXX;DATABASE=test&#39;  # MySQL</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="c1">#connect_string = &#39;DSN=PSQL;UID=devel;PWD=XXX;DATABASE=test&#39;   # PostgreSQL</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">connection</span> <span class="o">=</span> <span class="n">pyodbc</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="n">connect_string</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span> <span class="o">=</span> <span class="n">connection</span><span class="o">.</span><span class="n">cursor</span><span class="p">()</span>
    </pre></div>
***REMOVED***
***REMOVED***
    <div class="section" id="create-the-test-table-optional-if-already-done">
    <h2>Create the test table (optional if already done)<a class="headerlink" href="#create-the-test-table-optional-if-already-done" title="Permalink to this headline">¶</a></h2>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="k">try</span><span class="p">:</span>
    <span class="gp">... </span>    <span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;drop table market&#39;</span><span class="p">)</span>
    <span class="gp">... </span><span class="k">except</span><span class="p">:</span>
    <span class="gp">... </span>    <span class="k">pass</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;create table market (symbol_ varchar(5), open_ float, low_ float, high_ float, close_ float, volume_ int)&#39;</span><span class="p">)</span>
    </pre></div>
***REMOVED***
***REMOVED***
    <div class="section" id="fill-the-test-table-optional-if-already-done">
    <h2>Fill the test table (optional if already done)<a class="headerlink" href="#fill-the-test-table-optional-if-already-done" title="Permalink to this headline">¶</a></h2>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">time</span> <span class="k">import</span> <span class="n">time</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">t0</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">N</span> <span class="o">=</span> <span class="mi">1000</span><span class="o">*</span><span class="mi">1000</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">xrange</span><span class="p">(</span><span class="n">N</span><span class="p">):</span>
    <span class="gp">... </span>    <span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span>
    <span class="gp">... </span>        <span class="s2">&quot;insert into market(symbol_, open_, low_, high_, close_, volume_)&quot;</span>
    <span class="gp">... </span>        <span class="s2">&quot; values (?, ?, ?, ?, ?, ?)&quot;</span><span class="p">,</span>
    <span class="gp">... </span>        <span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">i</span><span class="p">),</span> <span class="nb">float</span><span class="p">(</span><span class="n">i</span><span class="p">),</span> <span class="nb">float</span><span class="p">(</span><span class="mi">2</span><span class="o">*</span><span class="n">i</span><span class="p">),</span> <span class="kc">None</span><span class="p">,</span> <span class="nb">float</span><span class="p">(</span><span class="mi">4</span><span class="o">*</span><span class="n">i</span><span class="p">),</span> <span class="n">i</span><span class="p">))</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s2">&quot;commit&quot;</span><span class="p">)</span>             <span class="c1"># not supported by SQLite</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">t1</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span> <span class="o">-</span> <span class="n">t0</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span> <span class="s2">&quot;Stored </span><span class="si">%d</span><span class="s2"> rows in </span><span class="si">%.3f</span><span class="s2">s&quot;</span> <span class="o">%</span> <span class="p">(</span><span class="n">N</span><span class="p">,</span> <span class="n">t1</span><span class="p">)</span>
    </pre></div>
***REMOVED***
***REMOVED***
    <div class="section" id="do-the-query-in-the-traditional-way">
    <h2>Do the query in the traditional way<a class="headerlink" href="#do-the-query-in-the-traditional-way" title="Permalink to this headline">¶</a></h2>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># Query of the full table using the traditional fetchall</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">query</span> <span class="o">=</span> <span class="s2">&quot;select * from market&quot;</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="n">query</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="o">%</span><span class="n">time</span> <span class="nb">all</span> <span class="o">=</span> <span class="n">cursor</span><span class="o">.</span><span class="n">fetchall</span><span class="p">()</span>
    <span class="go">CPU times: user 5.23 s, sys: 0.56 s, total: 5.79 s</span>
    <span class="go">Wall time: 7.09 s</span>
    </pre></div>
***REMOVED***
***REMOVED***
    <div class="section" id="do-the-query-and-get-a-dictionary-of-numpy-arrays">
    <h2>Do the query and get a dictionary of NumPy arrays<a class="headerlink" href="#do-the-query-and-get-a-dictionary-of-numpy-arrays" title="Permalink to this headline">¶</a></h2>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># Query of the full table using the fetchdictarray (retrieve a dictionary of arrays)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="n">query</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="o">%</span><span class="n">time</span> <span class="n">dictarray</span> <span class="o">=</span> <span class="n">cursor</span><span class="o">.</span><span class="n">fetchdictarray</span><span class="p">()</span>
    <span class="go">CPU times: user 0.92 s, sys: 0.10 s, total: 1.02 s</span>
    <span class="go">Wall time: 1.44 s</span>
    </pre></div>
***REMOVED***
***REMOVED***
    <div class="section" id="peek-into-the-retrieved-data">
    <h2>Peek into the retrieved data<a class="headerlink" href="#peek-into-the-retrieved-data" title="Permalink to this headline">¶</a></h2>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">dictarray</span><span class="o">.</span><span class="n">keys</span><span class="p">()</span>
    <span class="go">[&#39;high_&#39;, &#39;close_&#39;, &#39;open_&#39;, &#39;low_&#39;, &#39;volume_&#39;, &#39;symbol_&#39;]</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">dictarray</span><span class="p">[</span><span class="s1">&#39;high_&#39;</span><span class="p">]</span>
    <span class="go">array([ nan,  nan,  nan, ...,  nan,  nan,  nan])</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">dictarray</span><span class="p">[</span><span class="s1">&#39;symbol_&#39;</span><span class="p">]</span>
    <span class="go">array([&#39;0&#39;, &#39;1&#39;, &#39;2&#39;, ..., &#39;99999&#39;, &#39;99999&#39;, &#39;99999&#39;], dtype=&#39;|S6&#39;)</span>
    </pre></div>
***REMOVED***
***REMOVED***
    <div class="section" id="do-the-query-and-get-a-numpy-structured-array">
    <h2>Do the query and get a NumPy structured array<a class="headerlink" href="#do-the-query-and-get-a-numpy-structured-array" title="Permalink to this headline">¶</a></h2>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># Query of the full table using the fetchsarray (retrieve a structured array)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="n">query</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="o">%</span><span class="n">time</span> <span class="n">sarray</span> <span class="o">=</span> <span class="n">cursor</span><span class="o">.</span><span class="n">fetchsarray</span><span class="p">()</span>
    <span class="go">CPU times: user 1.08 s, sys: 0.11 s, total: 1.20 s</span>
    <span class="go">Wall time: 1.99 s</span>
    </pre></div>
***REMOVED***
***REMOVED***
    <div class="section" id="peek-into-retrieved-data">
    <h2>Peek into retrieved data<a class="headerlink" href="#peek-into-retrieved-data" title="Permalink to this headline">¶</a></h2>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">sarray</span><span class="o">.</span><span class="n">dtype</span>
    <span class="go">dtype([(&#39;symbol_&#39;, &#39;S6&#39;), (&#39;open_&#39;, &#39;&amp;lt;f8&#39;), (&#39;low_&#39;, &#39;&amp;lt;f8&#39;), (&#39;high_&#39;, &#39;&amp;lt;f8&#39;), (&#39;close_&#39;, &#39;&amp;lt;f8&#39;), (&#39;volume_&#39;, &#39;&amp;lt;i4&#39;)])</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">sarray</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">10</span><span class="p">]</span>
    <span class="go">array([(&#39;0&#39;, 0.0, 0.0, nan, 0.0, 0), (&#39;1&#39;, 1.0, 2.0, nan, 4.0, 1),</span>
    <span class="go">       (&#39;2&#39;, 2.0, 4.0, nan, 8.0, 2), (&#39;3&#39;, 3.0, 6.0, nan, 12.0, 3),</span>
    <span class="go">       (&#39;4&#39;, 4.0, 8.0, nan, 16.0, 4), (&#39;5&#39;, 5.0, 10.0, nan, 20.0, 5),</span>
    <span class="go">       (&#39;6&#39;, 6.0, 12.0, nan, 24.0, 6), (&#39;7&#39;, 7.0, 14.0, nan, 28.0, 7),</span>
    <span class="go">       (&#39;8&#39;, 8.0, 16.0, nan, 32.0, 8), (&#39;9&#39;, 9.0, 18.0, nan, 36.0, 9)],</span>
    <span class="go">      dtype=[(&#39;symbol_&#39;, &#39;S6&#39;), (&#39;open_&#39;, &#39;&amp;lt;f8&#39;), (&#39;low_&#39;, &#39;&amp;lt;f8&#39;), (&#39;high_&#39;, &#39;&amp;lt;f8&#39;), (&#39;close_&#39;, &#39;&amp;lt;f8&#39;), (&#39;volume_&#39;, &#39;&amp;lt;i4&#39;)])</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">sarray</span><span class="p">[</span><span class="s1">&#39;symbol_&#39;</span><span class="p">]</span>
    <span class="go">array([&#39;0&#39;, &#39;1&#39;, &#39;2&#39;, ..., &#39;99999&#39;, &#39;99999&#39;, &#39;99999&#39;], dtype=&#39;|S6&#39;)</span>
    </pre></div>
***REMOVED***
***REMOVED***
