=================================
 iopro.pyodbc Cancelling Queries
=================================

***REMOVED***

    <p>Starting with version 1.5, the pyodbc submodule of IOPro makes it
    possible to cancel operations. This is done by exposing the SQLCancel
    ODBC function as a cancel method in the Cursor object.</p>
    <div class="section" id="a-simple-example">
    <h2>A Simple Example<a class="headerlink" href="#a-simple-example" title="Permalink to this headline">¶</a></h2>
    <p>A very simple example would be:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="n">conn</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">pyodbc</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="n">conn_str</span><span class="p">)</span>
    <span class="n">cursor</span> <span class="o">=</span> <span class="n">conn</span><span class="o">.</span><span class="n">cursor</span><span class="p">()</span>
    <span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;SELECT something FROM sample_table&#39;</span><span class="p">)</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">cursor</span><span class="o">.</span><span class="n">fetchone</span><span class="p">()</span>
    <span class="n">cursor</span><span class="o">.</span><span class="n">cancel</span><span class="p">()</span>
    </pre></div>
***REMOVED***
    <p>This is not very interesting, and it doesn&#8217;t add much to the
    functionality of pyodbc.</p>
    <p>What makes the cancel method more interesting is that it is possible
    to cancel running statements that are blocking another thread.</p>
***REMOVED***
    <div class="section" id="a-sample-with-threading">
    <h2>A Sample With Threading<a class="headerlink" href="#a-sample-with-threading" title="Permalink to this headline">¶</a></h2>
    <p>Having access to the cancel method it is possible to stop running
    queries following different criteria. For example, it would be
    possible to execute queries with a time-out. If the time runs out, the
    query gets cancelled.</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">iopro.pyodbc</span>
    <span class="kn">import</span> <span class="nn">time</span>
    <span class="kn">import</span> <span class="nn">threading</span>

    <span class="k">def</span> <span class="nf">query_with_time_out</span><span class="p">(</span><span class="n">conn</span><span class="p">,</span> <span class="n">query</span><span class="p">,</span> <span class="n">timeout</span><span class="p">):</span>
        <span class="k">def</span> <span class="nf">watchdog</span><span class="p">(</span><span class="n">cursor</span><span class="p">,</span> <span class="n">time_out</span><span class="p">):</span>
            <span class="n">time</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="n">wait_time</span><span class="p">)</span>
            <span class="n">cursor</span><span class="o">.</span><span class="n">cancel</span><span class="p">()</span>

        <span class="n">cursor</span> <span class="o">=</span> <span class="n">conn</span><span class="o">.</span><span class="n">cursor</span><span class="p">()</span>

        <span class="n">t</span> <span class="o">=</span> <span class="n">threading</span><span class="o">.</span><span class="n">Thread</span><span class="p">(</span><span class="n">target</span><span class="o">=</span><span class="n">watchdog</span><span class="p">,</span> <span class="n">args</span><span class="o">=</span><span class="p">(</span><span class="n">cursor</span><span class="p">,</span> <span class="n">timeout</span><span class="p">))</span>
        <span class="n">t</span><span class="o">.</span><span class="n">start</span><span class="p">()</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="n">query</span><span class="p">)</span>

            <span class="n">result</span> <span class="o">=</span> <span class="n">cursor</span><span class="o">.</span><span class="n">fetchall</span><span class="p">()</span>
        <span class="k">except</span> <span class="n">iopro</span><span class="o">.</span><span class="n">pyodbc</span><span class="o">.</span><span class="n">Error</span><span class="p">:</span>
            <span class="n">result</span> <span class="o">=</span> <span class="s1">&#39;timed out&#39;</span>

        <span class="k">return</span> <span class="n">result</span>
    </pre></div>
***REMOVED***
    <p>This is just one possibility. As cursor exposes directly the
    SQLCancel, many oportunities open in implementing policies to cancel
    running queries.</p>
***REMOVED***
    <div class="section" id="finishing-notes">
    <h2>Finishing notes<a class="headerlink" href="#finishing-notes" title="Permalink to this headline">¶</a></h2>
    <p>In order for this to work, the underlying ODBC driver must support
    SQLCancel.</p>
    <p>The pyodbc submodule of IOPro releases the Python GIL when it calls
    ODBC, so while queries are being executed other Python threads
    continue to execute while the thread that performed the query is
    blocked. This allows for cancel to be called by another
    thread. Coupled with threading, the cancel method is a very useful
    primitive.</p>
***REMOVED***
