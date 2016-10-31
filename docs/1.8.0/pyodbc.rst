------------
iopro.pyodbc
------------

.. raw:: html

    <p>This project is an enhancement of the Python database module for ODBC
    that implements the Python DB API 2.0 specification.  You can see the
    original project here:</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">homepage:</th><td class="field-body"><a class="reference external" href="http://code.google.com/p/pyodbc">http://code.google.com/p/pyodbc</a></td>
    </tr>
    <tr class="field-even field"><th class="field-name">source:</th><td class="field-body"><a class="reference external" href="http://github.com/mkleehammer/pyodbc">http://github.com/mkleehammer/pyodbc</a></td>
    </tr>
    <tr class="field-odd field"><th class="field-name">source:</th><td class="field-body"><a class="reference external" href="http://code.google.com/p/pyodbc/source/list">http://code.google.com/p/pyodbc/source/list</a></td>
    </tr>
    </tbody>
    </table>
    <p>The enhancements are documented in this file.  For general info about
    the pyodbc package, please refer to the original project
    documentation.</p>
    <p>This module enhancement requires:</p>
    <ul class="simple">
    <li>Python 2.4 or greater</li>
    <li>ODBC 3.0 or greater</li>
    <li>NumPy 1.5 or greater (1.7 is required for datetime64 support)</li>
    </ul>
    <p>The enhancements in this module consist mainly in the addition of some
    new methods for fetching the data after a query and put it in a
    variety of NumPy containers.</p>
    <p>Using NumPy as data containers instead of the classical list of tuples
    has a couple of advantages:</p>
    <p>1) The NumPy container is much more compact, and hence, it
    requires much less memory, than the original approach.</p>
    <p>2) As a NumPy container can hold arbitrarily large arrays, it requires
    much less object creation than the original approach (one Python
    object per datum retrieved).</p>
    <p>This means that this enhancements will allow to fetch data out of
    relational databases in a much faster way, while consuming
    significantly less resources.</p>
    <div class="section" id="api-additions">
    <h2>API additions<a class="headerlink" href="#api-additions" title="Permalink to this headline">¶</a></h2>
    <div class="section" id="variables">
    <h3>Variables<a class="headerlink" href="#variables" title="Permalink to this headline">¶</a></h3>
    <ul class="simple">
    <li><cite>pyodbc.npversion</cite>  The version for the NumPy additions</li>
    </ul>
    </div>
    <div class="section" id="methods">
    <h3>Methods<a class="headerlink" href="#methods" title="Permalink to this headline">¶</a></h3>
    <p><strong>Cursor.fetchdictarray</strong> (size=cursor.arraysize)</p>
    <p>This is similar to the original <cite>Cursor.fetchmany(size)</cite>, but the data
    is returned in a dictionary where the keys are the names of the
    columns and the values are NumPy containers.</p>
    <p>For example, it a SELECT is returning 3 columns with names &#8216;a&#8217;, &#8216;b&#8217;
    and &#8216;c&#8217; and types <cite>varchar(10)</cite>, <cite>integer</cite> and <cite>timestamp</cite>, the
    returned object will be something similar to:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="p">{</span><span class="s1">&#39;a&#39;</span><span class="p">:</span> <span class="n">array</span><span class="p">([</span><span class="o">...</span><span class="p">],</span> <span class="n">dtype</span><span class="o">=</span><span class="s1">&#39;S11&#39;</span><span class="p">),</span>
     <span class="s1">&#39;b&#39;</span><span class="p">:</span> <span class="n">array</span><span class="p">([</span><span class="o">...</span><span class="p">],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">int32</span><span class="p">),</span>
     <span class="s1">&#39;c&#39;</span><span class="p">:</span> <span class="n">array</span><span class="p">([</span><span class="o">...</span><span class="p">],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">datetime64</span><span class="p">[</span><span class="n">us</span><span class="p">])}</span>
    </pre></div>
    </div>
    <p>Note that the <cite>varchar(10)</cite> type is translated automatically to a
    string type of 11 elements (&#8216;S11&#8217;).  This is because the ODBC driver
    needs one additional space to put the trailing &#8216;0&#8217; in strings, and
    NumPy needs to provide the room for this.</p>
    <p>Also, it is important to stress that all the <cite>timestamp</cite> types are
    translated into a NumPy <cite>datetime64</cite> type with a resolution of
    microseconds by default.</p>
    <p><strong>Cursor.fetchsarray</strong> (size=cursor.arraysize)</p>
    <p>This is similar to the original <cite>Cursor.fetchmany(size)</cite>, but the data
    is returned in a NumPy structured array, where the name and type of
    the fields matches to those resulting from the SELECT.</p>
    <p>Here it is an example of the output for the SELECT above:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="n">array</span><span class="p">([(</span><span class="o">...</span><span class="p">),</span>
           <span class="p">(</span><span class="o">...</span><span class="p">)],</span>
          <span class="n">dtype</span><span class="o">=</span><span class="p">[(</span><span class="s1">&#39;a&#39;</span><span class="p">,</span> <span class="s1">&#39;|S11&#39;</span><span class="p">),</span> <span class="p">(</span><span class="s1">&#39;b&#39;</span><span class="p">,</span> <span class="s1">&#39;&lt;i4&#39;</span><span class="p">),</span> <span class="p">(</span><span class="s1">&#39;c&#39;</span><span class="p">,</span> <span class="p">(</span><span class="s1">&#39;&lt;M8[us]&#39;</span><span class="p">,</span> <span class="p">{}))])</span>
    </pre></div>
    </div>
    <p>Note that, due to efficiency considerations, this method is calling the
    <cite>fetchdictarray()</cite> behind the scenes, and then doing a conversion to
    get an structured array.  So, in general, this is a bit slower than
    its <cite>fetchdictarray()</cite> counterpart.</p>
    </div>
    </div>
    <div class="section" id="data-types-supported">
    <h2>Data types supported<a class="headerlink" href="#data-types-supported" title="Permalink to this headline">¶</a></h2>
    <p>The new methods listed above have support for a subset of the standard
    ODBC.  In particular:</p>
    <ul class="simple">
    <li>String support (SQL_VARCHAR) is supported.</li>
    <li>Numerical types, be them integers or floats (single and double
    precision) are fully supported.  Here it is the complete list:
    SQL_INTEGER, SQL_TINYINT, SQL_SMALLINT, SQL_FLOAT and SQL_DOUBLE.</li>
    <li>Dates, times, and timestamps are mapped to the <cite>datetime64</cite> and
    <cite>timedelta</cite> NumPy types.  The list of supported data types are:
    SQL_DATE, SQL_TIME and SQL_TIMESTAMP,</li>
    <li>Binary data is not supported yet.</li>
    <li>Unicode strings are not supported yet.</li>
    </ul>
    </div>
    <div class="section" id="null-values">
    <h2>NULL values<a class="headerlink" href="#null-values" title="Permalink to this headline">¶</a></h2>
    <p>As there is not (yet) a definitive support for missing values (NA) in
    NumPy, this module represents NA data as particular values depending
    on the data type.  Here it is the current table of the particular
    values:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="n">int8</span><span class="p">:</span> <span class="o">-</span><span class="mi">128</span> <span class="p">(</span><span class="o">-</span><span class="mi">2</span><span class="o">**</span><span class="mi">7</span><span class="p">)</span>
    <span class="n">uint8</span><span class="p">:</span> <span class="mi">255</span> <span class="p">(</span><span class="mi">2</span><span class="o">**</span><span class="mi">8</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span>
    <span class="n">int16</span><span class="p">:</span> <span class="o">-</span><span class="mi">32768</span> <span class="p">(</span><span class="o">-</span><span class="mi">2</span><span class="o">**</span><span class="mi">15</span><span class="p">)</span>
    <span class="n">uint16</span><span class="p">:</span> <span class="mi">65535</span> <span class="p">(</span><span class="mi">2</span><span class="o">**</span><span class="mi">16</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span>
    <span class="n">int32</span><span class="p">:</span> <span class="o">-</span><span class="mi">2147483648</span> <span class="p">(</span><span class="o">-</span><span class="mi">2</span><span class="o">**</span><span class="mi">31</span><span class="p">)</span>
    <span class="n">uint32</span><span class="p">:</span> <span class="mi">4294967295</span> <span class="p">(</span><span class="mi">2</span><span class="o">**</span><span class="mi">32</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span>
    <span class="n">int64</span><span class="p">:</span> <span class="o">-</span><span class="mi">9223372036854775808</span> <span class="p">(</span><span class="o">-</span><span class="mi">2</span><span class="o">**</span><span class="mi">63</span><span class="p">)</span>
    <span class="n">uint64</span><span class="p">:</span> <span class="mi">18446744073709551615</span> <span class="p">(</span><span class="mi">2</span><span class="o">**</span><span class="mi">64</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span>
    <span class="n">float32</span><span class="p">:</span> <span class="n">NaN</span>
    <span class="n">float64</span><span class="p">:</span> <span class="n">NaN</span>
    <span class="n">datetime64</span><span class="p">:</span> <span class="n">NaT</span>
    <span class="n">timedelta64</span><span class="p">:</span> <span class="n">NaT</span> <span class="p">(</span><span class="ow">or</span> <span class="o">-</span><span class="mi">2</span><span class="o">**</span><span class="mi">63</span><span class="p">)</span>
    <span class="n">string</span><span class="p">:</span> <span class="s1">&#39;NA&#39;</span>
    </pre></div>
    </div>
    </div>
    <div class="section" id="improvements-for-1-1-release">
    <h2>Improvements for 1.1 release<a class="headerlink" href="#improvements-for-1-1-release" title="Permalink to this headline">¶</a></h2>
    <ul class="simple">
    <li>The rowcount is not trusted anymore for the <cite>fetchdict()</cite> and
    <cite>fetchsarray()</cite> methods.  Now the NumPy containers are built
    incrementally, using realloc for a better use of resources.</li>
    <li>The Python interpreter does not exit anymore when fetching an exotic
    datatype not supported by NumPy.</li>
    <li>The docsctrings for <cite>fetchdict()</cite> and <cite>fetchsarray()</cite> have been improved.</li>
    </ul>
    </div>
