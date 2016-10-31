-------------------------
iopro.pyodbc First Steps
-------------------------

.. raw:: html

    <p>iopro.pyodbc extends pyodbc with methods that allow data to be fetched directly into numpy containers. These functions are faster than regular fetch calls in pyodbc, providing also the convenience of being returned in a container appropriate to fast analysis.</p>
    <p>This notebook is intended to be a tutorial on iopro.pyodbc. Most of the material is applicable to pyodbc (and based on pyodbc tutorials). There will be some examples specific to iopro.pyodbc. When that&#8217;s the case, it will be noted.</p>
    <div class="section" id="concepts">
    <h2>Concepts<a class="headerlink" href="#concepts" title="Permalink to this headline">¶</a></h2>
    <dl class="docutils">
    <dt>In pyodbc there are two main classes to understand:</dt>
    <dd><ul class="first last simple">
    <li>connection</li>
    <li>cursor</li>
    </ul>
    </dd>
    </dl>
    <p>A connection is, as its name says, a connection to a datasource. A datasource is your database. It may be a database handled by a DBMS or just a plain file.
    A cursor allows you to interface with statements. Interaction with queries and other commands is performed through a cursor. A cursor is associated to a connection and commands over a cursor are performed over that connection to the datasource.
    In order to use iopro.pyodbc you must import it:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro.pyodbc</span> <span class="k">as</span> <span class="nn">pyodbc</span>
    </pre></div>
    </div>
    </div>
    <div class="section" id="connection-to-a-datasource">
    <h2>Connection to a datasource<a class="headerlink" href="#connection-to-a-datasource" title="Permalink to this headline">¶</a></h2>
    <p>In order to operate with pyodbc you need to connect to a datasource. Typically this will be a database. This is done by creating a connection object.
    To create a connection object you need a connection string. This string describes the datasource to use as well as some extra parameters. You can learn more about connection strings here.:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">connection_string</span> <span class="o">=</span> <span class="s1">&#39;&#39;&#39;DSN=SQLServerTest;DATABASE=Test&#39;&#39;&#39;</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">connection</span> <span class="o">=</span> <span class="n">pyodbc</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="n">connection_string</span><span class="p">)</span>
    </pre></div>
    </div>
    <p>pyodbc.connect supports a keyword parameter autocommit. This controls the way the connection is handle. The default value (False) means that the commands that modify the database statements need to be committed explicitly. All commands between commits will form a single transaction. If autocommit is enabled every command will be issued and committed.
    It is also possible to change autocommit status after the connection is established.:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">connection</span><span class="o">.</span><span class="n">autocommit</span> <span class="o">=</span> <span class="kc">True</span> <span class="c1">#enable autocommit</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">connection</span><span class="o">.</span><span class="n">autocommit</span> <span class="o">=</span> <span class="kc">False</span> <span class="c1"># disable autocommit</span>
    </pre></div>
    </div>
    <p>When not in autocommit mode, you can end a transaction by either commiting it or rolling it back.:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="n">In</span><span class="p">[</span><span class="mi">6</span><span class="p">]:</span> <span class="n">connection</span><span class="o">.</span><span class="n">commit</span><span class="p">()</span> <span class="c1"># commit the transaction</span>
    <span class="n">In</span><span class="p">[</span><span class="mi">7</span><span class="p">]:</span> <span class="n">connection</span><span class="o">.</span><span class="n">rollback</span><span class="p">()</span> <span class="c1"># rollback the transaction</span>
    </pre></div>
    </div>
    <p>Note that commit/rollback is always performed at the connection level. pyodbc provides a commit/rollback method in the cursor objects, but they will act on the associated connection.</p>
    </div>
    <div class="section" id="working-with-cursors">
    <h2>Working with cursors<a class="headerlink" href="#working-with-cursors" title="Permalink to this headline">¶</a></h2>
    <p>Command execution in pyodbc is handled through cursors. You can create a cursor from a connection using the cursor() method. The first step is creating a cursor:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="n">In</span><span class="p">[</span><span class="mi">8</span><span class="p">]:</span> <span class="n">cursor</span> <span class="o">=</span> <span class="n">connection</span><span class="o">.</span><span class="n">cursor</span><span class="p">()</span>
    </pre></div>
    </div>
    <p>With a cursor created, we can start issuing SQL commands using the execute method.</p>
    </div>
    <div class="section" id="creating-a-sample-table">
    <h2>Creating a sample table<a class="headerlink" href="#creating-a-sample-table" title="Permalink to this headline">¶</a></h2>
    <p>First, create a sample table in the database. The following code will create a sample table with three columns of different types.:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="k">def</span> <span class="nf">create_test_table</span><span class="p">(</span><span class="n">cursor</span><span class="p">):</span>
    <span class="gp">... </span>   <span class="k">try</span><span class="p">:</span>
    <span class="gp">... </span>       <span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;drop table test_table&#39;</span><span class="p">)</span>
    <span class="gp">... </span>   <span class="k">except</span><span class="p">:</span>
    <span class="gp">... </span>       <span class="k">pass</span>
    <span class="gp">... </span>   <span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;&#39;&#39;create table test_table (</span>
    <span class="gp">... </span><span class="s1">                                   name varchar(10),</span>
    <span class="gp">... </span><span class="s1">                                   fval float(24),</span>
    <span class="gp">... </span><span class="s1">                                   ival int)&#39;&#39;&#39;</span><span class="p">)</span>
    <span class="gp">... </span>   <span class="n">cursor</span><span class="o">.</span><span class="n">commit</span><span class="p">()</span>

    <span class="gp">&gt;&gt;&gt; </span><span class="n">create_test_table</span><span class="p">(</span><span class="n">cursor</span><span class="p">)</span>
    </pre></div>
    </div>
    </div>
    <div class="section" id="filling-the-sample-table-with-sample-data">
    <h2>Filling the sample table with sample data<a class="headerlink" href="#filling-the-sample-table-with-sample-data" title="Permalink to this headline">¶</a></h2>
    <p>After creating the table, rows can be inserted by executing insert into the table. Note you can pass parameters by placing a ? into the SQL statement. The parameters will be taken in order for the sequence appears in the next parameter.:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;&#39;&#39;insert into test_table values (?,?,?)&#39;&#39;&#39;</span><span class="p">,</span> <span class="p">(</span><span class="s1">&#39;foo&#39;</span><span class="p">,</span> <span class="mf">3.0</span><span class="p">,</span> <span class="mi">2</span><span class="p">))</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">rowcount</span>
    <span class="go">1</span>
    </pre></div>
    </div>
    <p>Using executemany a sequence of parameters to the SQL statement can be passed and the statement will be executed many times, each time with a different parameter set. This allows us to easily insert several rows into the database so that we have a small test set::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">executemany</span><span class="p">(</span><span class="s1">&#39;&#39;&#39;insert into test_table values (?,?,?)&#39;&#39;&#39;</span><span class="p">,</span> <span class="p">[</span>
    <span class="gp">... </span>                       <span class="p">(</span><span class="s1">&#39;several&#39;</span><span class="p">,</span> <span class="mf">2.1</span><span class="p">,</span> <span class="mi">3</span><span class="p">),</span>
    <span class="gp">... </span>                       <span class="p">(</span><span class="s1">&#39;tuples&#39;</span><span class="p">,</span> <span class="o">-</span><span class="mf">1.0</span><span class="p">,</span> <span class="mi">2</span><span class="p">),</span>
    <span class="gp">... </span>                       <span class="p">(</span><span class="s1">&#39;can&#39;</span><span class="p">,</span> <span class="mf">3.0</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span>
    <span class="gp">... </span>                       <span class="p">(</span><span class="s1">&#39;be&#39;</span><span class="p">,</span> <span class="mf">12.0</span><span class="p">,</span> <span class="o">-</span><span class="mi">3</span><span class="p">),</span>
    <span class="gp">... </span>                       <span class="p">(</span><span class="s1">&#39;inserted&#39;</span><span class="p">,</span> <span class="mf">0.0</span><span class="p">,</span> <span class="o">-</span><span class="mi">2</span><span class="p">),</span>
    <span class="gp">... </span>                       <span class="p">(</span><span class="s1">&#39;at&#39;</span><span class="p">,</span> <span class="mf">33.0</span><span class="p">,</span> <span class="mi">0</span><span class="p">),</span>
    <span class="gp">... </span>                       <span class="p">(</span><span class="s1">&#39;once&#39;</span><span class="p">,</span> <span class="mf">0.0</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span>
    <span class="gp">... </span>                       <span class="p">])</span>
    </pre></div>
    </div>
    <p>Remember that if autocommit is turned off the changes won&#8217;t be visible to any other connection unless we commit.:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">commit</span><span class="p">()</span> <span class="c1"># remember this is a shortcut to connection.commit() method</span>
    </pre></div>
    </div>
    </div>
    <div class="section" id="querying-the-sample-data-from-the-sample-table">
    <h2>Querying the sample data from the sample table<a class="headerlink" href="#querying-the-sample-data-from-the-sample-table" title="Permalink to this headline">¶</a></h2>
    <p>Having populated our sample database, we can retrieve the inserted data by executing select statements::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;&#39;&#39;select * from test_table&#39;&#39;&#39;</span><span class="p">)</span>
    <span class="go">&lt;pyodbc.Cursor at 0x6803510&gt;</span>
    </pre></div>
    </div>
    <p>After calling execute with the select statement we need to retrieve the data. This can be achieved by calling fetch methods in the cursor
    fetchone fetches the next row in the cursor, returning it in a tuple:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">fetchone</span><span class="p">()</span>
    <span class="go">(&#39;foo&#39;, 3.0, 2)</span>
    </pre></div>
    </div>
    <p>fetchmany retrieves several rows at a time in a list of tuples:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">fetchmany</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
    <span class="go">[(&#39;several&#39;, 2.0999999046325684, 3), (&#39;tuples&#39;, -1.0, 2), (&#39;can&#39;, 3.0, 1)]</span>
    </pre></div>
    </div>
    <p>fetchall retrieves all the remaining rows in a list of tuples:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">fetchall</span><span class="p">()</span>
    <span class="go">[(&#39;be&#39;, 12.0, -3), (&#39;inserted&#39;, 0.0, -2), (&#39;at&#39;, 33.0, 0), (&#39;once&#39;, 0.0, 0)]</span>
    </pre></div>
    </div>
    <p>All the calls to any kind of fetch advances the cursor, so the next fetch starts in the row after the last row fetched.
    execute returns the cursor object. This is handy to retrieve the full query by chaining fetchall. This results in a one-liner::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;&#39;&#39;select * from test_table&#39;&#39;&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">fetchall</span><span class="p">()</span>
    <span class="go">[(&#39;foo&#39;, 3.0, 2),</span>
    <span class="go"> (&#39;several&#39;, 2.0999999046325684, 3),</span>
    <span class="go"> (&#39;tuples&#39;, -1.0, 2),</span>
    <span class="go"> (&#39;can&#39;, 3.0, 1),</span>
    <span class="go"> (&#39;be&#39;, 12.0, -3),</span>
    <span class="go"> (&#39;inserted&#39;, 0.0, -2),</span>
    <span class="go"> (&#39;at&#39;, 33.0, 0),</span>
    <span class="go"> (&#39;once&#39;, 0.0, 0)]</span>
    </pre></div>
    </div>
    </div>
    <div class="section" id="iopro-pyodbc-extensions">
    <h2>iopro.pyodbc extensions<a class="headerlink" href="#iopro-pyodbc-extensions" title="Permalink to this headline">¶</a></h2>
    <p>When using iopro.pyodbc it is possible to retrieve the results from queries directly into numpy containers. This is accomplished by using the new cursor methods fetchdictarray and fetchsarray.</p>
    </div>
    <div class="section" id="fetchdictarray">
    <h2>fetchdictarray<a class="headerlink" href="#fetchdictarray" title="Permalink to this headline">¶</a></h2>
    <p>fetchdictarray fetches the results of a query in a dictionary. By default fetchdictarray fetches all remaining rows in the cursor.:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;&#39;&#39;select * from test_table&#39;&#39;&#39;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">dictarray</span> <span class="o">=</span> <span class="n">cursor</span><span class="o">.</span><span class="n">fetchdictarray</span><span class="p">()</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="nb">type</span><span class="p">(</span><span class="n">dictarray</span><span class="p">)</span>
    <span class="go">dict</span>
    </pre></div>
    </div>
    <p>The keys in the dictionary are the column names::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">dictarray</span><span class="o">.</span><span class="n">keys</span><span class="p">()</span>
    <span class="go">[&#39;ival&#39;, &#39;name&#39;, &#39;fval&#39;]</span>
    </pre></div>
    </div>
    <p>Each column name is mapped to a numpy array (ndarray) as its value::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="s1">&#39;, &#39;</span><span class="o">.</span><span class="n">join</span><span class="p">([</span><span class="nb">type</span><span class="p">(</span><span class="n">dictarray</span><span class="p">[</span><span class="n">i</span><span class="p">])</span><span class="o">.</span><span class="n">__name__</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">dictarray</span><span class="o">.</span><span class="n">keys</span><span class="p">()])</span>
    <span class="go">&#39;ndarray, ndarray, ndarray&#39;</span>
    </pre></div>
    </div>
    <p>The types of the numpy arrays are infered from the database column information. So for our columns we get an appropriate numpy type. Note that in the case of name the type is a string of 11 characters even if in test_table is defined as varchar(10). The extra parameter is there to null-terminate the string::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="s1">&#39;, &#39;</span><span class="o">.</span><span class="n">join</span><span class="p">([</span><span class="nb">repr</span><span class="p">(</span><span class="n">dictarray</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">dtype</span><span class="p">)</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">dictarray</span><span class="o">.</span><span class="n">keys</span><span class="p">()])</span>
    <span class="go">&quot;dtype(&#39;int32&#39;), dtype(&#39;|S11&#39;), dtype(&#39;float32&#39;)&quot;</span>
    </pre></div>
    </div>
    <p>The numpy arrays will have a shape containing a single dimension with the number of rows fetched::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="s1">&#39;, &#39;</span><span class="o">.</span><span class="n">join</span><span class="p">([</span><span class="nb">repr</span><span class="p">(</span><span class="n">dictarray</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">dictarray</span><span class="o">.</span><span class="n">keys</span><span class="p">()])</span>
    <span class="go">&#39;(8L,), (8L,), (8L,)&#39;</span>
    </pre></div>
    </div>
    <p>The values in the different column arrays are index coherent. So in order to get the values associated to a given row it suffices to access each column using the appropriate index. The following snippet shows this correspondence::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span> <span class="s1">&#39;</span><span class="se">\n</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span>
    <span class="gp">... </span><span class="p">[</span><span class="s1">&#39;, &#39;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span>
    <span class="gp">... </span>    <span class="p">[</span><span class="nb">repr</span><span class="p">(</span><span class="n">dictarray</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="n">j</span><span class="p">])</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">dictarray</span><span class="o">.</span><span class="n">keys</span><span class="p">()])</span>
    <span class="gp">... </span>        <span class="k">for</span> <span class="n">j</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">dictarray</span><span class="p">[</span><span class="s1">&#39;name&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])])</span>
    <span class="go">2, &#39;foo&#39;, 3.0</span>
    <span class="go">3, &#39;several&#39;, 2.0999999</span>
    <span class="go">2, &#39;tuples&#39;, -1.0</span>
    <span class="go">1, &#39;can&#39;, 3.0</span>
    <span class="go">-3, &#39;be&#39;, 12.0</span>
    <span class="go">-2, &#39;inserted&#39;, 0.0</span>
    <span class="go">0, &#39;at&#39;, 33.0</span>
    <span class="go">0, &#39;once&#39;, 0.0</span>
    </pre></div>
    </div>
    <p>Having the results in numpy containers makes it easy to use numpy to analyze the data::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">dictarray</span><span class="p">[</span><span class="s1">&#39;fval&#39;</span><span class="p">])</span>
    <span class="go">6.5124998092651367</span>
    </pre></div>
    </div>
    <p>fetchdictarray accepts an optional parameter that places an upper bound to the number of rows to fetch. If there are not enough elements left to be fetched in the cursor the arrays resulting will be sized accordingly. This way it is possible to work with big tables in chunks of rows.:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;&#39;&#39;select * from test_table&#39;&#39;&#39;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">dictarray</span> <span class="o">=</span> <span class="n">cursor</span><span class="o">.</span><span class="n">fetchdictarray</span><span class="p">(</span><span class="mi">6</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span> <span class="n">dictarray</span><span class="p">[</span><span class="s1">&#39;name&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span>
    <span class="go">(6L,)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">dictarray</span> <span class="o">=</span> <span class="n">cursor</span><span class="o">.</span><span class="n">fetchdictarray</span><span class="p">(</span><span class="mi">6</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span> <span class="n">dictarray</span><span class="p">[</span><span class="s1">&#39;name&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span>
    <span class="go">(2L,)</span>
    </pre></div>
    </div>
    </div>
    <div class="section" id="fetchsarray">
    <h2>fetchsarray<a class="headerlink" href="#fetchsarray" title="Permalink to this headline">¶</a></h2>
    <p>fetchsarray fetches the result of a query in a numpy structured array.:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">cursor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="s1">&#39;&#39;&#39;select * from test_table&#39;&#39;&#39;</span><span class="p">)</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">sarray</span> <span class="o">=</span> <span class="n">cursor</span><span class="o">.</span><span class="n">fetchsarray</span><span class="p">()</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span> <span class="n">sarray</span>
    <span class="go">[(&#39;foo&#39;, 3.0, 2) (&#39;several&#39;, 2.0999999046325684, 3) (&#39;tuples&#39;, -1.0, 2)</span>
    <span class="go"> (&#39;can&#39;, 3.0, 1) (&#39;be&#39;, 12.0, -3) (&#39;inserted&#39;, 0.0, -2) (&#39;at&#39;, 33.0, 0)</span>
    <span class="go"> (&#39;once&#39;, 0.0, 0)]</span>
    </pre></div>
    </div>
    <p>The type of the result is a numpy array (ndarray)::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="nb">type</span><span class="p">(</span><span class="n">sarray</span><span class="p">)</span>
    <span class="go">numpy.ndarray</span>
    </pre></div>
    </div>
    <p>The dtype of the numpy array contains the description of the columns and their types::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">sarray</span><span class="o">.</span><span class="n">dtype</span>
    <span class="go">dtype([(&#39;name&#39;, &#39;|S11&#39;), (&#39;fval&#39;, &#39;&amp;lt;f4&#39;), (&#39;ival&#39;, &#39;&amp;lt;i4&#39;)])</span>
    </pre></div>
    </div>
    <p>The shape of the array will be one-dimensional, with cardinality equal to the number of rows fetched::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">sarray</span><span class="o">.</span><span class="n">shape</span>
    <span class="go">(8L,)</span>
    </pre></div>
    </div>
    <p>It is also possible to get the shape of a column. In this way it will look similar to the code needed when using dictarrays:</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">sarray</span><span class="p">[</span><span class="s1">&#39;name&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span>
    <span class="go">(8L,)</span>
    </pre></div>
    </div>
    <p>In a structured array it is as easy to access data by row or by column::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">sarray</span><span class="p">[</span><span class="s1">&#39;name&#39;</span><span class="p">]</span>
    <span class="go">array([&#39;foo&#39;, &#39;several&#39;, &#39;tuples&#39;, &#39;can&#39;, &#39;be&#39;, &#39;inserted&#39;, &#39;at&#39;, &#39;once&#39;],</span>
    <span class="go">      dtype=&#39;|S11&#39;)</span>







    <span class="gp">&gt;&gt;&gt; </span><span class="n">sarray</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="go">(&#39;foo&#39;, 3.0, 2)</span>
    </pre></div>
    </div>
    <p>It is also very easy and efficient to feed data into numpy functions::</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">sarray</span><span class="p">[</span><span class="s1">&#39;fval&#39;</span><span class="p">])</span>
    <span class="go">6.5124998092651367</span>
    </pre></div>
    </div>
    </div>
    <div class="section" id="fetchdictarray-vs-fetchsarray">
    <h2>fetchdictarray vs fetchsarray<a class="headerlink" href="#fetchdictarray-vs-fetchsarray" title="Permalink to this headline">¶</a></h2>
    <p>Both methods provide ways to input data from a database into a numpy-friendly container. The structured array version provides more flexibility extracting rows in an easier way. The main difference is in the memory layout of the resulting object. An in-depth analysis of this is beyond the scope of this notebook. Suffice it to say that you can view the dictarray laid out in memory as an structure of arrays  (in fact, a dictionary or arrays), while the structured array would be laid out in memory like an array of structures. This can make a lot of difference performance-wise when working with large chunks of data.</p>
    </div>
