<<<<<<< HEAD
***REMOVED***-
MongoAdapter
***REMOVED***-

***REMOVED***
=======
------------
MongoAdapter
------------

.. raw:: html
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1

    <div class="contents topic" id="contents">
    <p class="topic-title first">Contents</p>
    <ul class="simple">
    <li><a class="reference internal" href="#mongoadapter" id="id1">MongoAdapter</a><ul>
    <li><a class="reference internal" href="#methods" id="id2">Methods</a></li>
<<<<<<< HEAD
    <li><a class="reference internal" href="#basic-usage" id="id3">***REMOVED***</a></li>
    </ul>
    </li>
    </ul>
***REMOVED***
=======
    <li><a class="reference internal" href="#basic-usage" id="id3">Basic Usage</a></li>
    </ul>
    </li>
    </ul>
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    <p>The MongoAdapter module reads data from a Mongo database collection and produces a
    NumPy array containing the loaded. The following features are currently implemented:</p>
    <ul class="simple">
    <li>The MongoAdapter engine is written in C to ensure data is loaded fast with minimal
    memory usage.</li>
    <li>Python slicing notation can be used to specify the subset of records to be
    read from the data source.</li>
    <li>The MongoAdapter engine has automatic type inference so the user does not have to
    specify dtypes of the output array.</li>
    </ul>
    <div class="section" id="methods">
    <h2><a class="toc-backref" href="#id2">Methods</a><a class="headerlink" href="#methods" title="Permalink to this headline">¶</a></h2>
    <p>The MongoAdapter module contains the follwowing constructor for creating MongoAdapter objects:</p>
    <dl class="docutils">
    <dt><strong>MongoAdapter</strong> (host, port, database, collection)</dt>
    <dd><div class="first line-block">
    <div class="line">MongoAdapter contructor</div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    <div class="last line-block">
    <div class="line">host - Host name where Mongo database is running.</div>
    <div class="line">port - Port number where Mongo database is running.</div>
    <div class="line">database - Mongo database to connect to</div>
    <div class="line">collection - Mongo database collection</div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    </dd>
    <dt><strong>set_field_names</strong> (names)</dt>
    <dd><div class="first last line-block">
    <div class="line">Set field names to read when creating output NumPy array.</div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    </dd>
    <dt><strong>get_field_names</strong> ()</dt>
    <dd><div class="first last line-block">
    <div class="line">Returns names of fields that will be read when reading data from Mongo database.</div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    </dd>
    <dt><strong>set_field_types</strong> (types=None)</dt>
    <dd><div class="first last line-block">
    <div class="line">Set NumPy dtypes for each field, specified as a dict of field names/indices and associated
    dtype. (Example: {0:&#8217;u4&#8217;, 1:&#8217;f8&#8217;, 2:&#8217;S10&#8217;})</div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    </dd>
    <dt><strong>get_field_types</strong> ()</dt>
    <dd><div class="first last line-block">
    <div class="line">Returns dict of field names/indices and associated NumPy dtype.</div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    </dd>
    </dl>
    <p>The MongoAdapter object contains the following properties:</p>
    <dl class="docutils">
    <dt><strong>size</strong> (readonly)</dt>
    <dd><div class="first last line-block">
    <div class="line">Number of documents in the Mongo database + collection specified in constructor.</div>
<<<<<<< HEAD
***REMOVED***
    </dd>
    </dl>
***REMOVED***
    <div class="section" id="basic-usage">
    <h2><a class="toc-backref" href="#id3">***REMOVED***</a><a class="headerlink" href="#basic-usage" title="Permalink to this headline">¶</a></h2>
=======
    </div>
    </dd>
    </dl>
    </div>
    <div class="section" id="basic-usage">
    <h2><a class="toc-backref" href="#id3">Basic Usage</a><a class="headerlink" href="#basic-usage" title="Permalink to this headline">¶</a></h2>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    <ol class="arabic">
    <li><p class="first">Create MongoAdapter object for data source</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">iopro</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">adapter</span> <span class="o">=</span> <span class="n">iopro</span><span class="o">.</span><span class="n">MongoAdapter</span><span class="p">(</span><span class="s1">&#39;localhost&#39;</span><span class="p">,</span> <span class="mi">27017</span><span class="p">,</span> <span class="s1">&#39;database_name&#39;</span><span class="p">,</span> <span class="s1">&#39;collection_name&#39;</span><span class="p">)</span>
    </pre></div>
<<<<<<< HEAD
***REMOVED***
=======
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
    </li>
    <li><p class="first">Load Mongo collection documents into NumPy array using slicing notation</p>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># read all records for &#39;field0&#39; field</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[</span><span class="s1">&#39;field0&#39;</span><span class="p">][:]</span>
    </pre></div>
<<<<<<< HEAD
***REMOVED***
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># read first ten records for &#39;field0&#39; and &#39;field1&#39; fields</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[[</span><span class="s1">&#39;field0&#39;</span><span class="p">,</span> <span class="s1">&#39;field1&#39;</span><span class="p">]][</span><span class="mi">0</span><span class="p">:</span><span class="mi">10</span><span class="p">]</span>
    </pre></div>
***REMOVED***
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># read last record</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[</span><span class="s1">&#39;field0&#39;</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
    </pre></div>
***REMOVED***
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># read every other record</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[</span><span class="s1">&#39;field0&#39;</span><span class="p">][::</span><span class="mi">2</span><span class="p">]</span>
    </pre></div>
***REMOVED***
    </li>
    </ol>
***REMOVED***
=======
    </div>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># read first ten records for &#39;field0&#39; and &#39;field1&#39; fields</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[[</span><span class="s1">&#39;field0&#39;</span><span class="p">,</span> <span class="s1">&#39;field1&#39;</span><span class="p">]][</span><span class="mi">0</span><span class="p">:</span><span class="mi">10</span><span class="p">]</span>
    </pre></div>
    </div>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># read last record</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[</span><span class="s1">&#39;field0&#39;</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
    </pre></div>
    </div>
    <div class="highlight-default"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># read every other record</span>
    <span class="gp">&gt;&gt;&gt; </span><span class="n">array</span> <span class="o">=</span> <span class="n">adapter</span><span class="p">[</span><span class="s1">&#39;field0&#39;</span><span class="p">][::</span><span class="mi">2</span><span class="p">]</span>
    </pre></div>
    </div>
    </li>
    </ol>
    </div>
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
