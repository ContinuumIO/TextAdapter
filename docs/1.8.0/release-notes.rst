IOPro Release Notes
===================

.. raw:: html

    <div class="section" id="id1">
    <h2>2016-04-05:  1.8.0:<a class="headerlink" href="#id1" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Add PostgresAdapter for reading data from PostgreSQL databases</li>
    <li>Add AccumuloAdapter for reading data from Accumulo databases</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id2">
    <h2>2015-10-09:  1.7.2:<a class="headerlink" href="#id2" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix an issue with pyodbc where result NumPy arrays could return
    uninitialized data after the actual data null character.  Now it pads
    the results with nulls.</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id3">
    <h2>2015-05-04:  1.7.1<a class="headerlink" href="#id3" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Properly cache output string objects for better performance</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id4">
    <h2>2015-03-02:  1.7.0<a class="headerlink" href="#id4" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Add Python 3 support</li>
    <li>Add support for parsing utf8 text files</li>
    <li>Add ability to set/get field types in MongoAdapter</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id5">
    <h2>2015-02-02:  1.6.11<a class="headerlink" href="#id5" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix issue with escape char not being parsed correctly inside quoted strings</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id6">
    <h2>2014-12-17:  1.6.10<a class="headerlink" href="#id6" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix issue with using field filters with json parser</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id7">
    <h2>2014-12-02:  1.6.9<a class="headerlink" href="#id7" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix issue with json field names getting mixed up</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id8">
    <h2>2014-11-20:  1.6.8<a class="headerlink" href="#id8" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix issue with return nulls returning wrong &#8220;null&#8221; for large queries
    (more than 10000 rows) in some circumpstances.</li>
    <li>Fix issue with reading slices of json data</li>
    <li>Change json parser so that strings fields of numbers do not get converted
    to number type by default</li>
    <li>Allow json field names to be specified with field_names constructor
    argument</li>
    <li>If user does not specify json field names, use json attribute names as
    field names in array result</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id9">
    <h2>2014-07-03:  1.6.7<a class="headerlink" href="#id9" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix issue when reading more than 10000 rows containing unicode strings in platfrom where ODBC uses UTF-16/UCS2 encoding (notably Windows and unixODBC). The resulting data could be corrupt.</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id10">
    <h2>2014-06-16:  1.6.6<a class="headerlink" href="#id10" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix possible segfault when dealing with unicode strings in platforms where ODBC uses UTF-16/UCS2 encoding (notably Windows and unixODBC)</li>
    <li>Add iopro_set_text_limit function to iopro. It globally limits the size of text fields read by fetchdictarray and fetchsarray. By default it is set to 1024 characters.</li>
    <li>Fix possible segfault in fetchdictarray and fetchsarray when failing to allocate some NumPy array. This could notably happen in the presence of &#8220;TEXT&#8221; fields. Now it will raise an OutOfMemory error.</li>
    <li>Add lazy loading of submodules in IOPro. This reduces upfront import time of IOPro. Features are imported as they are used for the first time.</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id11">
    <h2>2014-05-07:  1.6.5<a class="headerlink" href="#id11" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix crash when building textadapter index</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id12">
    <h2>2014-04-29:  1.6.4<a class="headerlink" href="#id12" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix default value for null strings in IOPro/pyodbc changed to be an empty string instead of &#8216;NA&#8217;. NA was not appropriate as it can collide with valid data (Namibia country code is &#8216;NA&#8217;, for example), and it failed with single character columns.</li>
    <li>Ignore SQlRowCount when performing queries with fetchsarray and fetchdictarray, since SQLRowCount sometimes returns incorrect number of rows.</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id13">
    <h2>2014-03-25:  1.6.3<a class="headerlink" href="#id13" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix SQL TINYINT is now returned as an unsigned 8 bit integer in fetchdictarray/fetchsarray. This is to match the range specified in SQL (0...255). It was being returned as a signed 8 bit integer before (range -128...127)</li>
    <li>Add Preliminary unicode string support in fetchdictarray/fetchsarray.</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id14">
    <h2>2014-02-12:  1.6.2<a class="headerlink" href="#id14" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Disable Numba support for version 0.12 due to lack of string support.</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id15">
    <h2>2014-01-30:  1.6.1<a class="headerlink" href="#id15" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix a regression that made possible some garbage in string fields when using fetchdictarray/fetchsarray.</li>
    <li>Fix a problem where heap corruption could happen in IOPro.pyodbc fetchdictarray/fetchsarray related to nullable string fields.</li>
    <li>Fix the allocation guard debugging code: iopro.pyodbc.enable_mem_guards(True|False) should no longer crash.</li>
    <li>Merge Vertica fix for cancelling queries</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id16">
    <h2>2013-10-30:  1.6.0<a class="headerlink" href="#id16" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Add JSON support</li>
    <li>Misc bug fixes</li>
    <li>Fix crash in IOPro.pyodbc when dealing with nullable datetimes in fetch_dictarray and fetch_sarray.</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id17">
    <h2>2013-06-12:  1.5.5<a class="headerlink" href="#id17" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Fix issue parsing negative ints with leading whitespace in csv data.</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id18">
    <h2>2013-06-10:  1.5.4<a class="headerlink" href="#id18" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Allow delimiter to be set to None for csv files with single field.</li>
    <li>Fill in missing csv fields with fill values.</li>
    <li>Fill in blank csv lines with fill values for pandas dataframe output.</li>
    <li>Allow list of field names for TextAdapter field_names parameter.</li>
    <li>Change default missing fill value to empty string for string fields.</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id19">
    <h2>2013-06-05:  1.5.3<a class="headerlink" href="#id19" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Temporary fix for IndexError exception in TextAdapter.__read_slice method.</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id20">
    <h2>2013-05-28:  1.5.2<a class="headerlink" href="#id20" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Add ability to specify escape character in csv data</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id21">
    <h2>2013-05-23:  1.5.1<a class="headerlink" href="#id21" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>fixed coredump when using datetime with numpy &lt; 1.7</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id22">
    <h2>2013-05-22:  1.5.0<a class="headerlink" href="#id22" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Added a cancel method to the Cursor object in iopro.pyodbc.
    This method wraps ODBC SQLCancel.</li>
    <li>DECIMAL and NUMERIC types are now working on iopro.pyodbc on regular fetch
    functions. They are still unsupported in fetchsarray and fetchdict and
    fetchsarray</li>
    <li>Add ftp support</li>
    <li>Performance improvements to S3 support</li>
    <li>Misc bug fixes</li>
    </ul>
    </div></blockquote>
    </div>
    <div class="section" id="id23">
    <h2>2013-04-05:  1.4.3<a class="headerlink" href="#id23" title="Permalink to this headline">¶</a></h2>
    <blockquote>
    <div><ul class="simple">
    <li>Update loadtxt and genfromtxt to reflect numpy versions&#8217; behavior
    for dealing with whitespace (default to any whitespace as delimiter,
    and treat multiple whitespace as one delimiter)</li>
    <li>Add read/write field_names property</li>
    <li>Add support for pandas dataframes as output</li>
    <li>Misc bug fixes</li>
    </ul>
    </div></blockquote>
    </div>
