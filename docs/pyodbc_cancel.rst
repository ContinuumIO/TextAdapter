=================================
 iopro.pyodbc Cancelling Queries
=================================

Starting with version 1.5, the pyodbc submodule of IOPro makes it
possible to cancel operations. This is done by exposing the SQLCancel
ODBC function as a cancel method in the Cursor object.


A Simple Example
================

A very simple example would be:


::

   conn = iopro.pyodbc.connect(conn_str)
   cursor = conn.cursor()
   cursor.execute('SELECT something FROM sample_table')
   result = cursor.fetchone()
   cursor.cancel()


This is not very interesting, and it doesn't add much to the
functionality of pyodbc.

What makes the cancel method more interesting is that it is possible
to cancel running statements that are blocking another thread.


A Sample With Threading
=======================

Having access to the cancel method it is possible to stop running
queries following different criteria. For example, it would be
possible to execute queries with a time-out. If the time runs out, the
query gets cancelled.

::

    import iopro.pyodbc
    import time
    import threading

    def query_with_time_out(conn, query, timeout):
        def watchdog(cursor, time_out):
            time.sleep(wait_time)
            cursor.cancel()

        cursor = conn.cursor()

        t = threading.Thread(target=watchdog, args=(cursor, timeout))
        t.start()
        try:
            cursor.execute(query)

            result = cursor.fetchall()
        except iopro.pyodbc.Error:
            result = 'timed out'

	return result


This is just one possibility. As cursor exposes directly the
SQLCancel, many oportunities open in implementing policies to cancel
running queries.


Finishing notes
===============

In order for this to work, the underlying ODBC driver must support
SQLCancel.

The pyodbc submodule of IOPro releases the Python GIL when it calls
ODBC, so while queries are being executed other Python threads
continue to execute while the thread that performed the query is
blocked. This allows for cancel to be called by another
thread. Coupled with threading, the cancel method is a very useful
primitive.


