from threading import Thread
from Queue import Queue, Empty
import time
import uuid
import socket
import atexit
import signal
from iopro import pyodbc
import sys

import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
def protect_vertica(connstring, label):
    """uses atexist to register a helper which kills all queries
    where client_label matches label. To use this, you must set LABEL
    to some unique value inside your connection string
    """
    def signal_helper(*args, **kwargs):
        sys.exit(1)
        
    def helper(*args, **kwargs):
        print('cancel all')
        _cancel_all(connstring, label)
        
    signal.signal(signal.SIGTERM, signal_helper)
    atexit.register(helper)

def _cancel_all(connstring, label):
    """cancel_all sessions where client_label matches label.
    to use this, you must set LABEL to some unique value
    inside your connection string
    """
    q = """select session_id, statement_id from v_monitor.sessions
    where client_label='%s'""" % label
    conn = pyodbc.connect(connstring, ansi=True)
    data = conn.cursor().execute(q).fetchall()
    _interrupt_statements(conn, data)

def _cancel_conn(conn, queryid):
    q = """
    select session_id, statement_id from v_monitor.sessions where
    current_statement like '%%%s%%'
    and current_statement not like '%%v_monitor.sessions%%';
    """
    q = q % queryid
    data = conn.cursor().execute(q).fetchall()
    if len(data) == 1:
        _interrupt_statements(conn, data)
    
def _cancel(connstring, timeout, queryid):
    """after some timeout, close the statement associated with
    queryid.  queryid should be some uuid you add via sql comments
    """
    time.sleep(timeout)
    conn = pyodbc.connect(connstring, ansi=True)
    q = """
    select session_id, statement_id from v_monitor.sessions where
    current_statement like '%%%s%%'
    and current_statement not like '%%v_monitor.sessions%%';
    """
    q = q % queryid
    data = conn.cursor().execute(q).fetchall()
    if len(data) == 1:
        _interrupt_statements(conn, data)
            
def _interrupt_statements(conn, statement_information):
    for session_id, statement_id in statement_information:
        if statement_id:
            log.info("interrupting session:%s, statement:%s", session_id, statement_id)
            q = "select interrupt_statement('%s', '%s')" %\
                (session_id, statement_id)
            cur = conn.cursor()
            cur.execute(q)
            print('results', cur.fetchall())
            
def fetchall_timeout(cursor, connstring, query, timeout=10):
    queryid = str(uuid.uuid4())
    query += " /* %s */ " % queryid
    q = Queue()
    def helper():
        try:
            result = cursor.execute(query).fetchall()
            q.put(result)
        except Exception as e:
            q.put(e)
    try:
        t = Thread(target=helper)
        t.start()
        val = q.get(True, timeout)
        if isinstance(val, Exception):
            raise val
        return val
    except (KeyboardInterrupt, Empty) as e:
        print('cancelling')
        _cancel(connstring, 0, queryid)
