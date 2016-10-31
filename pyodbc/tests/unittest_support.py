"""
support classes for IOPro/pyodbc tests
"""

from __future__ import absolute_import, print_function, division

from unittest import TestCase
import os
import iopro.pyodbc as pyodbc
from contextlib import contextmanager

import functools

_conn_string_key = 'IOPRO_PYODBC_TEST_CONNSTR'
_conn_string = os.environ.get(_conn_string_key)
_enable_logging = bool(os.environ.get('IOPRO_PYODBC_TEST_LOGGING'))
_unicode_results = bool(os.environ.get('IOPRO_PYODBC_TEST_UNICODE_RESULTS'))
_test_db = os.environ.get("IOPRO_PYODBC_TEST_DBMS")

_error_string = """
Set the environment variable "{0}" to the connection string for
your test database.
example (bash):
export IOPRO_PYODBC_TEST_CONNSTR='DRIVER={{FreeTDS}};SERVER=192.168.1.135;DATABASE=test;Port=1433;Uid=test;Pwd=test'
""".format(_conn_string_key)


# Configure pyodbc for execution ***REMOVED******REMOVED***

class IOProPyodbcTest(TestCase):
    """
    Provides a connection (self.conn) that is initialized from
    environment variables.

    Subclasses can implement a couple of methods to create/cleanup
    tables used as tests. This should be implemented as class
    methods so that the tables are created once per class.
    """

    def setUp(self):
        pyodbc.enable_mem_guards(True)
        pyodbc.enable_tracing(_enable_logging)
        self.assertIsNotNone(_conn_string, msg=_error_string)
        try:
            self.conn = pyodbc.connect(_conn_string, unicode_results=_unicode_results, timeout=3)
        except Exception as e:
            raise Exception('It seems that your {0} is not setup correctly. Attempting to connect resulted in:\n{1}'.format(_conn_string_key, e.args[1]))

    def tearDown(self):
        del self.conn


# decorators for test specific to some databases...
class DBMS(object):
    SQL_Server = 'sql_server'
    PostgreSQL = 'postgresql'

_supported_dbms = [getattr(DBMS, i) for i in dir(DBMS) if not i.startswith('_')]
_warn_message="""
Warn: Supplied IOPRO_PYODBC_TEST_DBMS '{0}' ignored.
Try one of the following:
\t{1}
"""

if _test_db and not _test_db in _supported_dbms:
    print(_warn_message.format(_test_db, '\n\t'.join(_supported_dbms))) 


class dbms_specific(object):
    """
    A decorator to mark tests as specific to a given (set) of DBMS.
    Because they use DBMS specific types/SQL extensions, for example.

    Sample use:
    @dbms_specific(DBMS.SQL_Server, DBMS.PostgreSQL)
    """
    def __init__(self, *args):
        self.dbms = args

    def __call__(self, fn):
        if _test_db in self.dbms:
            return fn
        else:
            @functools.wraps(fn)
            def fail(*args, **kwargs):
                raise SkipTest("only for dbms: {0}".format(', '.join(self.dbms)))
            return fail

def get_connection_string():
    return _conn_string


@contextmanager
def text_limit(size):
    old = pyodbc.iopro_set_text_limit(size)
    yield
    pyodbc.iopro_set_text_limit(old)

