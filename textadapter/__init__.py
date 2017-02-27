"""
    TextAdapter
    ~~~~~

    TextAdapter provides tools to interface large data files in a fast, memory-efficient way.
"""
from __future__ import absolute_import

from textadapter._version import get_versions
__version__ = get_versions()['version']
del get_versions

from textadapter.core.TextAdapter import (ArrayDealloc, CSVTextAdapter, 
                                          FixedWidthTextAdapter, JSONTextAdapter,
                                          RegexTextAdapter, s3_text_adapter,
                                          text_adapter)
from textadapter.core.loadtxt import loadtxt
from textadapter.core.genfromtxt import genfromtxt
from textadapter.lib.errors import (AdapterException, AdapterIndexError,
                                    ArgumentError, ConfigurationError,
                                    DataIndexError, DataTypeError,
                                    InternalInconsistencyError, NoSuchFieldError,
                                    ParserError, SourceError, SourceNotFoundError)


def test(verbosity=1, num_records=100000, results=[]):
    from textadapter.tests.test_TextAdapter import run as run_textadapter_tests
    result_text = run_textadapter_tests(verbosity=verbosity,
                                        num_records=num_records)
    results.append(result_text)
    
    from textadapter.tests.test_io import run as run_io_tests
    result_text = run_io_tests(verbosity=verbosity)
    results.append(result_text)
    
    for result in results:
        if not result.wasSuccessful():
            return False
    return True

