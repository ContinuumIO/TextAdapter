"""
    TextAdapter
    ~~~~~

    TextAdapter provides tools to work with large data files in a fast,
    memory-efficient way.
"""

from __future__ import absolute_import

from types import ModuleType
import sys
from ._version import get_versions
import os

__version__ = get_versions()['version']
del get_versions

module_to_attr_dict = {
    'TextAdapter.textadapter.TextAdapter':    
            ['ArrayDealloc', 'CSVTextAdapter', 
             'FixedWidthTextAdapter', 'JSONTextAdapter',
             'RegexTextAdapter', 's3_text_adapter',
             'text_adapter'],
    'TextAdapter.textadapter.loadtxt':        
            ['loadtxt'],
    'TextAdapter.textadapter.genfromtxt':     
            ['genfromtxt'],
    'TextAdapter.lib.errors':                 
            ['AdapterException', 'AdapterIndexError',
             'ArgumentError', 'ConfigurationError',
             'DataIndexError', 'DataTypeError',
             'InternalInconsistencyError', 'NoSuchFieldError',
             'ParserError', 'SourceError', 'SourceNotFoundError']
}

attr_to_module_dict = {}
for module, items in module_to_attr_dict.items():
    for item in items:
        attr_to_module_dict[item] = module


meta_attr = {
    '__file__':  __file__,
    '__package__': __name__,
    '__path__': __path__,
    '__doc__': __doc__,
    '__version__': __version__,
    '__all__': tuple([key for key in attr_to_module_dict.keys()] + 
                     ['textadapter', 'mongoadapter', 'postgresadapter'] + 
                     ['test', 'test_mongo', 'test_postgres', 'test_accumulo'])
}

class module(ModuleType):
    """
    The actual TextAdapter module. It provides lazy-loading of submodules.
    Import time is only paid for those submodules used
    """

    def __getattr__(self, name):
        module_to_load = attr_to_module_dict.get(name)
        if module_to_load is not None:
            __import__(module_to_load)
            module = sys.modules[module_to_load]
            for attr in module_to_attr_dict[module_to_load]:
                setattr(self, attr, getattr(module, attr))
        return ModuleType.__getattribute__(self, name)

    def __dir__(self):
        loaded = set(meta_attr.keys())
        potentially_not_loaded = set(self.__all__)
        return sorted(loaded | potentially_not_loaded)
    
    def test(self, verbosity=1, num_records=100000, results = []):
                
        from .tests.test_TextAdapter import run as run_textadapter_tests
        result_text = run_textadapter_tests(verbosity=verbosity,
                                            num_records=num_records)
        results.append(result_text)

        from .tests.test_io import run as run_io_tests
        result_text = run_io_tests(verbosity=verbosity)
        results.append(result_text)

        for result in results:
            if not result.wasSuccessful():
                return False
        return True

    def test_mongo(self, host='localhost', port=27017, verbosity=2):
        from .tests.test_MongoAdapter import run as run_mongo_tests
        return run_mongo_tests(verbosity, host, port)

    def test_postgres(self, host, dbname, user, verbose=True):
        import pytest
        test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tests')        
        postgres_test_script = 'test_PostgresAdapter.py'
        args = []
        args.append(os.path.join(test_dir, postgres_test_script))
        args.append('--pg_host {0}'.format(host))
        args.append('--pg_dbname {0}'.format(dbname))
        args.append('--pg_user {0}'.format(user))
        if verbose:
            args.append('-v')
    
        result = pytest.main(' '.join(args))
        if result == 0:
            return True
        return False
    
    def test_accumulo(self, host, user, password, verbose=True):
        import pytest
        test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tests')        
        accumulo_test_script = 'test_AccumuloAdapter.py'
        args = []
        args.append(os.path.join(test_dir, accumulo_test_script))
        args.append('--acc_host {0}'.format(host))
        args.append('--acc_user {0}'.format(user))
        args.append('--acc_password {0}'.format(password))
        if verbose:
            args.append('-v')
    
        result = pytest.main(' '.join(args))
        if result == 0:
            return True
        return False
    

new_module = module(__name__)
new_module.__dict__.update(meta_attr)

# this is somewhat magic, if not present it seems that the old module gets
# disposed causing problems.
_old_module = sys.modules[__name__]
# replace the module with our module class
sys.modules[__name__] = new_module

