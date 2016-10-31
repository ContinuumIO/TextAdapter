import os
import sys
from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import pyodbc_setup
import versioneer

class CleanInplace(Command):
    user_options = []

    def initialize_options(self):
        self.cwd = None

    def finalize_options(self):
        self.cwd = os.getcwd()

    def run(self):
        files = ['./iopro/textadapter/TextAdapter.c',
                 './iopro/textadapter/TextAdapter.so',
                 './iopro/mongoadapter/MongoAdapter.so',
                 './iopro/mongoadapter/MongoAdapter.so',
                 './iopro/pyodbc.so',
                 './iopro/postgresadapter/PostgresAdapter.so',
                 './iopro/pyodbc.cpython-35m-darwin.so']
        for file in files:
            try:
                os.remove(file)
            except OSError:
                pass


def setup_text(include_dirs, lib_dirs):
    src = ['iopro/textadapter/TextAdapter.pyx',
           'iopro/textadapter/text_adapter.c',
           'iopro/lib/converter_functions.c',
           'iopro/textadapter/io_functions.c',
           'iopro/lib/field_info.c',
           'iopro/textadapter/json_tokenizer.c']

    if sys.platform == 'win32':
        zlib_lib = 'zlibstatic'
    else:
        zlib_lib = 'z'

    compile_args = []
    if '--debug' in sys.argv:
        if sys.platform == 'win32':
            compile_args.append('/DDEBUG_ADAPTER')
        else:
            compile_args.append('-DDEBUG_ADAPTER')

    libraries = ['pcre', zlib_lib]
    include_dirs = ['iopro/textadapter'] + include_dirs

    return Extension("iopro.textadapter.TextAdapter",
                     src,
                     include_dirs=include_dirs,
                     library_dirs=lib_dirs,
                     libraries=libraries,
                     extra_compile_args=compile_args)


def setup_mongo(include_dirs, lib_dirs):
    src = ['iopro/mongoadapter/MongoAdapter.pyx',
           'iopro/mongoadapter/mongo_adapter.c',
           'iopro/lib/field_info.c',
           'iopro/lib/converter_functions.c']

    return Extension("iopro.mongoadapter.MongoAdapter",
                     src,
                     include_dirs=include_dirs,
                     libraries=['mongoc', 'bson'],
                     library_dirs=lib_dirs)


def setup_odbc(include_dirs, lib_dirs):
    src_path = os.path.join(os.path.dirname(__file__), 'pyodbc/src')
    src = [os.path.abspath(os.path.join(src_path, f))
           for f in os.listdir(src_path)
           if f.endswith('.cpp') ]

    if sys.platform == 'win32':
        libraries = ['odbc32', 'advapi32']
    elif sys.platform == 'darwin':
        if os.environ.get('UNIXODBC_PATH', ''):
            include_dirs.append(os.path.join(os.environ.get('UNIXODBC_PATH')))
            include_dirs.append(os.path.join(os.environ.get('UNIXODBC_PATH'), 'include'))
            lib_dirs.append(os.path.join(os.environ.get('UNIXODBC_PATH'), 'DriverManager', '.libs'))
            libraries = ['odbc']
        else:
            libraries = ['odbc']
    else:
        libraries = ['odbc']

    return Extension('iopro.pyodbc',
                     src,
                     include_dirs=include_dirs,
                     libraries=libraries,
                     library_dirs=lib_dirs)

def setup_postgres(include_dirs, lib_dirs):
    src = ['iopro/postgresadapter/PostgresAdapter.pyx',
           'iopro/postgresadapter/postgres_adapter.c',
           'iopro/postgresadapter/postgis_fields.c',
           'iopro/lib/converter_functions.c',
           'iopro/lib/kstring.c']

    libraries = []
    if sys.platform == 'win32':
        libraries = ['libpq', 'ws2_32', 'secur32', 'shell32', 'advapi32']
    else:
        libraries = ['pq']

    return Extension('iopro.postgresadapter.PostgresAdapter',
                     src,
                     include_dirs=include_dirs,
                     libraries=libraries,
                     library_dirs=lib_dirs)


def setup_accumulo(include_dirs, lib_dirs):
    src = ['iopro/accumuloadapter/AccumuloAdapter.pyx',
           'iopro/accumuloadapter/accumulo_adapter.cpp',
           'iopro/accumuloadapter/AccumuloProxy.cpp',
           'iopro/accumuloadapter/proxy_types.cpp',
           'iopro/accumuloadapter/proxy_constants.cpp']

    extra_compile_args = []
    if sys.platform == 'win32':
        if sys.version_info < (3,0):
            libs = ['thrift',
                    'boost_thread-vc90-mt-1_60',
                    'boost_system-vc90-mt-1_60',
                    'boost_chrono-vc90-mt-1_60']
        else:
            libs = ['thrift',
                    'boost_thread-vc100-mt-1_60',
                    'boost_system-vc100-mt-1_60',
                    'boost_chrono-vc100-mt-1_60']
        extra_compile_args.append('/D BOOST_ALL_DYN_LINK')
    else:
        libs = ['thrift']
        extra_compile_args = ['-std=c++11']

    return Extension('iopro.accumuloadapter.AccumuloAdapter',
                     src,
                     language='c++',
                     include_dirs=include_dirs,
                     library_dirs=lib_dirs,
                     libraries=libs,
                     extra_compile_args=extra_compile_args)


def run_setup(text=True,
              mongo=True,
              odbc=True,
              postgres=True,
              accumulo=True):

    include_dirs = [os.path.join('iopro', 'lib'),
                    numpy.get_include()]
    if sys.platform == 'win32':
        include_dirs.append(os.path.join(sys.prefix, 'Library', 'include'))
    else:
        include_dirs.append(os.path.join(sys.prefix, 'include'))

    lib_dirs = []
    if sys.platform == 'win32':
        lib_dirs.append(os.path.join(sys.prefix, 'Library', 'lib'))
    else:
        lib_dirs.append(os.path.join(sys.prefix, 'lib'))

    ext_modules = []
    packages = ['iopro', 'iopro.lib', 'iopro.tests']
    if text:
        ext_modules.append(setup_text(include_dirs, lib_dirs))
        packages.append('iopro.textadapter')
    if mongo:
        ext_modules.append(setup_mongo(include_dirs, lib_dirs))
        packages.append('iopro.mongoadapter')
    if odbc:
        ext_modules.append(setup_odbc(include_dirs, lib_dirs))
    if postgres:
        ext_modules.append(setup_postgres(include_dirs, lib_dirs))
        packages.append('iopro.postgresadapter')
    if accumulo:
        ext_modules.append(setup_accumulo(include_dirs, lib_dirs))
        packages.append('iopro.accumuloadapter')

    versioneer.versionfile_source = 'iopro/_version.py'
    versioneer.versionfile_build = 'iopro/_version.py'
    versioneer.tag_prefix = ''
    versioneer.parentdir_prefix = 'iopro-'

    cmdclass = versioneer.get_cmdclass()
    cmdclass['build_ext'] = build_ext
    cmdclass['cleanall'] = CleanInplace

    setup(name='iopro',
          version = versioneer.get_version(),
          description='optimized IO for NumPy/Blaze',
          author='Jay Bourque',
          author_email='jay.bourque@continuum.io',
          ext_modules=ext_modules,
          packages=packages,
          cmdclass=cmdclass)


if __name__ == '__main__':

    def parse_build_arg(name):
        if '--build_{0}'.format(name) in sys.argv:
            sys.argv.remove('--build_{0}'.format(name))
            return True
        return False

    build_text = parse_build_arg('text')
    build_mongo = parse_build_arg('mongo')
    build_odbc = parse_build_arg('odbc')
    build_postgres = parse_build_arg('postgres')
    build_accumulo = parse_build_arg('accumulo')

    if (not build_text and
            not build_mongo and
            not build_odbc and
            not build_postgres and
            not build_accumulo):
        # Build all modules by default
        run_setup()
    else:
        # User has selected specific modules to build
        run_setup(text=build_text,
                  mongo=build_mongo,
                  odbc=build_odbc,
                  postgres=build_postgres,
                  accumulo=build_accumulo)
