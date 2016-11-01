import os
import sys
from glob import glob
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
        files = ['./TextAdapter/textadapter/TextAdapter.c',
                 './TextAdapter/textadapter/TextAdapter.so',
                 './TextAdapter/mongoadapter/MongoAdapter.so',
                 './TextAdapter/mongoadapter/MongoAdapter.so',
                 './TextAdapter/pyodbc.so',
                 './TextAdapter/postgresadapter/PostgresAdapter.so',
                 './TextAdapter/pyodbc.cpython-35m-darwin.so']

        dirs = ['./TextAdapter/build/*',
                './__pycache__/*',
                './TextAdapter/__pycache__/*']
        for dir in dirs:
            for file in glob(dir):
                files.append(file)

        for file in files:
            try:
                os.remove(file)
            except OSError:
                pass


def setup_text(include_dirs, lib_dirs):
    src = ['TextAdapter/textadapter/TextAdapter.pyx',
           'TextAdapter/textadapter/text_adapter.c',
           'TextAdapter/lib/converter_functions.c',
           'TextAdapter/textadapter/io_functions.c',
           'TextAdapter/lib/field_info.c',
           'TextAdapter/textadapter/json_tokenizer.c']

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
    include_dirs = ['TextAdapter/textadapter'] + include_dirs

    return Extension("TextAdapter.textadapter.TextAdapter",
                     src,
                     include_dirs=include_dirs,
                     library_dirs=lib_dirs,
                     libraries=libraries,
                     extra_compile_args=compile_args)

def run_setup():

    include_dirs = [os.path.join('TextAdapter', 'lib'),
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
    packages = ['TextAdapter', 'TextAdapter.lib', 'TextAdapter.tests']
    ext_modules.append(setup_text(include_dirs, lib_dirs))
    packages.append('TextAdapter.textadapter')

    versioneer.versionfile_source = 'TextAdapter/_version.py'
    versioneer.versionfile_build = 'TextAdapter/_version.py'
    versioneer.tag_prefix = ''
    versioneer.parentdir_prefix = 'TextAdapter-'

    cmdclass = versioneer.get_cmdclass()
    cmdclass['build_ext'] = build_ext
    cmdclass['cleanall'] = CleanInplace

    setup(name='TextAdapter',
          version = versioneer.get_version(),
          description='optimized IO for NumPy/Blaze',
          author='Continuum Analytics',
          author_email='david.mertz@continuum.io',
          ext_modules=ext_modules,
          packages=packages,
          cmdclass=cmdclass)


if __name__ == '__main__':

    def parse_build_arg(name):
        if '--build_{0}'.format(name) in sys.argv:
            sys.argv.remove('--build_{0}'.format(name))
            return True
        return False

    run_setup()
