import os
import sys
from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import versioneer


class CleanInplace(Command):
    user_options = []

    def initialize_options(self):
        self.cwd = None

    def finalize_options(self):
        self.cwd = os.getcwd()

    def run(self):
        files = ['./textadapter/core/TextAdapter.c',
                 './textadapter/core/TextAdapter.so']
        for file in files:
            try:
                os.remove(file)
            except OSError:
                pass


def setup_text(include_dirs, lib_dirs):
    src = ['textadapter/core/TextAdapter.pyx',
           'textadapter/core/text_adapter.c',
           'textadapter/lib/converter_functions.c',
           'textadapter/core/io_functions.c',
           'textadapter/lib/field_info.c',
           'textadapter/core/json_tokenizer.c']

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
    include_dirs = ['textadapter/core'] + include_dirs

    return Extension("textadapter.core.TextAdapter",
                     src,
                     include_dirs=include_dirs,
                     library_dirs=lib_dirs,
                     libraries=libraries,
                     extra_compile_args=compile_args)


def run_setup():
    include_dirs = [os.path.join('textadapter', 'lib'),
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
    packages = ['textadapter', 'textadapter.lib', 'textadapter.tests']

    ext_modules.append(setup_text(include_dirs, lib_dirs))
    packages.append('textadapter.core')

    versioneer.versionfile_source = 'textadapter/_version.py'
    versioneer.versionfile_build = 'textadapter/_version.py'
    versioneer.tag_prefix = ''
    versioneer.parentdir_prefix = 'textadapter-'

    cmdclass = versioneer.get_cmdclass()
    cmdclass['build_ext'] = build_ext
    cmdclass['cleanall'] = CleanInplace

    setup(name='textadapter',
          version = versioneer.get_version(),
          description='optimized IO for NumPy/Blaze',
          author='Continuum Analytics',
          author_email='david.mertz@continuum.io',
          ext_modules=ext_modules,
          packages=packages,
          cmdclass=cmdclass)


if __name__ == '__main__':
    run_setup()
