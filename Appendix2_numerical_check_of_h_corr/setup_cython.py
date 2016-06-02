'''
program:       setup_cython.py
author:        tc
last-modified: 2016-05-12 -- 19 CEST
description:   compiles a cython module (including the directives for
               line profiling)
notes:         to be executed through
               $ python setup_cython.py build_ext --inplace
'''

import glob
from distutils.extension import Extension
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Compiler.Options import directive_defaults

define_macros = []
# uncomment the next block if you want to line-profile the cython code
#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True
#directive_defaults['profile'] = True
#define_macros = [('CYTHON_TRACE', '1')]

libraries = []
# uncomment the next line if your cython code uses gsl functions
#libraries = ['gsl', 'gslcblas']

if __name__ == '__main__':
    for lib in glob.glob('*.pyx'):
        print 'compiling %s' % lib
        basename = lib[:-4]
        ext_modules = [Extension(basename, [basename + '.pyx'],
                                 libraries=libraries,
                                 define_macros=define_macros
                                 )]
        # 'embedsignature' compiler directive is useful to use pydoc
        #for e in ext_modules:
        #    e.pyrex_directives = {'embedsignature': True}
        setup(cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
        print 'done'
