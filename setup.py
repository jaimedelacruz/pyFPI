from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

import platform as plt
import sys
import pathlib

os.system('rm pyFPI.*.so src/python_interface.cpp')
p = pathlib.Path(sys.executable)
root_dir = str(pathlib.Path(*p.parts[0:-2]))


if(plt.system() == 'Darwin'):
    root_dir = '/opt/local/' # using this one if macports are installed
    CC = 'clang'
    CXX= 'clang++'
    link_opts = ["-bundle","-undefined","dynamic_lookup", "-fopenmp"]
    comp_flags = ["-mcpu=native", "-ffp-model=fast"]
else:
    root_dir = '/usr/'
    CC = 'gcc'
    CXX= 'g++'
    link_opts = ["-shared", "-fopenmp"]
    comp_flags = ["-march=native", "-funsafe-math-optimizations"]

os.environ["CC"] = CC
os.environ["CXX"] = CXX


# Optimization flags. 

debug = False

if(debug):
    comp_flags += ['-Os', '-g3','-fstrict-aliasing',\
                   '-std=c++20','-fPIC','-fopenmp', '-I./src',\
                   "-DNPY_NO_DEPRECATED_API", '-pedantic', '-Wall']
else:
    comp_flags += ['-O3','-g0','-fstrict-aliasing',\
                   '-std=c++20','-fPIC','-fopenmp', '-I./src', "-DNPY_NO_DEPRECATED_API",\
                   '-DNDEBUG',  "-flto"]
    


extension = Extension("pyFPI",
                      sources=["src/python_interface.pyx", "src/fpi.cpp", "src/fpi_individual.cpp", \
                               "src/invert.cpp"], 
                      include_dirs=["./",numpy.get_include(), './eigen3', root_dir+"/include/", \
                                    root_dir+'/include/eigen3/'],
                      language="c++",
                      extra_compile_args=comp_flags,
                      extra_link_args=comp_flags+link_opts,
                      library_dirs=['./',"/usr/lib/",root_dir+"/lib/"],
                      libraries=['fftw3'])

extension.cython_directives = {'language_level': "3"}

setup(
    name = 'pyFPI',
    version = '1.0',
    author = 'J. de la Cruz Rodriguez (ISP-SU 2025)',
    ext_modules=[extension],
    cmdclass = {'build_ext': build_ext}
)

