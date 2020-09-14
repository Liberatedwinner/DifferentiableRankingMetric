from setuptools import setup
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
import numpy
import os
import platform
import glob
#python setup.py build_ext --inplace
# 하면 이 디렉토리에 설치가 됨;

def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = ['/opt/local/bin/g++-mp-[0-9].[0-9]',
                '/opt/local/bin/g++-mp-[0-9]',
                '/usr/local/bin/g++-[0-9].[0-9]',
                '/usr/local/bin/g++-[0-9]']
    if platform.system() == 'Darwin':
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()
        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            return gcc
        else:
            return None
    else:
        return None



def set_gcc():
    """Try to use GCC on OSX for OpenMP support."""
    # For macports and homebrew
    if platform.system() == 'Darwin':
        gcc = extract_gcc_binaries()

        if gcc is not None:
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc

        else:
            global use_openmp
            use_openmp = False
            logging.warning('No GCC available. Install gcc from Homebrew '
                            'using brew install gcc.')


set_gcc()


extensions = [
    Extension("rec_eval",
              sources=["rec_eval.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3", "-std=c++14"],
              extra_link_args=["-std=c++14"],
              language="c++")

]

setup(ext_modules=cythonize(extensions))
