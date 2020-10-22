import os, sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11

class get_pybind_include(object):

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

cpp_args = ['-std=c++11']

ext_modules = [
    Extension(
    'lvssa',
        ['Gillespie_lotkaVolterra.cpp'],
        include_dirs=[get_pybind_include(), 
        get_pybind_include(user=True)],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]
setup(
    name='Gillespie SSA',
    version='0.1',
    author='sanmitra ghosh',
    author_email='sanmitra.ghosh@mrc-bsu.cam.ac.uk',
    description='Gillespie SSA',
    ext_modules=ext_modules,
)
