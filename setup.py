# -*- coding: utf-8 -*-
from glob import glob

from setuptools import setup, Extension

install_requires = [
    'wheel',
    'cython',
    'networkx',
    'numpy',
    'scipy',
    'matplotlib',
    'powerline-status',
    'pytest',
    'IPython',
    'pyvis',
]


def extension_modules():
    """
    Find the cython extension modules to install
    :return:
    """
    import numpy
    ext = []
    files = glob('**/*.pyx', recursive=True)
    packages = ['rgr']
    for file in files:
        if any(file.startswith(pkg) for pkg in packages):
            ext_name = file[:-4].replace('/', '.')
            source_name = './' + file
            new_extension = Extension(name=ext_name,
                                      sources=[source_name],
                                      include_dirs=[numpy.get_include()],
                                      extra_compile_args=['-ffast-math', '-march=native'],
                                      define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
            ext.append(new_extension)

            print(f'Create new Extension for: {ext_name.split(".")[-1]}')

    return ext


# Create automatically the extenstions
extensions = extension_modules()

for e in extensions:
    e.cython_directives = {'language_level': "3",  # all are Python-3
                           'embedsignature': True}

setup(name='regularity-graph-reduction',
      version='0.0.1',
      description='A Cython package to perform spectral graph operations',
      author='Anthony Gillioz',
      author_email='anthony.gillioz@outlook.com',
      install_requires=install_requires,
      setup_requires=[
          'setuptools>=18.0',  # automatically handles Cython extensions
          'cython>=0.28.4',
      ],
      ext_modules=extensions,
      )
