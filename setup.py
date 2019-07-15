#!/usr/bin/env python

from setuptools import setup, find_packages

def _read(fname):
    try:
        with open(fname) as fobj:
            return fobj.read()

    except IOError:
        return ''


setup(
    name='approximate_dictionary',
    version='0.2.0',
    description='Dictionary with approximate search',
    long_description=_read("Readme.md"),
    author='Matthias Ossadnik',
    author_email='ossadnik.matthias@gmail.com',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url="https://github.com/mossadnik/approximate-dictionary.git",
    setup_requires=['pytest-runner'],
    install_requires=[
        'numpy>=1.16',
        'numba>=0.44',
    ],
    tests_require=[
        'pytest',
        'python-levenshtein'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Indexing',
    ],
)
