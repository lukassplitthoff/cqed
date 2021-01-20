from setuptools import setup, find_packages

setup(
    name='cqed',
    version='0.0.1',
    description='cQED experiments with qcodes in the TOPO group',
    author='cQED team from 12/2019',
    packages=find_packages(),
    url='https://github.com/kouwenhovenlab', install_requires=['numpy', 'pysweep', 'scipy', 'matplotlib']
)
