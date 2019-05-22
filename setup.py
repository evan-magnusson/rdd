from setuptools import setup

setup(
    name='rdd',
    url='https://github.com/evan-magnusson/rdd',
    author='Evan Magnusson',
    author_email='evan.magnusson@alumni.stanford.edu',
    packages=['rdd'],
    install_requires=['pandas', 'numpy', 'statsmodels'],
    version='0.1',
    license='MIT',
    description='Tools to implement regression discontinuity designs in Python.',
    long_description=open('README.txt').read(),
)
