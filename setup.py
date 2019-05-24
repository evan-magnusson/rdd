import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rdd',
    url='https://github.com/evan-magnusson/rdd',
    author='Evan Magnusson',
    author_email='evan.magnusson@alumni.stanford.edu',
    packages=setuptools.find_packages(),
    install_requires=['pandas', 'numpy', 'statsmodels'],
    version='0.0.3',
    license='MIT',
    description='Tools to implement regression discontinuity designs in Python.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
