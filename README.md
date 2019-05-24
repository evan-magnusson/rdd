# rdd

rdd is a set of tools for implementing regression discontinuity designs in Python.  At present, it only allows for inputs that are pandas Series or DataFrames.  Check out the tutorial [here](https://github.com/evan-magnusson/rdd/blob/master/tutorial/tutorial.ipynb) for a guide to using this package.

## Current Features:

* Computes the Imbens-Kalyanaraman optimal bandwidth (see [this article](http://www.nber.org/papers/w14726.pdf) for details)
* Creates a dataset containing only observations within a given bandwidth
* Generates a reduced pandas DataFrame for graphical analysis (such as binned scatter plots) when the number of observations is large
* Implements a sharp RDD
  * With or without user supplied equation and/or controls
  * Via OLS or WLS

## Features to Come:

* Tutorial on how to check rdd's statistical assumptions in Python, such as using:
  * Continuity plots
  * Density plots
  * Tests for discontinuities
  * Checking for balance and testing for random assignment (though not a requirement, it can still be a useful check)
* McCrary tests
* Implementation of Fuzzy RDD
