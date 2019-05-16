import numpy as np
import pandas as pd

import functions as rdd

'''
To Do:
     - Put testing functions in another folder
     - test different input types, combos of bad items, etc
'''

# Set seed
np.random.seed(42)

# Simulate data
N = 10000
x = np.random.uniform(-.5, .5, N)
epsilon = np.random.normal(0, 1, N)
forcing = np.round(x+.5)
y = .5 * forcing + 2 * x + 1 + epsilon

data = pd.DataFrame({'y':y, 'x': x})
print(data.head())

h = rdd.optimal_bandwidth(data['y'], data['x'])
print(h)

data_rdd = rdd.truncated_data(data, 'x', h)

results = rdd.rdd(data_rdd, 'y', 'x')

print(results.summary())
