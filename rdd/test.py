import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# x = np.random.uniform(-.5, .5, N)
x = np.random.normal(0, 1, N)
epsilon = np.random.normal(0, 1, N)
forcing = np.round(x+.5)
y = .5 * forcing + 2 * x + 1 + epsilon
w1 = np.random.normal(0, 1, N)
w2 = np.random.normal(0, 4, N)

data = pd.DataFrame({'y':y, 'x': x, 'w1':w1, 'w2':w2})
print(data.head())

h = rdd.optimal_bandwidth(data['y'], data['x'])
print(h)

# data_rdd = rdd.truncated_data(data, 'x', h)

# results = rdd.rdd(data_rdd, 'x', 'y')

# print(results.summary())

# data_binned = rdd.bin_data(data, 'y', 'x', 100)

# plt.figure()
# plt.scatter(data_binned['x'], data_binned['y'],
#     s = data_binned['n_obs'], facecolors='none', edgecolors='r')
# plt.show()
# plt.close()

# print(data_binned['n_obs'].describe())

# Show a spline
# show placebo with different cuts