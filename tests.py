import numpy as np
import pandas as pd

from rdd import rdd
'''
To Do:
     - test different input types, combos of bad items, etc
'''

# Set seed
np.random.seed(42)
# Simulate data
N = 10000
x = np.random.normal(1, 1, N)
epsilon = np.random.normal(0, 1, N)
forcing = np.where(x >= 1, 1, 0)
y = .5 * forcing + 2 * x + 1 + epsilon
w1 = np.random.normal(0, 1, N)
w2 = np.random.normal(0, 4, N)

data = pd.DataFrame({'y':y, 'x': x, 'w1':w1, 'w2':w2})

# TEST optimal_bandwidth()
print("optimal_bandwidth() tests:")
flag_optimal_bandwidth = 0

h = rdd.optimal_bandwidth(data['y'], data['x'], 1)
if np.round(h, 5)!=.75117:
    print("\tFAIL: value of h is wrong")
    flag_optimal_bandwidth = 1
if flag_optimal_bandwidth==0:
    print("\tNo Failures")

# TEST truncated_data()

# data_rdd = rdd.truncated_data(data, 'x', h)

# TEST rdd()

# model = rdd.rdd(data_rdd, 'x', 'y')

# print(results.fit().summary())


# TEST bin_data()

# data_binned = rdd.bin_data(data, 'y', 'x', 100)

# plt.figure()
# plt.scatter(data_binned['x'], data_binned['y'],
#     s = data_binned['n_obs'], facecolors='none', edgecolors='r')
# plt.show()
# plt.close()

# print(data_binned['n_obs'].describe())
