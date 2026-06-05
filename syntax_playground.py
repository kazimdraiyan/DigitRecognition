# For learning Python, Numpy syntax

import numpy as np


w1 = np.zeros((60000, 784))
print(w1.shape)
w2 = w1[[0]]
print(w2.shape)