import numpy as np

data = np.loadtxt('ex1data1.txt', dtype=np.float32, delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

