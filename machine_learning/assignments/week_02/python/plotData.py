import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt', dtype=np.float32, delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

plt.figure()
plt.plot(X, y, 'r+', markersize=10)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.axis([4, 20, -5, 25])
plt.show()