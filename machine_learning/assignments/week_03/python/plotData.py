import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex2data1.txt', dtype=np.float32, delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

pos = np.where(y == 1)
neg = np.where(y == 0)

plt.figure()
plt.plot(X[pos, 0], X[pos, 1], 'k+', markersize=7)
plt.plot(X[neg, 0], X[neg, 1], 'yo', markersize=7)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.axis([30, 100, 30, 100])
plt.show()