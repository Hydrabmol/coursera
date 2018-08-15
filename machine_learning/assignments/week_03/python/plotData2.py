import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex2data2.txt', dtype=np.float32, delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = len(y)


def plotData(X, y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.plot(X[pos,0], X[pos,1], marker='+', markersize=9, color='k')[0]
    plt.plot(X[neg,0], X[neg,1], marker='o', markersize=7, color='y')[0]
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()


plotData2(X, y)