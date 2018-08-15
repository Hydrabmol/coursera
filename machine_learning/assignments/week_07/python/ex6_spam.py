import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import math
from scipy.io import loadmat
from sklearn import svm


spam_train = loadmat('spamTrain.mat')
spam_test = loadmat('spamTest.mat')

X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y']
ytest = spam_test['ytest']

svc = svm.SVC(C=1, kernel='linear')
svc.fit(X, y.ravel())
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))
