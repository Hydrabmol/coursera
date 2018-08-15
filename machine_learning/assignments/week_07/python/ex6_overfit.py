import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import math
from scipy.io import loadmat
from sklearn import svm


raw_data = loadmat('ex6data3.mat')


X = raw_data['X']
Xval = raw_data['Xval']
y = raw_data['y']
yval = raw_data['yval']


C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_params = {'C': None, 'gamma': None}

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)

        if score > best_score:
            best_params['C'] = C
            best_params['gamma'] = gamma
            best_score = score

print best_score, best_params
        