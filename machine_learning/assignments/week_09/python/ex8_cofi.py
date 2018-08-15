import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


def cofi_cost_function(params, Y, R, num_users, num_movies, num_features, lmb):
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:num_movies*num_features+num_users*num_features].reshape(num_users, num_features)

    J = 0.5 * np.sum((np.multiply(np.dot(X, Theta.T), R) - Y)**2) + 0.5 * lmb * np.sum(X**2) + 0.5 * lmb * np.sum(Theta**2)
    X_grad = np.dot(np.multiply(np.dot(X, Theta.T), R) - Y, Theta) + lmb * X
    Theta_grad = np.dot(np.multiply(np.dot(Theta, X.T), R.T) - Y.T, X) + lmb * Theta
    grad = np.append(X_grad, Theta_grad)

    return J, grad




# Loading movies ratings dataset
data = loadmat('ex8_movies.mat')
R = data['R']
Y = data['Y']
data = loadmat('ex8_movieParams.mat')
Theta = data['Theta']
X = data['X']
num_features = data['num_features']
num_movies = data['num_movies']
num_users = data['num_users']

print("The first movie is Toy Story and its average rating is {}".format(np.mean(Y[0, R[0,:].astype(bool)])))

fig, ax = plt.subplots(figsize=(12,12))
ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')
fig.tight_layout()
#plt.show()



# Reduce dataset size so that it runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]

params = np.append(X.flatten(), Theta.flatten())
lmb = 1.5
[J, grad] = cofi_cost_function(params, Y, R, num_users, num_movies, num_features, lmb)
print("J = {}".format(J))
print("grad = {}".format(grad))

movie_idx = {}
f = open('movie_ids.txt')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

ratings = np.zeros((1682, 1))

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))
print('Rated {0} with {1} stars.'.format(movie_idx[6], str(int(ratings[6]))))
print('Rated {0} with {1} stars.'.format(movie_idx[11], str(int(ratings[11]))))
print('Rated {0} with {1} stars.'.format(movie_idx[53], str(int(ratings[53]))))
print('Rated {0} with {1} stars.'.format(movie_idx[63], str(int(ratings[63]))))
print('Rated {0} with {1} stars.'.format(movie_idx[65], str(int(ratings[65]))))
print('Rated {0} with {1} stars.'.format(movie_idx[68], str(int(ratings[68]))))
print('Rated {0} with {1} stars.'.format(movie_idx[97], str(int(ratings[97]))))
print('Rated {0} with {1} stars.'.format(movie_idx[182], str(int(ratings[182]))))
print('Rated {0} with {1} stars.'.format(movie_idx[225], str(int(ratings[225]))))
print('Rated {0} with {1} stars.'.format(movie_idx[354], str(int(ratings[354]))))


# Let's append the ratings to our data
data = loadmat('ex8_movies.mat')
R = data['R']
Y = data['Y']
R = np.append(R, ratings!=0, axis=1)
Y = np.append(Y, ratings, axis=1)

print "\n"
movies = Y.shape[0]
users = Y.shape[1]
features = 10
learning_rate = 10

X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(users, features))
params = np.append(X.flatten(), Theta.flatten())

Ymean = np.zeros((movies, 1))
Ynorm = np.zeros((movies, users))

for i in range(movies):
    idx = np.where(R[i,:]==1)[0]
    Ymean[i] = Y[i,idx].mean()
    Ynorm[i, idx] = Y[i, idx] - Ymean[i]


fmin = minimize(fun=cofi_cost_function, x0=params, args=(Ynorm, R, users, movies, features, learning_rate),
                method='CG', jac=True, options={'maxiter': 100})

X = fmin.x[0:movies*features].reshape(movies, features)
Theta = fmin.x[movies*features : movies*features + users*features].reshape(users, features)

predictions = np.dot(X, Theta.T)
my_preds = predictions[:, -1] + Ymean.reshape(Ymean.shape[0])
sorted_preds = np.sort(my_preds, axis=0)[::-1]
sorted_preds[:10]
print(sorted_preds)
print('\n')

idx = np.argsort(my_preds, axis=0)[::-1]
print("Top 10 movies predictions")
for i in range(10):
    j = int(idx[i])
    print("Predicted rating of {} for movie {}".format(my_preds[j], movie_idx[j]))




