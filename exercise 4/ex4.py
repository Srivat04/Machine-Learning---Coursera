import numpy as np
from NN import nn_mnist
from scipy.io import loadmat

data1 = loadmat('ex4data1.mat')
X, y = data1["X"], data1["y"]

model = nn_mnist(lamda = 1)
model.train(X,y)
h = model.predict(X)

print(np.sum(h==y.T)*100/len(y))