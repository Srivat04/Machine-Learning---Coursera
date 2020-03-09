import numpy as np
from scipy.io import loadmat
from NN import sample_nn

nn = sample_nn()

data1 = loadmat('ex3data1.mat')
X, y = data1["X"], data1["y"]

pred = nn.predict(X)

acc = np.mean(pred == y)*100
print(acc)