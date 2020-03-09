import numpy as np
from scipy.io import loadmat
from LogisticRegressor import Log_reg

data1 = loadmat('ex3data1.mat')
num_labels = 10

X, y = data1["X"], data1["y"]
acc = 0
preds = np.array(y.shape)

for i in range(1,num_labels + 1) :
	
	y_i = np.array(y == i, dtype = np.uint) 
	model = Log_reg(np.append(X, y_i, axis = 1), lamda = 0.1)
	model.train()
	pred = model.predict(X)
	pred.shape = (len(y_i),1)
	acc += np.mean(pred == y_i)*100

acc /= num_labels
print(acc)
