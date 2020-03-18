import numpy as np
from scipy.io import loadmat
from PolyRegressor import poly_reg2D
import matplotlib.pyplot as plt

data = loadmat("ex5data1.mat")

X, Xtest, Xval = data["X"], data["Xtest"], data["Xval"]
y, ytest, yval = data["y"], data["ytest"], data["yval"]

model = poly_reg2D()
model.train(X,y)
x = np.reshape(np.linspace(-80,80,161),(161,1))
pred = model.predict(x)

plt.figure(figsize = (8,6))
plt.title('Data',fontsize = 15)
plt.plot(X,y, 'rx', label = "Training Data")
plt.plot(x,pred, 'b--', label = "Polynomial Fit")
plt.xlabel("Input X", fontsize = 13)
plt.ylabel("Output y", fontsize = 13)
plt.legend()
plt.show()

index = []
train_loss = []
cross_loss = []

for i in range(2,len(X)) :

	model.train(X[:i],y[:i])
	index.append(i)
	train_loss.append(model.loss(X[:i],y[:i]))
	cross_loss.append(model.loss(Xval,yval))


plt.figure(figsize = (8,6))
plt.title("Learning Curve", fontsize = 15)
plt.plot(index, train_loss, 'r-', label = "Training loss")
plt.plot(index, cross_loss, 'b-', label = "Validation loss")
plt.ylabel("Loss", fontsize = 13)
plt.xlabel("# of training samples", fontsize = 13)
plt.legend()
plt.show()

lamdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

train_error = []
cross_error = []
for lamda in lamdas :
	model = poly_reg2D(lamda = lamda)
	model.train(X, y)
	train_error.append(model.loss(X,y))
	cross_error.append(model.loss(Xval,yval))

plt.figure(figsize = (8,6))
plt.title("Validation Curve", fontsize = 15)
plt.plot(lamdas[2:], train_loss[2:], 'r-', label = "Training loss")
plt.plot(lamdas[2:], cross_loss[2:], 'b-', label = "Validation loss")
plt.ylabel("Error", fontsize = 13)
plt.xlabel("lambda", fontsize = 13)
plt.legend()
plt.show()


