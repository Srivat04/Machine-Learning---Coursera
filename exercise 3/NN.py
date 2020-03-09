import numpy as np
from scipy.io import loadmat

class sample_nn :

	def __init__(self, pretrained = 'ex3weights.mat', train_data = None) :
		if pretrained :
			weights = loadmat(pretrained)
			self.W1 = weights["Theta1"]
			self.W2 = weights["Theta2"]

	def sigmoid(self, x) :

		return 1/(1+np.exp(-x))

	def predict(self, X) :

		X  = np.append(np.ones((X.shape[0],1)), X, axis = 1)
		a1 = np.append(np.ones((1,X.shape[0])),self.sigmoid(self.W1@X.T), axis = 0)

		out = self.sigmoid(self.W2@a1).T

		out = np.argmax(out, axis = 1) + 1
		out.shape = (out.shape[0],1)

		return out
