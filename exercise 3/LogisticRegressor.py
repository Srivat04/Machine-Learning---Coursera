import numpy as np
import scipy.optimize as opt

class Log_reg :

	def __init__(self, train_data, lamda = 0) :
		self.train_data = train_data
		self.m, self.n = self.train_data.shape
		self.lamda = lamda
		self.X = np.append(np.ones((self.m,1)), train_data[:,:-1], axis = 1)
		self.y = train_data[:,-1]
		self.theta = np.zeros(self.n)

	def sigmoid(self, x) :

		return 1/(1+np.exp(-x)) 

	def loss(self,theta, X, y) :

		m = self.m
		sigmoid = self.sigmoid

		J	 = np.average((np.log(sigmoid(X@theta)))*y-np.log(1-sigmoid(X@(theta)))*(1-y)) + self.lamda/(2*m)*np.sum(theta*theta)
		grad = np.concatenate(((1/m)*(sigmoid(X.dot(theta))-y).T.dot(X[:,0]), (1/m)*(sigmoid(X.dot(theta))-y).T.dot(X[:,1:]) + self.lamda/m*theta[1:]), axis = None)

		return J, grad

	def train (self, iter = 400) :
		
		options = {'maxiter' : iter}
		result = opt.minimize(self.loss, self.theta, args = (self.X,self.y), jac = True, method = "TNC", options = options)
		self.theta = result.x

	def predict (self, X) :

		X = np.append(np.ones((self.m,1)), X, axis = 1)
		h = self.sigmoid(X.dot(self.theta))
		h[h >= 0.5] = 1
		h[h < 1] = 0

		return h

# Clean up the cost fucntion part of the code using * and @ operators
