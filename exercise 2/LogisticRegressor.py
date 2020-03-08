import numpy as np
import scipy.optimize as opt
from FeatureMap import map_feature
class Log_reg :

	def __init__(self, train_data, poly_features = 1, lamda = 0) :
		self.train_data = train_data
		self.lamda = lamda
		self.X = map_feature(train_data[:,:-1], poly_features)
		self.y = train_data[:,-1]
		self.m, self.n = self.X.shape
		self.theta = np.zeros((self.X.shape[1]))
		self.poly_features = poly_features

	def sigmoid(self, x) :

		return 1/(1+np.exp(-x)) 

	def gradient(self, theta, X, y) :

		m = X.shape[0]
		sigmoid = self.sigmoid
		
		grad = np.concatenate(((1/m)*(sigmoid(X.dot(theta))-y).T.dot(X[:,0]), (1/m)*(sigmoid(X.dot(theta))-y).T.dot(X[:,1:]) + self.lamda/m*theta[1:]), axis = None)
		
		return grad

	def loss(self,theta, X, y) :

		m = X.shape[0]
		sigmoid = self.sigmoid

		J =  1/m*(-y.T.dot(np.log(sigmoid(X.dot(theta))))-(1-np.array(y)).T.dot(np.log(1-sigmoid(X.dot(theta))))) + self.lamda/(2*m)*theta.T.dot(theta)
		grad = self.gradient(theta, X, y)

		return J, grad

	def train (self, iter = 400) :
		options = {'maxiter' : iter}
		result = opt.minimize(self.loss, self.theta, args = (self.X,self.y), jac = True, method = "TNC", options = options)
		self.theta = result.x

	def predict (self, X) :

		X = map_feature(X, self.poly_features)
		h = self.sigmoid(X.dot(self.theta))
		h[h >= 0.5] = 1
		h[h < 1] = 0

		return h

